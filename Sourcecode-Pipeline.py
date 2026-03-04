import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

# Reproducibility

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1) Multi-season feature builder (Elo/Form/Rest/H2H)

def prepare_multi(seasons=("2324","2425"), init_elo=1500, k=20, use_odds=True, devig=True):
    # load & concat in chronological order
    frames = []
    for sc in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{sc}/E0.csv"
        d = pd.read_csv(url)                               
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        d["Season"] = sc
        frames.append(d)

    df = pd.concat(frames, ignore_index=True).sort_values("Date", ignore_index=True)

    # keep only rows with a known result (avoid NaNs in y / Elo / H2H)
    df = df[df["FTR"].isin(["H","D","A"])].copy()

    # target
    df["y"] = df["FTR"].map({"H":2,"D":1,"A":0}).astype("int8")

    # rolling form (use shift to avoid leakage)
    df["HomePts"] = df["FTR"].map({"H":3,"D":1,"A":0})
    df["AwayPts"] = df["FTR"].map({"H":0,"D":1,"A":3})
    df["HomeForm"] = (
        df.groupby("HomeTeam")["HomePts"]
          .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
          .fillna(0)
    )
    df["AwayForm"] = (
        df.groupby("AwayTeam")["AwayPts"]
          .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
          .fillna(0)
    )

    # rest days
    last = {}
    rest_h, rest_a = [], []
    for _, r in df.iterrows():
        for col, team in [("h", r["HomeTeam"]), ("a", r["AwayTeam"])]:
            prev = last.get(team, pd.NaT)
            delta = (r["Date"] - prev).days if pd.notna(prev) else np.nan
            (rest_h if col=="h" else rest_a).append(delta)
            last[team] = r["Date"]
    df["RestHome"], df["RestAway"] = rest_h, rest_a
    df["RestHome"] = df["RestHome"].fillna(7)
    df["RestAway"] = df["RestAway"].fillna(7)

    # head-to-head last 5 (shifted)
    from collections import defaultdict, Counter
    h2h = defaultdict(list)
    h2h_feats = []
    for _, r in df.iterrows():
        key = (r["HomeTeam"], r["AwayTeam"])
        past = h2h[key][-5:]
        if past:
            c = Counter(past); tot = len(past)
            h2h_feats.append([c[2]/tot, c[1]/tot, c[0]/tot])
        else:
            h2h_feats.append([0.0, 0.0, 0.0])
        h2h[key].append(int(r["y"]))
    h2h_df = pd.DataFrame(h2h_feats, columns=["H2H_H","H2H_D","H2H_A"], index=df.index)
    df = pd.concat([df, h2h_df], axis=1)

    # Elo across entire timeline (skip updates safely if needed)
    from collections import defaultdict as dd
    elo = dd(lambda: init_elo)
    elodiffs = []
    for _, r in df.iterrows():
        h, a, res = r["HomeTeam"], r["AwayTeam"], int(r["y"])
        elodiffs.append(elo[h] - elo[a])
        exp_h   = 1.0 / (1.0 + 10 ** ((elo[a] - elo[h]) / 400))
        score_h = {2:1.0, 1:0.5, 0:0.0}[res]
        elo[h] += k * (score_h - exp_h)
        elo[a] += k * ((1 - score_h) - (1 - exp_h))
    df["EloDiff"] = elodiffs

    # odds → features (optional), convert to implied probs & de-vig (optional)
    odd_cols = []
    if use_odds:
        candidate = ["B365H","B365D","B365A"]
        odd_cols = [c for c in candidate if c in df.columns]
        if devig and len(odd_cols) == 3:
            # guard against zeros / missing
            inv = 1.0 / df[odd_cols].replace(0, np.nan)
            over = inv.sum(axis=1)
            probs = inv.div(over, axis=0)
            probs.columns = ["IMP_H","IMP_D","IMP_A"]
            df = pd.concat([df, probs], axis=1)
            odd_cols = ["IMP_H","IMP_D","IMP_A"]

    # final feature list
    base_feats = ["EloDiff","HomeForm","AwayForm","RestHome","RestAway","H2H_H","H2H_D","H2H_A"]
    features = base_feats + odd_cols

    # drop rows with missing selected features or target
    df = df.dropna(subset=features + ["y"]).reset_index(drop=True)

    # train = all seasons except last test
    last_season = seasons[-1]
    train_df = df[df["Season"] != last_season].copy()
    test_df  = df[df["Season"] == last_season].copy()

    X_train, y_train = train_df[features], train_df["y"]
    X_test,  y_test  = test_df[features],  test_df["y"]
    return train_df, test_df, X_train, X_test, y_train, y_test, features


SEASONS_ALL = (
    "1415","1516","1617","1718","1819",
    "1920","2021","2122","2223","2324",
    "2425"
)

train_df, test_df, X_train, X_test, y_train, y_test, FEATURES = prepare_multi(
    seasons=SEASONS_ALL,
    init_elo=1500, k=20,
    use_odds=True,
    devig=True
)


# 2) Teams + Scaling

all_teams = sorted(set(train_df['HomeTeam'])|set(train_df['AwayTeam'])|
                   set(test_df['HomeTeam']) |set(test_df['AwayTeam']))
team2idx = {t:i for i,t in enumerate(all_teams)}

def map_idx(df):
    return (
        torch.tensor(df['HomeTeam'].map(team2idx).values, dtype=torch.long),
        torch.tensor(df['AwayTeam'].map(team2idx).values, dtype=torch.long)
    )

home_idx_tr, away_idx_tr = map_idx(train_df)
home_idx_te, away_idx_te = map_idx(test_df)

scaler = StandardScaler().fit(X_train)
X_tr = scaler.transform(X_train)
X_te = scaler.transform(X_test)

# 3) Chronological val split

cutoff = int(0.8 * len(train_df))  # 80% train, 20% val in time order
tr_idx = np.arange(cutoff)
va_idx = np.arange(cutoff, len(train_df))

class FootyDS(Dataset):
    def __init__(self, X, h_idx, a_idx, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.h = h_idx
        self.a = a_idx
        self.y = torch.tensor(np.array(y), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.h[i], self.a[i], self.y[i]

cutoff = int(0.8 * len(train_df))  # 80% train, 20% val in time order
tr_idx_np = np.arange(cutoff)
va_idx_np = np.arange(cutoff, len(train_df))

# convert to torch long indices for tensor indexing
tr_idx_t = torch.tensor(tr_idx_np, dtype=torch.long)
va_idx_t = torch.tensor(va_idx_np, dtype=torch.long)

train_ds = FootyDS(
    X_tr[tr_idx_np],
    home_idx_tr[tr_idx_t],
    away_idx_tr[tr_idx_t],
    y_train.iloc[tr_idx_np],
)
val_ds = FootyDS(
    X_tr[va_idx_np],
    home_idx_tr[va_idx_t],
    away_idx_tr[va_idx_t],
    y_train.iloc[va_idx_np],
)

test_ds  = FootyDS(X_te,        home_idx_te,         away_idx_te,         y_test)

# Weighted sampler
from collections import Counter
counts_tr = Counter(y_train.iloc[tr_idx].values)
w_per_cls = {cls: 1.0/counts_tr[cls] for cls in counts_tr}
sample_weights = [w_per_cls[int(lbl)] for lbl in y_train.iloc[tr_idx]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)


# 4) Residual MLP + embeddings

class Block(nn.Module):
    def __init__(self, d, dp=0.3):
        super().__init__()
        self.lin1 = nn.Linear(d,d); self.ln1 = nn.LayerNorm(d)
        self.lin2 = nn.Linear(d,d); self.ln2 = nn.LayerNorm(d)
        self.dp = nn.Dropout(dp)
    def forward(self, x):
        h = self.dp(F.gelu(self.ln1(self.lin1(x))))
        h = self.dp(F.gelu(self.ln2(self.lin2(h))))
        return x + h

class DeepNet(nn.Module):
    def __init__(self, n_num, n_teams, emb=8, widths=[256,128,64], dp=0.3):
        super().__init__()
        self.he = nn.Embedding(n_teams, emb)
        self.ae = nn.Embedding(n_teams, emb)
        dims = [n_num + 2*emb] + widths
        self.inp = nn.Linear(dims[0], dims[1])
        self.blocks = nn.ModuleList([Block(dims[i], dp) for i in range(1,len(dims))])
        self.downs  = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(1,len(dims)-1)])
        self.out = nn.Linear(dims[-1], 3)
    def forward(self, x, h, a):
        x = torch.cat([x, self.he(h), self.ae(a)], dim=1)
        x = F.gelu(self.inp(x))
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i < len(self.downs):
                x = F.gelu(self.downs[i](x))
        return self.out(x)

model = DeepNet(n_num=X_tr.shape[1], n_teams=len(team2idx), emb=8, widths=[256,128,64], dp=0.3).to(DEVICE)

# 5) Loss, Opt, Scheduler

loss_fn = nn.CrossEntropyLoss()
opt     = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)

# 6) Train + Early Stopping

best_val = float('inf')
wait, patience = 0, 5

def run_epoch(dl, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_n = 0.0, 0
    preds, trues = [], []
    with torch.set_grad_enabled(train):
        for xb, hb, ab, yb in dl:
            xb, hb, ab, yb = xb.to(DEVICE), hb.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
            if train:
                opt.zero_grad()
            logits = model(xb, hb, ab)
            loss = loss_fn(logits, yb)
            if train:
                loss.backward()
                opt.step()
            total_loss += loss.item() * yb.size(0)
            preds.append(logits.argmax(1).detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
            total_n += yb.size(0)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return total_loss/total_n, accuracy_score(y_true, y_pred)

for epoch in range(1, 51):
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    va_loss, va_acc = run_epoch(val_dl,   train=False)
    print(f"Epoch {epoch:02d} — train_loss:{tr_loss:.4f} — val_loss:{va_loss:.4f} — val_acc:{va_acc:.2%}")
    sched.step(va_loss)

    if va_loss < best_val:
        best_val = va_loss
        wait = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

model.load_state_dict(torch.load("best.pt", map_location=DEVICE))


# 7) Temperature scaling (calibration) on VAL

class Temperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))  # T = exp(logT), init 1.0
    def forward(self, logits):
        T = torch.exp(self.logT)
        return logits / T

def collect_logits_targets(dl):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, hb, ab, yb in dl:
            xb, hb, ab = xb.to(DEVICE), hb.to(DEVICE), ab.to(DEVICE)
            logits = model(xb, hb, ab)
            all_logits.append(logits.cpu())
            all_y.append(yb)
    return torch.cat(all_logits,0), torch.cat(all_y,0)

temp = Temperature().to(DEVICE)
optim_T = optim.LBFGS(temp.parameters(), lr=0.5, max_iter=50)

val_logits, val_y = collect_logits_targets(val_dl)
val_logits = val_logits.to(DEVICE)
val_y = val_y.to(DEVICE)

def _closure():
    optim_T.zero_grad()
    scaled = temp(val_logits)
    loss = F.cross_entropy(scaled, val_y)
    loss.backward()
    return loss

optim_T.step(_closure)
with torch.no_grad():
    print("Calibrated T =", float(torch.exp(temp.logT).cpu().numpy()))

def apply_temperature(logits):
    temp.eval()
    with torch.no_grad():
        return temp(logits)


# 8) Metrics

def brier_multi(probs, y_true, n_classes=3):
    Y = np.eye(n_classes)[y_true]
    return np.mean(np.sum((probs - Y)**2, axis=1))

def expected_calibration_error(probs, y_true, n_bins=15):
    conf = probs.max(1)
    preds = probs.argmax(1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc = (preds[mask] == y_true[mask]).mean()
            avg_conf = conf[mask].mean()
            ece += (mask.mean()) * abs(avg_conf - acc)
    return float(ece)


# 9) Final Test Eval + result table

model.eval()
probs_list, preds_list = [], []
with torch.no_grad():
    for xb, hb, ab, yb in test_dl:
        xb, hb, ab = xb.to(DEVICE), hb.to(DEVICE), ab.to(DEVICE)
        logits = model(xb, hb, ab)
        logits = apply_temperature(logits)  # calibrated
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(p)
        preds_list.append(p.argmax(1))

probs_all = np.vstack(probs_list)
preds_all = np.concatenate(preds_list)
y_true = np.array(y_test)

# metrics
test_logloss = log_loss(y_true, probs_all, labels=[0,1,2])
test_acc = accuracy_score(y_true, preds_all)
test_f1_macro = f1_score(y_true, preds_all, average='macro')
test_brier = brier_multi(probs_all, y_true, n_classes=3)
test_ece = expected_calibration_error(probs_all, y_true, n_bins=15)

print("\n=== TEST METRICS (calibrated) ===")
print(f"Log loss:      {test_logloss:.4f}")
print(f"Accuracy:      {test_acc:.4f}")
print(f"Macro F1:      {test_f1_macro:.4f}")
print(f"Brier score:   {test_brier:.4f}")
print(f"ECE (15 bins): {test_ece:.4f}")

# Build output table
out = test_df.reset_index(drop=True)[["Date","HomeTeam","AwayTeam","FTR"]].copy()
out["Actual"]     = out["FTR"].map({"H":"Home","D":"Draw","A":"Away"})
label_map         = {2:"Home", 1:"Draw", 0:"Away"}
out["Predicted"]  = [label_map[p] for p in preds_all]
out["P(Home win)"] = probs_all[:,2]
out["P(Draw)"]     = probs_all[:,1]
out["P(Away win)"] = probs_all[:,0]

# format as percentages
for col in ["P(Home win)", "P(Draw)", "P(Away win)"]:
    out[col] = (out[col] * 100).map("{:.1f}%".format)

# print full table
pd.set_option("display.max_rows", None)
print("\n=== Predictions with probabilities (calibrated) ===")
print(out)
pd.reset_option("display.max_rows")

# save CSV
out.to_csv("predictions_with_probs.csv", index=False)

# also dump metrics to a small txt
with open("test_metrics.txt","w") as f:
    f.write(f"LogLoss,{test_logloss:.6f}\n")
    f.write(f"Accuracy,{test_acc:.6f}\n")
    f.write(f"MacroF1,{test_f1_macro:.6f}\n")
    f.write(f"Brier,{test_brier:.6f}\n")
    f.write(f"ECE,{test_ece:.6f}\n")

print("\nSaved:")
print(" - predictions_with_probs.csv")
print(" - test_metrics.txt")






