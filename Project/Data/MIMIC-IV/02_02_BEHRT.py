from __future__ import annotations
import math, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
from scipy.special import expit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

BATCH_SIZE     = 16
MAX_EPOCHS     = 50
EARLY_PATIENCE = 5
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 1e-2
RUN_FAIRNESS   = True
SENS_COLS = ('age_bucket', 'ethnicity_cat', 'race_cat', 'insurance_cat')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class StructDS(Dataset):
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        self.sens = {c: df[c].astype(str).values for c in SENS_COLS}
        self.y = df['readmit_30d'].astype(np.float32).values

        # Feature detection
        ignore_cols = set(['subject_id', 'hadm_id', 'readmit_30d', 'gender', *SENS_COLS])
        feature_prefixes = ('chartevents_', 'labevents_', 'inputevents_', 'outputevents_')
        self.lab_cols = [
            c for c in df.columns if c not in ignore_cols and c.startswith(feature_prefixes)
        ]
        if not self.lab_cols:
            self.lab_cols = [
                c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
        assert self.lab_cols, "No lab/feature columns detected!"

        df[self.lab_cols] = df[self.lab_cols].apply(
            lambda s: (s - s.mean()) / s.std() if s.std() else 0.
        ).replace([np.inf, -np.inf], 0.).fillna(0.).astype(np.float32)
        self.X = df[self.lab_cols].values.astype(np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

class Encoder(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 768, heads: int = 12, layers: int = 6, ff_size: int = 3072):
        super().__init__()
        self.token = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))
        block = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=ff_size, dropout=0.1, batch_first=True,
            activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(block, layers)
    def forward(self, x):  # x: [B, seq_len]
        h = self.token(x.unsqueeze(-1)) + self.pos
        return self.enc(h).mean(1)   # [B, d_model]

class BEHRTReadmit(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 768):
        super().__init__()
        self.encoder = Encoder(seq_len, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(0.1), nn.GELU(),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):      # x: [B, seq_len]
        return self.head(self.encoder(x)).squeeze(1)

def class_weight(y):
    pos, neg = (y == 1).sum(), (y == 0).sum()
    return neg / pos if pos else 1.0

def train(model, tr_loader, val_loader, device):
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, 'min', factor=0.1, patience=2, verbose=False)
    y_all = torch.cat([y for x, y in tr_loader])
    loss_f = FocalLoss(gamma=2.5, pos_weight=torch.tensor(class_weight(y_all.numpy()), device=device))
    best, wait = math.inf, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); tr_losses = []
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_f(model(x), y)
            if torch.isnan(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float('nan')
        model.eval(); val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                loss = loss_f(model(x.to(device)), y.to(device)).item()
                val_losses.append(loss)
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")
        if math.isnan(val_loss): print("   ↳ validation loss is NaN – stopping"); break
        if val_loss < best:
            best, wait = val_loss, 0
            torch.save(model.state_dict(), 'best_behrt_readmit.pt')
            print("   ↳ best model saved")
        else:
            wait += 1
            if wait >= EARLY_PATIENCE:
                print("   ↳ early stopping"); break
        sched.step(val_loss)

def evaluate(model, loader, device):
    model.eval(); logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    logits = torch.cat(logits).numpy()
    labels = torch.cat(labels).numpy()
    probs = expit(logits)
    preds = (probs > 0.5).astype(int)
    pr, rc, _ = precision_recall_curve(labels, logits)
    return dict(
        auroc = roc_auc_score(labels, logits),
        auprc = auc(rc, pr),
        precision = precision_score(labels, preds, zero_division=0),
        recall = recall_score(labels, preds),
        f1 = f1_score(labels, preds)
    ), logits, labels, preds

def _tpr_fpr(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return (tp / (tp + fn) if tp + fn else 0.0,
            fp / (fp + tn) if fp + tn else 0.0)

def equalised_odds_gap(attr, y_true, y_pred):
    tpr_d, fpr_d = {}, {}
    for g in np.unique(attr):
        m = attr == g
        tpr_d[g], fpr_d[g] = _tpr_fpr(y_true[m], y_pred[m])
    groups, n = list(tpr_d), len(tpr_d)
    if n < 2: return 0.0
    t_gap = sum(abs(tpr_d[g1] - tpr_d[g2]) for i, g1 in enumerate(groups) for g2 in groups[i + 1:])
    f_gap = sum(abs(fpr_d[g1] - fpr_d[g2]) for i, g1 in enumerate(groups) for g2 in groups[i + 1:])
    return (t_gap + f_gap) / (n ** 2)

def eddi(attr, y_true, y_score, thr=0.5):
    y_pred = (y_score > thr).astype(int)
    err_all = np.mean(y_pred != y_true)
    denom = max(err_all, 1 - err_all) if err_all not in (0, 1) else 1.0
    sub = {}
    for g in np.unique(attr):
        m = attr == g
        err_g = np.mean(y_pred[m] != y_true[m]) if m.any() else np.nan
        sub[g] = (err_g - err_all) / denom
    overall = np.sqrt(np.nanmean(np.square(list(sub.values()))))
    return overall, sub

def detailed_eddi(y_true, y_pred, probs, sens_test, orders):
    eddi_vals = []
    print("\n--- Detailed EDDI by Sensitive Attribute ---")
    for col in SENS_COLS:
        overall, sub = eddi(sens_test[col], y_true, probs)
        eddi_vals.append(overall)
        print(f"\n{col}:")
        for group in orders[col]:
            print(f"  {group}: {sub.get(group, np.nan):.4f}")
        print(f"Overall {col} EDDI: {overall:.4f}")
    total_eddi = np.sqrt(np.sum(np.array(eddi_vals) ** 2) / len(eddi_vals))
    print(f"\nTOTAL EDDI (RMS across all attributes): {total_eddi:.4f}")
    return total_eddi

def encode_for_strat(df, sens_cols, target_col):
    strat_cols = []
    for c in sens_cols:
        strat_cols.append(pd.Categorical(df[c]).codes)
    strat_cols.append(df[target_col].astype(int).values)
    return np.vstack(strat_cols).T

def main():
    torch.manual_seed(42); np.random.seed(42)
    csv = Path("structured_common_subjects.csv")  
    if not csv.exists():
        raise SystemExit(f"{csv} not found – run the pre-processing first.")
    ds = StructDS(csv)
    df = pd.read_csv(csv)
    Y_strat = encode_for_strat(df, SENS_COLS, target_col='readmit_30d')

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trv_idx, test_idx = next(msss.split(np.zeros(len(Y_strat)), Y_strat))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05/0.80, random_state=42)
    tr_idx, val_idx = next(msss2.split(np.zeros(len(trv_idx)), Y_strat[trv_idx]))

    print(f"Split → train {len(tr_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    mk = lambda ids, sh=False: DataLoader(Subset(ds, ids), batch_size=BATCH_SIZE, shuffle=sh, drop_last=False)
    tr_loader, val_loader, test_loader = mk(tr_idx, True), mk(val_idx), mk(test_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model = BEHRTReadmit(seq_len=len(ds.lab_cols), d_model=768).to(device)
    train(model, tr_loader, val_loader, device)
    model.load_state_dict(torch.load('best_behrt_readmit.pt', map_location=device))
    metrics, logits, labels, preds = evaluate(model, test_loader, device)
    print("\nTEST METRICS")
    print("AUROC  AUPRC  F1   Precision  Recall")
    print("{auroc:.4f} {auprc:.4f} {f1:.4f} {precision:.4f} {recall:.4f}".format(**metrics))

    if RUN_FAIRNESS:
        probs = expit(logits)
        sens_test = {k: v[test_idx] for k, v in ds.sens.items()}
        # Get sorted group orders from your test set values
        orders = {
            'age_bucket': sorted(set(df['age_bucket'])),
            'ethnicity_cat': sorted(set(df['ethnicity_cat'])),
            'race_cat': sorted(set(df['race_cat'])),
            'insurance_cat': sorted(set(df['insurance_cat']))
        }
        detailed_eddi(labels, preds, probs, sens_test, orders)
        print("\nFAIRNESS – Equalised-Odds Δ and EDDI (shorthand):")
        for col, v in sens_test.items():
            eo  = equalised_odds_gap(v, labels, preds)
            edd, sub = eddi(v, labels, probs)
            print(f"\n▪ {col}")
            print(f"  ΔEO  : {eo:.4f}")
            print(f"  EDDI : {edd:.4f}")
            for g, d in sub.items():
                print(f"     {g:<12} {d:+.4f}")

if __name__ == "__main__":
    main()
