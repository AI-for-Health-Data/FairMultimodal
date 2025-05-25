import os, math, torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
from scipy.special import expit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

BATCH_SIZE     = 16
MAX_EPOCHS     = 50
EARLY_PATIENCE = 5
LEARNING_RATE  = 2e-4
WEIGHT_DECAY   = 1e-2

SENS_COLS = ('age_bucket', 'ethnicity', 'race', 'insurance')
OUTCOME_COLS = ['mortality', 'pe', 'ph']

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
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        self.sens = {c: df[c].astype(str).values for c in SENS_COLS}
        self.y = np.vstack([df[o].astype(np.float32).values for o in OUTCOME_COLS]).T 
        ignore_cols = set(['subject_id', 'hadm_id'] + list(SENS_COLS) + OUTCOME_COLS)
        self.lab_cols = [
            c for c in df.columns
            if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        df[self.lab_cols] = df[self.lab_cols].apply(
            lambda s: (s - s.mean()) / s.std() if s.std() else 0.
        ).replace([np.inf, -np.inf], 0.).fillna(0.).astype(np.float32)
        self.X = df[self.lab_cols].values.astype(np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class BEHRTStruct(nn.Module):
    def __init__(self, seq_len, d_model=768, nhead=8, nlayers=2):
        super().__init__()
        self.token = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(0.1), nn.GELU(), nn.Linear(d_model, 1)
        ) for _ in OUTCOME_COLS])

    def forward(self, x):
        h = self.token(x.unsqueeze(-1)) + self.pos  
        h = self.transformer(h)
        pooled = h.mean(1)
        return [head(pooled).squeeze(1) for head in self.heads]

def class_weight(y):
    weights = []
    for k in range(y.shape[1]):
        pos = (y[:, k] == 1).sum()
        neg = (y[:, k] == 0).sum()
        weights.append(neg / max(pos, 1))
    return weights

def train(model, tr_loader, val_loader, device, class_weights):
    # Set custom gamma for each outcome
    gammas = [2.0, 5.0, 3.0]  # [mortality, PE, PH]
    loss_fs = [
        FocalLoss(gamma=gammas[k], pos_weight=torch.tensor(class_weights[k], device=device))
        for k in range(len(OUTCOME_COLS))
    ]
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, 'min', patience=2, verbose=False)
    best, wait = math.inf, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); tr_losses = []
        for X, Y in tr_loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = sum(loss_fs[k](logits[k], Y[:, k]) for k in range(len(OUTCOME_COLS)))
            if torch.isnan(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float('nan')
        model.eval(); val_losses = []
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                logits = model(X)
                loss = sum(loss_fs[k](logits[k], Y[:, k]) for k in range(len(OUTCOME_COLS)))
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best:
            best, wait = val_loss, 0
            torch.save(model.state_dict(), 'best_behrt.pt')
            print("   ↳ best model saved")
        else:
            wait += 1
            if wait >= EARLY_PATIENCE:
                print("   ↳ early stopping"); break
        sched.step(val_loss)


def evaluate(model, loader, device):
    model.eval()
    all_logits = [ [] for _ in OUTCOME_COLS ]
    all_labels = [ [] for _ in OUTCOME_COLS ]
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            for k in range(len(OUTCOME_COLS)):
                all_logits[k].append(logits[k].cpu())
                all_labels[k].append(Y[:,k].cpu())
    logits = [ torch.cat(l).numpy() for l in all_logits ]
    labels = [ torch.cat(l).numpy() for l in all_labels ]
    probs = [ expit(logit) for logit in logits ]
    preds = [ (p > 0.5).astype(int) for p in probs ]
    metrics = {}
    for k, outcome in enumerate(OUTCOME_COLS):
        pr, rc, _ = precision_recall_curve(labels[k], logits[k])
        metrics[outcome] = {
            "auroc": roc_auc_score(labels[k], logits[k]),
            "auprc": auc(rc, pr),
            "f1": f1_score(labels[k], preds[k]),
            "precision": precision_score(labels[k], preds[k]),
            "recall": recall_score(labels[k], preds[k]),
        }
    return metrics, logits, labels, preds, probs

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

def detailed_eddi(y_true, probs, sens_test, sens_cols, outcome_idx, outcome_name):
    print(f"\n===== EDDI for {outcome_name} =====")
    eddi_vals = []
    for col in sens_cols:
        overall, sub = eddi(sens_test[col], y_true, probs)
        eddi_vals.append(overall)
        print(f"{col}:")
        for group, val in sub.items():
            print(f"  {group:<12}: {val:+.4f}")
        print(f"  [Total {col} EDDI: {overall:.4f}]")
    total_eddi = np.sqrt(np.mean(np.array(eddi_vals) ** 2))
    print(f"  TOTAL EDDI (across all attributes): {total_eddi:.4f}")
    return total_eddi

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    csv = Path("final_structured.csv")
    assert csv.exists(), "Data file missing!"
    ds = StructDS(csv)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trv_idx, test_idx = next(msss.split(np.zeros(len(ds)), ds.y))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05 / 0.80, random_state=42)
    tr_idx, val_idx = next(msss2.split(np.zeros(len(trv_idx)), ds.y[trv_idx]))

    print(f"Split → train {len(tr_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    mk = lambda ids, sh=False: DataLoader(Subset(ds, ids), batch_size=BATCH_SIZE, shuffle=sh, drop_last=False)
    tr_loader = mk(tr_idx, True)
    val_loader = mk(val_idx)
    test_loader = mk(test_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model = BEHRTStruct(seq_len=len(ds.lab_cols), d_model=768).to(device)
    class_weights = class_weight(ds.y[tr_idx])
    train(model, tr_loader, val_loader, device, class_weights)
    model.load_state_dict(torch.load('best_behrt.pt', map_location=device))

    metrics, logits, labels, preds, probs = evaluate(model, test_loader, device)
    print("\n==== EVALUATION METRICS ====")
    for outcome in OUTCOME_COLS:
        print(f"{outcome.upper()}:")
        for k, v in metrics[outcome].items():
            print(f"  {k:10}: {v:.4f}")

    # --- EDDI for each outcome
    sens_test = {k: v[test_idx] for k, v in ds.sens.items()}
    total_eddis = []
    for i, outcome in enumerate(OUTCOME_COLS):
        total_eddis.append(detailed_eddi(labels[i], probs[i], sens_test, SENS_COLS, i, outcome))
    print("\n==== TOTAL EDDI (all outcomes, RMS) ====")
    print("  OVERALL:", np.sqrt(np.mean(np.array(total_eddis)**2)))

if __name__ == "__main__":
    main()
