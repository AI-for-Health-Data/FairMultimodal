from __future__ import annotations
print(" BEHRT-structured – code loaded")

import math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)
from scipy.special import expit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def calculate_tpr_and_fpr(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr

def calculate_equalized_odds_difference(tpr_dict, fpr_dict):
    groups = list(tpr_dict.keys())
    n = len(groups)
    if n == 0:
        return {'EOTPR': 0.0, 'EOFPR': 0.0, 'EO': 0.0}
    tpr_diff_sum = 0.0
    fpr_diff_sum = 0.0
    for i in range(n):
        for j in range(i+1, n):
            tpr_diff_sum += abs(tpr_dict[groups[i]] - tpr_dict[groups[j]])
            fpr_diff_sum += abs(fpr_dict[groups[i]] - fpr_dict[groups[j]])
    EOTPR = tpr_diff_sum / (n**2)
    EOFPR = fpr_diff_sum / (n**2)
    EO_metric = (EOTPR + EOFPR) / 2.0
    return {'EOTPR': EOTPR, 'EOFPR': EOFPR, 'EO': EO_metric}

def calculate_predictive_parity(sensitive_attr, y_true, y_pred):
    unique_groups = np.unique(sensitive_attr)
    group_precision = {}
    for group in unique_groups:
        mask = (sensitive_attr == group)
        true_positives = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
        total_predicted = np.sum(y_pred[mask] == 1)
        precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        group_precision[group] = precision
    return group_precision

def calculate_multiclass_fairness_metrics(sensitive_attr, y_true, y_pred):
    unique_groups = np.unique(sensitive_attr)
    group_tpr = {}
    group_fpr = {}
    group_precision = {}
    for group in unique_groups:
        mask = (sensitive_attr == group)
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        tpr, fpr = calculate_tpr_and_fpr(y_true_group, y_pred_group)
        group_tpr[group] = tpr
        group_fpr[group] = fpr
        tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
        total_pred = np.sum(y_pred_group == 1)
        prec = tp / total_pred if total_pred > 0 else 0.0
        group_precision[group] = prec

    equalized_odds = calculate_equalized_odds_difference(group_tpr, group_fpr)
    equal_opportunity_diff = np.max(list(group_tpr.values())) - np.min(list(group_tpr.values())) if len(group_tpr) > 0 else 0.0

    return {
        'group_tpr': group_tpr,
        'group_fpr': group_fpr,
        'group_precision': group_precision,
        'equalized_odds': equalized_odds,
        'equal_opportunity_diff': equal_opportunity_diff
    }

def equalised_odds_gap(attr, y_true, y_pred):
    tpr, fpr = {}, {}
    for g in np.unique(attr):
        m = attr==g
        tpr[g], fpr[g] = calculate_tpr_and_fpr(y_true[m], y_pred[m])
    return calculate_equalized_odds_difference(tpr, fpr)['EO']

def compute_eddi(sensitive_attr, true_labels, pred_scores, threshold=0.5):
    y_pred_bin = (pred_scores > threshold).astype(int)
    unique_groups = np.unique(sensitive_attr)
    subgroup_eddi = {}
    overall_error = np.mean(y_pred_bin != true_labels)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_attr == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            er_group = np.mean(y_pred_bin[mask] != true_labels[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom
    overall_eddi = np.sqrt(np.nansum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return overall_eddi, subgroup_eddi

def geom_mean_eddi(*vals):
    return np.sqrt(np.sum(np.array(vals)**2))/len(vals)

class Encoder(nn.Module):
    def __init__(self, seq_len, d_model=768, nhead=8, layers=2):
        super().__init__()
        self.token = nn.Linear(1, d_model)
        self.pos   = nn.Parameter(torch.randn(seq_len, d_model))
        block = nn.TransformerEncoderLayer(d_model, nhead, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(block, layers)
    def forward(self, x):
        # x: [batch, seq_len]
        h = self.token(x.unsqueeze(-1)) + self.pos
        return self.enc(h).mean(1)

class BEHRT(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.backbone = Encoder(seq_len)
        self.fuse = nn.Sequential(nn.Linear(768, 768), nn.Dropout(0.1), nn.GELU())
        self.cls = nn.ModuleList([nn.Linear(768, 1) for _ in range(3)])  # Mortality, PE, PH
    def forward(self, x):
        h = self.fuse(self.backbone(x))
        return tuple(cl(h) for cl in self.cls)

class StructuredDataset(Dataset):
    _SENS = ('age_bucket', 'ethnicity', 'race_group', 'insurance_group', 'gender')
    def __init__(self, csv):
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip().str.lower()
        df.fillna(0, inplace=True)
        self.token_cols = sorted([c for c in df.columns if c.startswith('lab_') or c.startswith('chartevent_')])
        if not self.token_cols:
            raise ValueError("No structured lab/chartevent columns found!")
        df[self.token_cols] = df[self.token_cols].apply(lambda s: (s-s.mean())/s.std() if s.std() else 0.)
        label_map = {
            'short_term_mortality': 'mortality',
            'pe': 'pe',
            'ph': 'ph'
        }
        for k, v in label_map.items():
            if k in df.columns:
                df = df.rename(columns={k: v})
        if not {'mortality', 'pe', 'ph'} <= set(df.columns):
            raise ValueError("Missing required label columns!")
        self.labels = df[['mortality', 'pe', 'ph']].values.astype(np.float32)
        self.feats = df[self.token_cols].values.astype(np.float32)
        self.sens = {c: df[c].astype(str).values for c in self._SENS if c in df.columns}
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.from_numpy(self.feats[idx]), torch.from_numpy(self.labels[idx])

def class_weights(loader):
    Y = torch.cat([y for _, y in loader])
    return [(Y[:, i] == 0).sum().item()/(Y[:, i] == 1).sum().item() if (Y[:, i] == 1).sum() else 1.0 for i in range(3)]


def train(model, tr_loader, val_loader, device, epochs=50, patience=5, lr=1e-5, wd=1e-2):
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = ReduceLROnPlateau(opt, 'min', factor=0.1, patience=2, verbose=True)
    w = class_weights(tr_loader)
    crit = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(wi, device=device)) for wi in w]
    best = math.inf; wait = 0
    for ep in range(1, epochs+1):
        model.train(); tl = []
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = sum(crit[i](out[i].squeeze(), y[:, i]) for i in range(3))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
        tr_loss = float(np.mean(tl))
        model.eval(); vl = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vl.append(sum(crit[i](out[i].squeeze(), y[:, i]).item() for i in range(3)))
        val_loss = float(np.mean(vl))
        sched.step(val_loss)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best:
            best, wait = val_loss, 0
            torch.save(model.state_dict(), 'best_behrt.pt')
            print("   ↳ best model saved")
        else:
            wait += 1
            if wait >= patience:
                print("   ↳ early stopping")
                break

def evaluate(model, loader, device):
    model.eval(); logits = []; labels = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            logits.append(torch.stack([o.squeeze() for o in out], 1).cpu())
            labels.append(y)
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    metrics = {}
    for i, n in enumerate(['mortality', 'pe', 'ph']):
        auroc = roc_auc_score(labels[:, i], logits[:, i])
        pr, rc, _ = precision_recall_curve(labels[:, i], logits[:, i]); auprc = auc(rc, pr)
        pred = (expit(logits[:, i]) > 0.5).astype(int)
        metrics[n] = dict(auroc=auroc, auprc=auprc,
                          precision=precision_score(labels[:, i], pred, zero_division=0),
                          recall=recall_score(labels[:, i], pred),
                          f1=f1_score(labels[:, i], pred))
    return metrics, logits, labels


def main():
    torch.manual_seed(42); np.random.seed(42)
    csv = Path("final_structured_dataset.csv")
    assert csv.exists(), "final_structured_dataset.csv not found"
    ds = StructuredDataset(csv)
    Y = ds.labels
    msss = MultilabelStratifiedShuffleSplit(1, test_size=0.20, random_state=42)
    trv_idx, test_idx = next(msss.split(np.zeros(len(Y)), Y))
    msss2 = MultilabelStratifiedShuffleSplit(1, test_size=0.05/0.80, random_state=42)
    tr_idx_rel, val_idx_rel = next(msss2.split(np.zeros(len(trv_idx)), Y[trv_idx]))
    tr_idx, val_idx = np.array(trv_idx)[tr_idx_rel], np.array(trv_idx)[val_idx_rel]
    print(f"Split → train {len(tr_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    mk = lambda idx, sh=False: DataLoader(Subset(ds, idx), batch_size=32, shuffle=sh)
    tr_loader, val_loader, test_loader = mk(tr_idx, True), mk(val_idx), mk(test_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print("Device:", device)
    model = BEHRT(len(ds.token_cols))
    train(model, tr_loader, val_loader, device)
    model.load_state_dict(torch.load('best_behrt.pt', map_location=device))
    metrics, logits, labels = evaluate(model, test_loader, device)
    print("\nTEST METRICS")
    for n, m in metrics.items():
        print(f"{n:<10} AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | "
              f"F1 {m['f1']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f}")

    probs = expit(logits); preds_bin = (probs > 0.5).astype(int)
    sens = {c: ds.sens[c][test_idx] for c in ds.sens}

    print("\nFAIRNESS – Equalised-Odds Δ")
    for i, out in enumerate(['mortality', 'pe', 'ph']):
        print(f"\nOutcome {out.upper()}")
        for col, val in sens.items():
            print(f"  {col:<12} ΔEO = {equalised_odds_gap(val, labels[:, i], preds_bin[:, i]):.4f}")

    print("\nFAIRNESS – EDDI (error-rate disparity)")
    for i, out in enumerate(['mortality', 'pe', 'ph']):
        print(f"\nOutcome {out.upper()}")
        attr_overall = []
        for col, val in sens.items():
            overall, sub = compute_eddi(val, labels[:, i], probs[:, i])
            attr_overall.append(overall)
            print(f"   {col:<12} overall={overall:.4f}")
            for g, v in sub.items(): print(f"       {g}: {v:+.4f}")
        if attr_overall:
            print(f"   COMBINED EDDI = {geom_mean_eddi(*attr_overall):.4f}")

    print("\nFAIRNESS – Detailed group metrics & EO/EOp/Parity")
    for i, out in enumerate(['mortality', 'pe', 'ph']):
        print(f"\nOutcome {out.upper()}")
        for col, val in sens.items():
            print(f"  Attribute: {col}")
            md = calculate_multiclass_fairness_metrics(val, labels[:, i], preds_bin[:, i])
            print(f"    EO metric : {md['equalized_odds']['EO']:.4f}")
            print(f"    EQ-Opportunity diff : {md['equal_opportunity_diff']:.4f}")
            print("    Per-group TPR / FPR / Precision")
            for g in md['group_tpr']:
                print(f"      {g:<12} TPR {md['group_tpr'][g]:.3f} | "
                      f"FPR {md['group_fpr'][g]:.3f} | "
                      f"P {md['group_precision'][g]:.3f}")

if __name__ == "__main__":
    main()
