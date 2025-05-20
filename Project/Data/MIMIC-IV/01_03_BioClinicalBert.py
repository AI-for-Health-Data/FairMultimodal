import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            reduction='none',
            pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()

class UnstructuredDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, y_mort, y_pe, y_ph):
        self.X = torch.tensor(np.nan_to_num(embeddings, nan=0.0), dtype=torch.float32)
        labels = np.vstack([y_mort, y_pe, y_ph]).T
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = self.body(x)
        return self.head(h)

def train_one_epoch(model, loader, optimizer, losses, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = sum(
            losses[i](logits[:, i:i+1], yb[:, i:i+1])
            for i in range(3)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(yb.numpy())
    logits = np.vstack(all_logits)
    y_true = np.vstack(all_labels)

    y_prob = 1 / (1 + np.exp(-logits))
    y_prob = np.nan_to_num(y_prob, nan=0.5)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}
    names = ['Mortality','PE','PH']
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        # mask any remaining NaNs (should not happen after nan_to_num)
        mask = ~np.isnan(yp)
        yt_m, yp_m = yt[mask], yp[mask]
        ypr = (yp_m >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt_m, ypr, labels=[0,1]).ravel()
        tpr = tp/(tp+fn) if tp+fn>0 else 0.0
        fpr = fp/(fp+tn) if fp+tn>0 else 0.0

        metrics[name] = {
            'AUROC'    : roc_auc_score(yt_m, yp_m),
            'AUPRC'    : average_precision_score(yt_m, yp_m),
            'F1'       : f1_score(yt_m, ypr, zero_division=0),
            'Recall'   : recall_score(yt_m, ypr, zero_division=0),
            'Precision': precision_score(yt_m, ypr, zero_division=0),
            'TPR'      : tpr,
            'FPR'      : fpr
        }
    return metrics

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    p = argparse.ArgumentParser(description="Unstructured‐only classifier")
    p.add_argument('-u','--unstruct',
                   default='final_unstructured_embeddings.csv',
                   help='CSV with emb_* columns and labels: short_term_mortality, pe, ph')
    p.add_argument('-e','--epochs', type=int, default=20)
    p.add_argument('-b','--batch',  type=int, default=32)
    args = p.parse_args()

    df = pd.read_csv(args.unstruct, low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    for lbl in ['short_term_mortality','pe','ph']:
        if lbl not in df.columns:
            raise ValueError(f"Missing label column: {lbl}")

    X_all = df[emb_cols].values
    Y_all = df[['short_term_mortality','pe','ph']].values

    X_tmp, y_tmp, X_test, y_test = iterative_train_test_split(X_all, Y_all, test_size=0.2)
    val_frac = 0.05/0.8
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_tmp, y_tmp, test_size=val_frac)

    train_ds = UnstructuredDataset(X_train, y_train[:,0], y_train[:,1], y_train[:,2])
    val_ds   = UnstructuredDataset(X_val,   y_val[:,0],   y_val[:,1],   y_val[:,2])
    test_ds  = UnstructuredDataset(X_test,  y_test[:,0],  y_test[:,1],  y_test[:,2])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = Classifier(input_dim=X_train.shape[1]).to(device)

    pos_weights = []
    for i in range(3):
        pos = y_train[:,i].sum()
        neg = len(y_train) - pos
        w = min(neg/(pos+1e-6), 10.0)
        pos_weights.append(torch.tensor([w], device=device))
    losses = [FocalLoss(pos_weight=pw).to(device) for pw in pos_weights]

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_cnt  = 0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, losses, device)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                out = model(Xb.to(device))
                for i in range(3):
                    val_loss += losses[i](out[:,i:i+1], yb[:,i:i+1].to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f"[Epoch {epoch}] Train={tr_loss:.4f} Val={val_loss:.4f}")

        # always save on epoch 1 or when improved
        if epoch == 1 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_cnt = 0
            print(" → saved best_model.pt")
        else:
            patience_cnt += 1
            print(f" → no improve {patience_cnt}/5")
            if patience_cnt >= 5:
                print("Early stopping.")
                break

    if not os.path.exists("best_model.pt"):
        raise FileNotFoundError("best_model.pt not found after training.")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n=== Test Set Metrics ===")
    for name, stats in test_metrics.items():
        summary = ", ".join(f"{k}={v:.3f}" for k,v in stats.items())
        print(f"{name}: {summary}")

if __name__ == '__main__':
    main()
