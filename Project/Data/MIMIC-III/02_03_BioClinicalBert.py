#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

class UnstructuredDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BinaryClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_binary(model, loader, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1/(1 + np.exp(-logits))  # sigmoid
    preds = (probs >= threshold).astype(int)

    return {
        'auroc': roc_auc_score(labels, probs),
        'auprc': average_precision_score(labels, probs),
        'f1'   : f1_score(labels, preds),
        'recall': recall_score(labels, preds),
        'precision': precision_score(labels, preds)
    }

def main(
    csv_path: str = 'cohort_unstructured_common_subjects.csv',
    model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    df = pd.read_csv(csv_path, low_memory=False)
    note_cols = [c for c in df.columns if c not in ['subject_id','hadm_id','readmit_30d']]
    labels = df['readmit_30d'].values

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_bert  = AutoModel.from_pretrained(model_name).to(device)
    bert_ft     = BioClinicalBERT_FT(base_bert).to(device)

    print("Embedding notes with Bio_ClinicalBERT...")
    H = base_bert.config.hidden_size
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Embedding'):
        chunks = [row[c] for c in note_cols if pd.notna(row[c])]
        if not chunks:
            embeddings.append(np.zeros(H))
            continue

        reps = []
        for text in chunks:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = enc.input_ids.to(device)
            mask      = enc.attention_mask.to(device)
            with torch.no_grad():
                rep = bert_ft(input_ids, mask).cpu().numpy()
            reps.append(rep)
        embeddings.append(np.mean(np.vstack(reps), axis=0))

    embeddings = np.vstack(embeddings)

    pos = labels.sum()
    neg = len(labels) - pos
    pos_weight = torch.tensor(neg/pos, dtype=torch.float32, device=device)
    print("Pos weight:", pos_weight.item())

    idx = np.random.permutation(len(labels))
    n = len(idx)
    tr, va, te = idx[:int(0.8*n)], idx[int(0.8*n):int(0.9*n)], idx[int(0.9*n):]

    train_ds = UnstructuredDataset(embeddings[tr], labels[tr])
    val_ds   = UnstructuredDataset(embeddings[va], labels[va])
    test_ds  = UnstructuredDataset(embeddings[te], labels[te])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    model     = BinaryClassifier(in_dim=H).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_auc = 0.0
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_binary(model, val_loader, device)
        print(f"Epoch {epoch:2d} | Train Loss: {loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f}")
        if val_metrics['auroc'] > best_auc:
            best_auc = val_metrics['auroc']
            torch.save(model.state_dict(), 'best_readmit_model.pt')

    model.load_state_dict(torch.load('best_readmit_model.pt'))
    print("\nBest model loaded — evaluating on test set:")
    test_metrics = eval_binary(model, test_loader, device)
    for k,v in test_metrics.items():
        print(f"{k.upper():8s}: {v:.4f}")

if __name__ == "__main__":
    main()
