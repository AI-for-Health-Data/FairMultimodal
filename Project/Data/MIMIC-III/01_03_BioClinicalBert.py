import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:,0,:] 

class UnstructuredDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Classifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

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


def eval_model(model, loader, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {}
    names = ['mortality', 'PE', 'PH']
    for i, name in enumerate(names):
        try:
            metrics[name] = {
                'auroc': roc_auc_score(labels[:, i], probs[:, i]),
                'auprc': average_precision_score(labels[:, i], probs[:, i]),
                'f1': f1_score(labels[:, i], preds[:, i]),
                'recall': recall_score(labels[:, i], preds[:, i]),
                'precision': precision_score(labels[:, i], preds[:, i])
            }
        except:
            metrics[name] = {k: float('nan') for k in ['auroc', 'auprc', 'f1', 'recall', 'precision']}
    return metrics

def main(unstruct_path, epochs=20, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(unstruct_path, low_memory=False)

    note_cols = [c for c in df.columns if c.startswith('note_chunk_')]
    labels = df[['mortality', 'PE', 'PH']].values

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    base_bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    bert_ft = BioClinicalBERT_FT(base_bert).to(device)

    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Embedding notes'):
        chunks = [row[c] for c in note_cols if pd.notna(row[c])]
        if not chunks:
            embeddings.append(np.zeros(base_bert.config.hidden_size))
            continue
        reps = []
        for text in chunks:
            enc = tokenizer(text, max_length=512, truncation=True,
                             padding='max_length', return_tensors='pt')
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask.to(device)
            with torch.no_grad():
                rep = bert_ft(ids, mask).cpu().numpy()
            reps.append(rep)
        embeddings.append(np.mean(np.vstack(reps), axis=0))
    embeddings = np.vstack(embeddings)

    # Compute positive weights for imbalance
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos_weight = torch.tensor((neg / pos), dtype=torch.float32, device=device)

    ds = UnstructuredDataset(embeddings, labels)
    total = len(ds)
    idx = np.arange(total)
    np.random.shuffle(idx)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds   = torch.utils.data.Subset(ds, val_idx)
    test_ds  = torch.utils.data.Subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = Classifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val AUROC(Mortality)={val_metrics['mortality']['auroc']:.4f}")

    test_metrics = eval_model(model, test_loader, device)
    print("\nTest Set Metrics:")
    for outcome, m in test_metrics.items():
        print(f"{outcome}: AUROC={m['auroc']:.3f}, AUPRC={m['auprc']:.3f}, F1={m['f1']:.3f}")

if __name__ == '__main__':
    main('final_unstructured.csv')
