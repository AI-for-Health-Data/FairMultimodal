import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from scipy.special import expit

df = pd.read_csv('final_unstructured.csv', low_memory=False)
print(f"Total records: {len(df)}")

note_cols = [c for c in df.columns if c.startswith('note_chunk_')]
label_cols = ['mortality', 'PE', 'PH']

df_unique = df.drop_duplicates(subset=['subject_id']).reset_index(drop=True)

print("Class distribution:")
print(df_unique[label_cols].sum())
print(df_unique[label_cols].shape[0] - df_unique[label_cols].sum())

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

class ClinicalBERTWrapper(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bert = ClinicalBERTWrapper(base_model).to(device)

def aggregate_embeddings(df_rows, note_cols, tokenizer, model, device, max_len=512):
    embs = []
    for _, row in tqdm(df_rows.iterrows(), total=len(df_rows), desc='Embedding patients'):
        texts = [row[col] for col in note_cols if isinstance(row[col], str) and row[col].strip()]
        if not texts:
            embs.append(np.zeros(model.bert.config.hidden_size))
            continue
        reps = []
        for text in texts:
            enc = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            with torch.no_grad():
                rep = model(input_ids, attn).cpu().numpy()
            reps.append(rep)
        reps = np.vstack(reps)
        embs.append(reps.mean(axis=0))
    return np.vstack(embs)

embeddings = aggregate_embeddings(df_unique, note_cols, tokenizer, model_bert, device)
labels = df_unique[label_cols].values

class MultiLabelDataset(Dataset):
    def __init__(self, embs, labels):
        self.x = torch.tensor(embs, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

from sklearn.model_selection import train_test_split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

data_train = DataLoader(MultiLabelDataset(X_tr, y_tr), batch_size=16, shuffle=True)
data_val = DataLoader(MultiLabelDataset(X_val, y_val), batch_size=16)
data_test = DataLoader(MultiLabelDataset(X_te, y_te), batch_size=16)

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, len(label_cols))
        )
    def forward(self, x): return self.net(x)

clf = Classifier(embeddings.shape[1]).to(device)
optimizer = AdamW(clf.parameters(), lr=2e-5)

# Compute pos_weight for each label: neg_count/pos_count
counts = df_unique[label_cols].sum().values
total = len(df_unique)
pos_weight = torch.tensor((total - counts) / counts, dtype=torch.float32).to(device)
print(f"Positive weights: {pos_weight}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val = float('inf'); patience = 0; max_patience = 3
for epoch in range(20):
    clf.train()
    train_loss = 0
    for xb, yb in data_train:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = clf(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        train_loss += loss.item()
    train_loss /= len(data_train)

    clf.eval(); val_loss = 0
    with torch.no_grad():
        for xb, yb in data_val:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(clf(xb), yb).item()
    val_loss /= len(data_val)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val:
        best_val, patience = val_loss, 0
        torch.save(clf.state_dict(), 'best_model.pt')
    else:
        patience += 1
        if patience >= max_patience:
            print("Early stopping.")
            break

def evaluate(loader, threshold=0.5):
    clf.load_state_dict(torch.load('best_model.pt')); clf.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = clf(xb.to(device)).cpu().numpy()
            all_true.append(yb.numpy()); all_prob.append(expit(logits))
    y_true = np.vstack(all_true); y_prob = np.vstack(all_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    for i, name in enumerate(label_cols):
        metrics[name] = {
            'AUROC': roc_auc_score(y_true[:,i], y_prob[:,i]),
            'AUPRC': average_precision_score(y_true[:,i], y_prob[:,i]),
            'F1': f1_score(y_true[:,i], y_pred[:,i], zero_division=0),
            'Recall': recall_score(y_true[:,i], y_pred[:,i], zero_division=0),
            'Precision': precision_score(y_true[:,i], y_pred[:,i], zero_division=0)
        }
    return metrics

print("Test Metrics:", evaluate(data_test))
