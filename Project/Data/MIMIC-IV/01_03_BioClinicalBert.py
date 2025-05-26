import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.special import expit  # for logistic sigmoid
from skmultilearn.model_selection import iterative_train_test_split

class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.bert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

def embed_notes(notes, tokenizer, model, device, max_length=512):
    embeddings = []
    for note in notes:
        if not note or not isinstance(note, str) or note.strip() == "":
            continue
        encoded = tokenizer.encode_plus(
            text=note, add_special_tokens=True, max_length=max_length,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attn_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            emb = model(input_ids, attn_mask)
        embeddings.append(emb.cpu().numpy())
    if not embeddings:
        return np.zeros(model.bert.config.hidden_size)
    return np.mean(np.vstack(embeddings), axis=0)

def aggregate_note_embeddings(df, note_columns, tokenizer, model, device):
    patient_embeddings, patient_ids = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating embeddings"):
        notes = []
        for col in note_columns:
            val = row.get(col, None)
            if pd.notnull(val) and isinstance(val, str) and val.strip() != "":
                notes.append(val)
        emb = embed_notes(notes, tokenizer, model, device)
        patient_embeddings.append(emb)
        patient_ids.append(row['subject_id'])
    return np.stack(patient_embeddings), patient_ids

class UnstructuredDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return self.embeddings.size(0)
    def __getitem__(self, idx): return self.embeddings[idx], self.labels[idx]

class UnstructuredClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x): return self.classifier(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def compute_class_weights(y):
    class_weights = []
    for i in range(y.shape[1]):
        pos = np.sum(y[:, i])
        neg = y.shape[0] - pos
        if pos == 0:  # avoid div0
            w = 1.0
        else:
            w = neg / pos
        class_weights.append(w)
    return class_weights

def train_model(model, dataloader, optimizer, device, criteria):
    model.train()
    running_loss = 0.0
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = 0
        for i, criterion in enumerate(criteria):
            loss += criterion(logits[:, i].unsqueeze(1), Y[:, i].unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, device, criteria):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = 0
            for i, criterion in enumerate(criteria):
                loss += criterion(logits[:, i].unsqueeze(1), Y[:, i].unsqueeze(1))
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(Y.cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return logits, labels, total_loss / len(dataloader)

def compute_eddi(y_true, y_pred, sensitive_labels):
    unique_groups = np.unique(sensitive_labels)
    overall_error = np.mean(y_pred != y_true)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    subgroup_eddi = {}
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            er_group = np.mean(y_pred[mask] != y_true[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom
    eddi_attr = np.sqrt(np.nansum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

def calculate_eo_metric(y_true, y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    tpr_list, fpr_list = {}, {}
    for group in unique_groups:
        mask = sensitive_attr == group
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tpr = fpr = 0
        tpr_list[group] = tpr
        fpr_list[group] = fpr
    tpr_diffs, fpr_diffs = [], []
    groups = list(unique_groups)
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            tpr_diffs.append(abs(tpr_list[groups[i]] - tpr_list[groups[j]]))
            fpr_diffs.append(abs(fpr_list[groups[i]] - fpr_list[groups[j]]))
    EOTPR = np.mean(tpr_diffs) if tpr_diffs else 0.0
    EOFPR = np.mean(fpr_diffs) if fpr_diffs else 0.0
    eo_metric = np.mean([EOTPR, EOFPR])
    return eo_metric, EOTPR, EOFPR

def train_pipeline():
    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader
    from skmultilearn.model_selection import iterative_train_test_split
    from scipy.special import expit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv("final_unstructured_embeddings.csv")
    emb_columns = [col for col in df.columns if col.startswith("emb_")]
    embeddings = df[emb_columns].values
    y = df[['short_term_mortality', 'pe', 'ph']].values.astype(float)

    # Multi-label stratified split
    X = np.arange(len(df)).reshape(-1, 1)  # Use index since we already have embeddings
    X_trainval, y_trainval, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_trainval, y_trainval, test_size=0.1)

    train_idx = X_train.flatten()
    val_idx = X_val.flatten()
    test_idx = X_test.flatten()
    print(f"Train size: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    X_train_e, y_train_e = embeddings[train_idx], y[train_idx]
    X_val_e, y_val_e = embeddings[val_idx], y[val_idx]
    X_test_e, y_test_e = embeddings[test_idx], y[test_idx]

    # Dataset and loaders
    ds_train = UnstructuredDataset(X_train_e, y_train_e)
    ds_val = UnstructuredDataset(X_val_e, y_val_e)
    ds_test = UnstructuredDataset(X_test_e, y_test_e)

    loader_train = DataLoader(ds_train, batch_size=16, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=16, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)

    # Compute class weights (per label) for Focal Loss
    class_weights = compute_class_weights(y_train_e)
    print("Class weights (inverse positive freq):", class_weights)

    # Focal loss for each task
    focal_criteria = [
        FocalLoss(gamma=2, pos_weight=torch.tensor(class_weights[i], dtype=torch.float32, device=device))
        for i in range(3)
    ]

    classifier = UnstructuredClassifier(input_size=embeddings.shape[1], hidden_size=256, output_size=3).to(device)
    optimizer = AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 30
    for epoch in range(max_epochs):
        train_loss = train_model(classifier, loader_train, optimizer, device, focal_criteria)
        _, _, val_loss = evaluate_model(classifier, loader_val, device, focal_criteria)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), "best_model.pt")
            epochs_no_improve = 0
            print("Saved best model.")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 5:
            break

    classifier.load_state_dict(torch.load("best_model.pt"))
    logits, labels, _ = evaluate_model(classifier, loader_test, device, focal_criteria)
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    task_names = ['mortality', 'pe', 'ph']
    for i, name in enumerate(task_names):
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
        auc = roc_auc_score(labels[:,i], probs[:,i]) if len(np.unique(labels[:,i])) > 1 else np.nan
        auprc = average_precision_score(labels[:,i], probs[:,i]) if len(np.unique(labels[:,i])) > 1 else np.nan
        f1 = f1_score(labels[:,i], preds[:,i], zero_division=0)
        rec = recall_score(labels[:,i], preds[:,i], zero_division=0)
        prec = precision_score(labels[:,i], preds[:,i], zero_division=0)
        print(f"\n{name.upper()} AUROC={auc:.3f}, AUPRC={auprc:.3f}, F1={f1:.3f}, Recall={rec:.3f}, Prec={prec:.3f}")

    sensitive_dict = {
        "age": df.iloc[test_idx]['age_bucket'].values if 'age_bucket' in df else df.iloc[test_idx]['anchor_age'].values,
        "ethnicity": df.iloc[test_idx]['ethnicity'].fillna("Other").str.lower().values,
        "race": df.iloc[test_idx]['race_group'].fillna("Other").str.lower().values,
        "insurance": df.iloc[test_idx]['insurance_group'].fillna("Other").str.lower().values
    }
    for i, name in enumerate(task_names):
        print(f"\n--- {name.upper()} FAIRNESS METRICS ---")
        for sattr in sensitive_dict:
            eddi, _ = compute_eddi(labels[:,i], preds[:,i], sensitive_dict[sattr])
            eo, _, _ = calculate_eo_metric(labels[:,i], preds[:,i], sensitive_dict[sattr])
            print(f"{sattr.capitalize():<10} EDDI={eddi:.4f}, EO={eo:.4f}")

if __name__ == "__main__":
    train_pipeline()
