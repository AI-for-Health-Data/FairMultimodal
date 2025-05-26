import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
)
from scipy.special import expit  # for logistic sigmoid
from skmultilearn.model_selection import iterative_train_test_split

class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token
        return outputs.last_hidden_state[:, 0, :]

class PatientDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class PatientClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.classifier(x)

def apply_bioclinicalbert_on_notes(df, note_columns, tokenizer, model, device, aggregation="mean", max_length=512):
    """Aggregate BioClinicalBERT CLS embeddings for each patient."""
    patient_ids = df["subject_id"].values
    aggregated_embeddings = []
    model.eval()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="BERT embeddings"):
        note_embs = []
        for col in note_columns:
            note = row.get(col, "")
            if isinstance(note, str) and note.strip():
                encoded = tokenizer.encode_plus(
                    note, add_special_tokens=True, max_length=max_length, padding='max_length',
                    truncation=True, return_attention_mask=True, return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attn_mask = encoded['attention_mask'].to(device)
                with torch.no_grad():
                    emb = model(input_ids, attn_mask)  # (1, 768)
                note_embs.append(emb.cpu().numpy()[0])
        if note_embs:
            patient_emb = np.mean(note_embs, axis=0) if aggregation == "mean" else np.max(note_embs, axis=0)
        else:
            patient_emb = np.zeros(model.bert.config.hidden_size)
        aggregated_embeddings.append(patient_emb)
    return np.vstack(aggregated_embeddings), patient_ids

def compute_class_weights(labels):
    """Return pos_weight for binary cross-entropy loss."""
    pos = labels.sum()
    neg = len(labels) - pos
    pos_weight = neg / pos if pos > 0 else 1.0
    return torch.tensor(pos_weight, dtype=torch.float)

def compute_eddi(y_true, y_pred, sensitive):
    unique = np.unique(sensitive)
    overall_err = np.mean(y_true != y_pred)
    denom = max(overall_err, 1-overall_err) if overall_err not in [0, 1] else 1.0
    subgroup_eddi = {}
    for group in unique:
        mask = sensitive == group
        if mask.sum() == 0: continue
        err_g = np.mean(y_true[mask] != y_pred[mask])
        subgroup_eddi[group] = (err_g - overall_err) / denom
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values()))**2)) / len(subgroup_eddi)
    return eddi_attr, subgroup_eddi

def calculate_eo_metric(y_true, y_pred, sensitive_attr):
    unique = np.unique(sensitive_attr)
    tpr_list, fpr_list = [], []
    for group in unique:
        mask = sensitive_attr == group
        if mask.sum() == 0:
            tpr, fpr = 0, 0
        else:
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    tpr_diff = np.mean([abs(a-b) for i,a in enumerate(tpr_list) for b in tpr_list[i+1:]])
    fpr_diff = np.mean([abs(a-b) for i,a in enumerate(fpr_list) for b in fpr_list[i+1:]])
    eo_metric = np.mean([tpr_diff, fpr_diff])
    return eo_metric, tpr_diff, fpr_diff

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("unstructured_common_subjects.csv", low_memory=False)
    print("Loaded data shape:", df.shape)

    note_columns = [col for col in df.columns if col.startswith("note_")]
    print(f"Note columns: {note_columns}")

    df = df[df[note_columns].apply(lambda x: x.str.strip().any(), axis=1)].reset_index(drop=True)
    print("After filtering, rows:", len(df))

    labels = df["readmit_30d"].values.astype(int)

    df['age_bucket'] = df['age_bucket'].fillna('Other')
    df['race_cat'] = df['race_cat'].fillna('Other')
    df['ethnicity_cat'] = df['ethnicity_cat'].fillna('Other')
    df['insurance_cat'] = df['insurance_cat'].fillna('Other')

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_base = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    bert_ft = BioClinicalBERT_FT(bert_base).to(device)
    bert_ft.eval()

    emb_file = "patient_note_embeddings.npy"
    if not os.path.exists(emb_file):
        embeddings, patient_ids = apply_bioclinicalbert_on_notes(df, note_columns, tokenizer, bert_ft, device)
        np.save(emb_file, embeddings)
        print(f"Saved patient embeddings to {emb_file}")
    else:
        embeddings = np.load(emb_file)
        patient_ids = df["subject_id"].values

    X = df[["subject_id"]].values
    y = labels.reshape(-1, 1)
    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)
    subj_train_val = set(X_train_val.flatten().astype(str))
    subj_test = set(X_test.flatten().astype(str))

    df_train_val = df[df['subject_id'].astype(str).isin(subj_train_val)]
    df_test = df[df['subject_id'].astype(str).isin(subj_test)]
    emb_train_val = embeddings[df['subject_id'].astype(str).isin(subj_train_val)]
    emb_test = embeddings[df['subject_id'].astype(str).isin(subj_test)]
    y_train_val = labels[df['subject_id'].astype(str).isin(subj_train_val)]
    y_test = labels[df['subject_id'].astype(str).isin(subj_test)]

    val_frac = 0.05 / 0.8
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_train_val, y_train_val.reshape(-1, 1), test_size=val_frac
    )
    subj_train = set(X_train.flatten().astype(str))
    subj_val = set(X_val.flatten().astype(str))
    df_train = df_train_val[df_train_val['subject_id'].astype(str).isin(subj_train)]
    df_val = df_train_val[df_train_val['subject_id'].astype(str).isin(subj_val)]
    emb_train = emb_train_val[df_train_val['subject_id'].astype(str).isin(subj_train)]
    emb_val = emb_train_val[df_train_val['subject_id'].astype(str).isin(subj_val)]
    y_train = y_train_val[df_train_val['subject_id'].astype(str).isin(subj_train)]
    y_val = y_train_val[df_train_val['subject_id'].astype(str).isin(subj_val)]

    train_dataset = PatientDataset(emb_train, y_train)
    val_dataset = PatientDataset(emb_val, y_val)
    test_dataset = PatientDataset(emb_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    classifier = PatientClassifier(input_size=emb_train.shape[1], hidden_size=256).to(device)
    pos_weight = compute_class_weights(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience = 5
    wait = 0
    for epoch in range(30):
        classifier.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_losses = []
        classifier.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = classifier(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(losses):.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), "best_classifier.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    classifier.load_state_dict(torch.load("best_classifier.pt"))

    classifier.eval()
    y_prob, y_pred, y_true = [], [], []
    for xb, yb in test_loader:
        xb = xb.to(device)
        with torch.no_grad():
            logits = classifier(xb)
        prob = expit(logits.cpu().numpy()).flatten()
        pred = (prob >= 0.5).astype(int)
        y_prob.extend(prob.tolist())
        y_pred.extend(pred.tolist())
        y_true.extend(yb.numpy().flatten().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print("\n--- Test Metrics ---")
    print("AUROC:", roc_auc_score(y_true, y_prob))
    print("AUPRC:", average_precision_score(y_true, y_prob))
    print("F1-score:", f1_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))

    sens_attrs = {
        "age_bucket": df_test['age_bucket'].values,
        "ethnicity_cat": df_test['ethnicity_cat'].values,
        "race_cat": df_test['race_cat'].values,
        "insurance_cat": df_test['insurance_cat'].values,
    }
    for attr, values in sens_attrs.items():
        eddi, sub = compute_eddi(y_true, y_pred, np.array(values))
        eo, tprd, fprd = calculate_eo_metric(y_true, y_pred, np.array(values))
        print(f"\n--- Fairness on {attr} ---")
        print(f"  EDDI: {eddi:.4f}")
        print(f"  EO Metric (mean TPR/FPR diff): {eo:.4f} (TPR diff: {tprd:.4f}, FPR diff: {fprd:.4f})")
        print(f"  Subgroup EDDI: {sub}")

if __name__ == "__main__":
    train_pipeline()
