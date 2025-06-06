import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from scipy.special import expit  # for logistic sigmoid

# Focal Loss Definition
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Compute Class Weights (Inverse of Sample Count)
def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

def get_pos_weight(labels_series, device, clip_max=10.0):
    positive = labels_series.sum()
    negative = len(labels_series) - positive
    if positive == 0:
        weight = torch.tensor(1.0, dtype=torch.float, device=device)
    else:
        w = negative / positive
        w = min(w, clip_max)
        weight = torch.tensor(w, dtype=torch.float, device=device)
    print("Positive weight:", weight.item())
    return weight

# BioClinicalBERT Fine-Tuning Wrapper
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.bert = base_model  
        self.device = device

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

# Apply BioClinicalBERT on Patient Notes
def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean", max_length=128):
    patient_ids = df["subject_id"].unique()
    aggregated_embeddings = []
    for pid in tqdm(patient_ids, desc="Aggregating text embeddings"):
        patient_data = df[df["subject_id"] == pid]
        notes = []
        for col in note_columns:
            vals = patient_data[col].dropna().tolist()
            notes.extend([v for v in vals if isinstance(v, str) and v.strip() != ""])
        if len(notes) == 0:
            # Use a vector of zeros if no valid note exists
            aggregated_embeddings.append(np.zeros(model.bert.config.hidden_size))
        else:
            embeddings = []
            for note in notes:
                encoded = tokenizer.encode_plus(
                    text=note,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attn_mask = encoded['attention_mask'].to(device)
                with torch.no_grad():
                    emb = model(input_ids, attn_mask)
                embeddings.append(emb.cpu().numpy())
            embeddings = np.vstack(embeddings)
            agg_emb = np.mean(embeddings, axis=0) if aggregation == "mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings, patient_ids

# Unstructured Dataset Definition
class UnstructuredDataset(Dataset):
    def __init__(self, embeddings, mortality_labels, readmission_labels):
        """
        embeddings: NumPy array of shape (num_patients, hidden_size)
        mortality_labels, readmission_labels: arrays/lists of binary labels
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.mortality_labels = torch.tensor(mortality_labels, dtype=torch.float32).unsqueeze(1)
        self.readmission_labels = torch.tensor(readmission_labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.mortality_labels[idx], self.readmission_labels[idx]

# Unstructured Classifier Definition
class UnstructuredClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super(UnstructuredClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)  
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits

# Training Function
def train_model(model, dataloader, optimizer, device, criterion_mort, criterion_readm):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        embeddings, labels_mort, labels_readm = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(embeddings)  # Shape: (batch_size, 2)
        loss_mort = criterion_mort(logits[:, 0].unsqueeze(1), labels_mort)
        loss_readm = criterion_readm(logits[:, 1].unsqueeze(1), labels_readm)
        loss = loss_mort + loss_readm
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Evaluation Function (for Classification Metrics)
def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels_mort, labels_readm = [x.to(device) for x in batch]
            logits = model(embeddings)
            all_logits.append(logits.cpu())
            # Concatenate labels for both tasks: column 0 for mortality, column 1 for readmission
            all_labels.append(torch.cat((labels_mort.cpu(), labels_readm.cpu()), dim=1))
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    
    print("Probability stats:")
    print("Min:", np.min(probs, axis=0))
    print("Mean:", np.mean(probs, axis=0))
    print("Max:", np.max(probs, axis=0))
    
    preds = (probs >= threshold).astype(int)
    metrics = {}
    tasks = ["mortality", "readmission"]
    for i, task in enumerate(tasks):
        try:
            aucroc = roc_auc_score(all_labels[:, i], probs[:, i])
        except Exception:
            aucroc = float('nan')
        try:
            auprc = average_precision_score(all_labels[:, i], probs[:, i])
        except Exception:
            auprc = float('nan')
        f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        recall = recall_score(all_labels[:, i], preds[:, i], zero_division=0)
        precision = precision_score(all_labels[:, i], preds[:, i], zero_division=0)
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1, "recall": recall, "precision": precision}
    return metrics

# Inference: Get Predicted Probabilities
def get_patient_probabilities(model, dataloader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, _, _ = [x.to(device) for x in batch]
            logits = model(embeddings)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    return probs

# Age Bucket 
def assign_age_bucket(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    elif 70 <= age <= 89:
        return '70-89'
    else:
        return 'other'

# Fairness Metric: Compute EDDI for a Sensitive Attribute
def compute_eddi(df, sensitive_attr, true_label_col, pred_label_col):
    """
    Compute the Error Distribution Disparity Index (EDDI) for a given sensitive attribute.
    
    EDDI = (1/|S|) * sum_{s in S} (|ER_s - OER| / max(OER, 1-OER))
    
    where ER_s is the error rate in subgroup s and OER is the overall error rate.
    """
    groups = df.groupby(sensitive_attr)
    overall_error = np.mean(df[true_label_col] != df[pred_label_col])
    eddi_sum = 0.0
    num_groups = 0
    for group_name, group_df in groups:
        group_error = np.mean(group_df[true_label_col] != group_df[pred_label_col])
        eddi_sum += abs(group_error - overall_error) / max(overall_error, 1 - overall_error)
        num_groups += 1
    eddi = eddi_sum / num_groups if num_groups > 0 else 0.0
    return eddi

# Main Training and Evaluation Pipeline
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Read the CSV data. 
    df = pd.read_csv("filtered_unstructured.csv", low_memory=False)
    print("Data shape:", df.shape)
    
    # Identify note columns (columns starting with 'note_')
    note_columns = [col for col in df.columns if col.startswith("note_")]
    print("Note columns found:", note_columns)
    
    # Filter rows with at least one valid note
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = df[df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))
    
    # Prepare tokenizer and BioClinicalBERT model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    
    # Compute aggregated text embeddings for each patient
    print("Computing aggregated text embeddings for each patient...")
    aggregated_embeddings_np, patient_ids = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean", max_length=128
    )
    print("Aggregated text embeddings shape:", aggregated_embeddings_np.shape)
    
    # Use unique patients for labels and demographics 
    df_unique = df_filtered.drop_duplicates(subset="subject_id")
    
    mortality_labels = df_unique["short_term_mortality"].values
    readmission_labels = df_unique["readmission_within_30_days"].values

    # Build dataset and dataloader (for the classifier)
    dataset = UnstructuredDataset(aggregated_embeddings_np, mortality_labels, readmission_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate classifier
    classifier = UnstructuredClassifier(input_size=768, hidden_size=256).to(device)
    
    # Compute class weights for mortality and readmission
    class_weights_mort = compute_class_weights(df_unique, "short_term_mortality")
    class_weights_readm = compute_class_weights(df_unique, "readmission_within_30_days")
    pos_weight_mort = torch.tensor(class_weights_mort.iloc[1], dtype=torch.float, device=device)
    pos_weight_readm = torch.tensor(class_weights_readm.iloc[1], dtype=torch.float, device=device)
    
    criterion_mort = FocalLoss(gamma=2, pos_weight=pos_weight_mort, reduction='mean')
    criterion_readm = FocalLoss(gamma=2, pos_weight=pos_weight_readm, reduction='mean')
    
    optimizer = AdamW(classifier.parameters(), lr=2e-5)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(classifier, dataloader, optimizer, device, criterion_mort, criterion_readm)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}")
        for thresh in [0.4, 0.5]:
            metrics = evaluate_model(classifier, dataloader, device, threshold=thresh)
            print(f"[Epoch {epoch+1}/{num_epochs}] Threshold: {thresh} Metrics: {metrics}")
    print("Training complete.")
    
    # Inference and Fairness Evaluation for Both Tasks
    # Obtain predicted probabilities for all patients in the dataset
    probs = get_patient_probabilities(classifier, dataloader, device)
    
    # Build a results dataframe by merging predictions with demographics.
    demo_cols = ['subject_id', 'age', 'ethnicity_category', 'insurance_category']
    df_demo = df_unique[demo_cols].copy()
    
    # Create predictions dataframe using patient_ids ordering from aggregated embeddings
    df_probs = pd.DataFrame({
        'subject_id': patient_ids,
        'mortality_prob': probs[:, 0],
        'readmission_prob': probs[:, 1]
    })
    
    # Merge demographics with prediction probabilities
    df_results = pd.merge(df_demo, df_probs, on='subject_id', how='inner')
    
    # Compute binary predictions for both outcomes using threshold = 0.5
    threshold = 0.5
    df_results['mortality_pred'] = (df_results['mortality_prob'] >= threshold).astype(int)
    df_results['readmission_pred'] = (df_results['readmission_prob'] >= threshold).astype(int)
    
    # Merge true labels for both outcomes from df_unique
    df_results = pd.merge(df_results,
                          df_unique[['subject_id', 'short_term_mortality', 'readmission_within_30_days']],
                          on='subject_id', how='inner')
    df_results.rename(columns={
        'short_term_mortality': 'mortality_true',
        'readmission_within_30_days': 'readmission_true'
    }, inplace=True)
    
    # Create age buckets for fairness evaluation
    df_results['age_bucket'] = df_results['age'].apply(assign_age_bucket)
    
    # Fairness Evaluation: Mortality Outcome
    eddi_ethnicity_mort = compute_eddi(df_results, sensitive_attr='ethnicity_category',
                                       true_label_col='mortality_true', pred_label_col='mortality_pred')
    eddi_age_mort = compute_eddi(df_results, sensitive_attr='age_bucket',
                                 true_label_col='mortality_true', pred_label_col='mortality_pred')
    eddi_insurance_mort = compute_eddi(df_results, sensitive_attr='insurance_category',
                                       true_label_col='mortality_true', pred_label_col='mortality_pred')
    overall_eddi_mort = np.mean([eddi_ethnicity_mort, eddi_age_mort, eddi_insurance_mort])
    
    print("\nFairness Evaluation (Mortality Task):")
    print("EDDI for Ethnicity:", eddi_ethnicity_mort)
    print("EDDI for Age Bucket:", eddi_age_mort)
    print("EDDI for Insurance:", eddi_insurance_mort)
    print("Overall EDDI:", overall_eddi_mort)
    
    # Fairness Evaluation: Readmission Outcome
    eddi_ethnicity_readm = compute_eddi(df_results, sensitive_attr='ethnicity_category',
                                        true_label_col='readmission_true', pred_label_col='readmission_pred')
    eddi_age_readm = compute_eddi(df_results, sensitive_attr='age_bucket',
                                  true_label_col='readmission_true', pred_label_col='readmission_pred')
    eddi_insurance_readm = compute_eddi(df_results, sensitive_attr='insurance_category',
                                        true_label_col='readmission_true', pred_label_col='readmission_pred')
    overall_eddi_readm = np.mean([eddi_ethnicity_readm, eddi_age_readm, eddi_insurance_readm])
    
    print("\nFairness Evaluation (Readmission Task):")
    print("EDDI for Ethnicity:", eddi_ethnicity_readm)
    print("EDDI for Age Bucket:", eddi_age_readm)
    print("EDDI for Insurance:", eddi_insurance_readm)
    print("Overall EDDI:", overall_eddi_readm)

if __name__ == "__main__":
    train_pipeline()

