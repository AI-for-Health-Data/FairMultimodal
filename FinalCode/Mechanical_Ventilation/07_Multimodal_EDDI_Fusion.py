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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from scipy.special import expit

# Global debug flag
DEBUG = True

# ================================
# 1. Loss Functions and Class Weight Computation
# ================================
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
    if DEBUG:
        print("Positive weight:", weight.item())
    return weight

# ================================
# 2. BioClinicalBERT Fine-Tuning and Note Aggregation
# ================================
class BioClinicalBERT_FT(nn.Module):
    """
    Fine-tuning wrapper for BioClinicalBERT to obtain the CLS token embedding.
    """
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean"):
    """
    For each unique patient (by subject_id), extracts all non-null notes,
    tokenizes them, computes the CLS embedding, and aggregates them.
    Returns an array of shape (num_patients, hidden_size).
    """
    patient_ids = df["subject_id"].unique()
    aggregated_embeddings = []
    for pid in tqdm(patient_ids, desc="Aggregating text embeddings"):
        patient_data = df[df["subject_id"] == pid]
        notes = []
        for col in note_columns:
            vals = patient_data[col].dropna().tolist()
            notes.extend([v for v in vals if isinstance(v, str) and v.strip() != ""])
        if len(notes) == 0:
            aggregated_embeddings.append(np.zeros(model.BioBert.config.hidden_size))
        else:
            embeddings = []
            for note in notes:
                encoded = tokenizer.encode_plus(
                    text=note,
                    add_special_tokens=True,
                    max_length=128,
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
    return aggregated_embeddings

# ================================
# 3. BEHRT Model for Structured Data
# ================================
class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs,
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()
        config = BertConfig(
            vocab_size=num_diseases,
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=num_segments,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(config)
        self.age_embedding = nn.Embedding(num_ages, hidden_size)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.admission_loc_embedding = nn.Embedding(num_admission_locs, hidden_size)
        self.discharge_loc_embedding = nn.Embedding(num_discharge_locs, hidden_size)
        self.gender_embedding = nn.Embedding(num_genders, hidden_size)
        self.ethnicity_embedding = nn.Embedding(num_ethnicities, hidden_size)
        self.insurance_embedding = nn.Embedding(num_insurances, hidden_size)

    def forward(self, input_ids, attention_mask, age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        # Clamp indices to valid ranges.
        age_ids = torch.clamp(age_ids, 0, self.age_embedding.num_embeddings - 1)
        segment_ids = torch.clamp(segment_ids, 0, self.segment_embedding.num_embeddings - 1)
        adm_loc_ids = torch.clamp(adm_loc_ids, 0, self.admission_loc_embedding.num_embeddings - 1)
        disch_loc_ids = torch.clamp(disch_loc_ids, 0, self.discharge_loc_embedding.num_embeddings - 1)
        gender_ids = torch.clamp(gender_ids, 0, self.gender_embedding.num_embeddings - 1)
        ethnicity_ids = torch.clamp(ethnicity_ids, 0, self.ethnicity_embedding.num_embeddings - 1)
        insurance_ids = torch.clamp(insurance_ids, 0, self.insurance_embedding.num_embeddings - 1)
        # Get extra feature embeddings.
        age_embeds = self.age_embedding(age_ids)
        seg_embeds = self.segment_embedding(segment_ids)
        adm_embeds = self.admission_loc_embedding(adm_loc_ids)
        disch_embeds = self.discharge_loc_embedding(disch_loc_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        eth_embeds = self.ethnicity_embedding(ethnicity_ids)
        ins_embeds = self.insurance_embedding(insurance_ids)
        extra = (age_embeds + seg_embeds + adm_embeds + disch_embeds +
                 gender_embeds + eth_embeds + ins_embeds) / 7.0
        structured_embedding = cls_token + extra
        return structured_embedding

# ================================
# 4. Multimodal Transformer for Mechanical Ventilation Prediction
# ================================
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, BEHRT, device, hidden_size=512):
        super(MultimodalTransformer, self).__init__()
        self.BEHRT = BEHRT
        self.device = device

        self.structured_projector = nn.Sequential(
            nn.Linear(BEHRT.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding):
        structured_emb = self.BEHRT(dummy_input_ids, dummy_attn_mask,
                                    age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                                    gender_ids, ethnicity_ids, insurance_ids)
        proj_struct = self.structured_projector(structured_emb)
        proj_text = self.text_projector(aggregated_text_embedding)
        combined = torch.cat((proj_struct, proj_text), dim=1)
        logits = self.classifier(combined)
        return logits

# ================================
# 5. Training and Evaluation Functions
# ================================
def train_step(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (dummy_input_ids, dummy_attn_mask,
         age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
         gender_ids, ethnicity_ids, insurance_ids,
         aggregated_text_embedding,
         labels) = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(dummy_input_ids, dummy_attn_mask,
                       age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                       gender_ids, ethnicity_ids, insurance_ids,
                       aggregated_text_embedding)
        loss = criterion(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels) = [x.to(device) for x in batch]
            logits = model(dummy_input_ids, dummy_attn_mask,
                           age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                           gender_ids, ethnicity_ids, insurance_ids,
                           aggregated_text_embedding)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy().squeeze()
    all_labels = torch.cat(all_labels, dim=0).numpy().squeeze()
    probs = expit(all_logits)
    preds = (probs >= threshold).astype(int)
    aucroc = roc_auc_score(all_labels, probs) if len(np.unique(all_labels)) > 1 else float('nan')
    auprc = average_precision_score(all_labels, probs) if len(np.unique(all_labels)) > 1 else float('nan')
    f1 = f1_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    precision = precision_score(all_labels, preds, zero_division=0)
    metrics = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
               "recall": recall, "precision": precision}
    return metrics, probs, all_labels

def compute_eddi(y_true, y_pred, sensitive_labels):
    unique_groups = np.unique(sensitive_labels)
    subgroup_d = {}
    overall_error = np.mean(y_pred != y_true)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_d[group] = np.nan
        else:
            er_group = np.mean(y_pred[mask] != y_true[mask])
            subgroup_d[group] = (er_group - overall_error) / denom
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_d.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_d

# ================================
# Helper Functions for Sensitive Attribute Mapping
# ================================
def get_age_bucket(age):
    if 15 <= age <= 29:
        return "15-29"
    elif 30 <= age <= 49:
        return "30-49"
    elif 50 <= age <= 69:
        return "50-69"
    elif 70 <= age <= 89:
        return "70-89"
    else:
        return "others"

def map_ethnicity(code):
    # Mapping order: white, black, hispanic, asian, others
    mapping = {0: "white", 1: "black", 2: "hispanic", 3: "asian"}
    return mapping.get(code, "others")

def map_insurance(code):
    # Mapping order: government, medicare, Medicaid, private, self pay, others
    mapping = {0: "government", 1: "medicare", 2: "Medicaid", 3: "private", 4: "self pay"}
    return mapping.get(code, "others")

# ================================
# 6. Main Training Pipeline for Mechanical Ventilation Prediction
# ================================
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Merge Structured and Unstructured Data 
    structured_data = pd.read_csv('final_structured_with_mechanical_ventilation.csv', low_memory=False)
    unstructured_data = pd.read_csv('final_unstructured_with_mechanical_ventilation.csv', low_memory=False)
    drop_cols = ["age", "ETHNICITY", "INSURANCE", "GENDER", "mechanical_ventilation"]
    unstructured_data.drop(columns=drop_cols, errors='ignore', inplace=True)
    
    merged_df = pd.merge(structured_data, unstructured_data, on=["subject_id", "hadm_id"], how="inner")
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check your merge keys and data.")
    
    merged_df["mechanical_ventilation"] = merged_df["mechanical_ventilation"].astype(int)
    
    # Determine note columns.
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))
    
    # Create age bucket for fairness analysis.
    if "age" in df_filtered.columns:
        df_filtered['age_bucket'] = df_filtered['age'].apply(lambda x: get_age_bucket(x))
    
    # Compute Aggregated Text Embeddings (Text Branch)
    print("Computing aggregated text embeddings for each patient...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    print("Aggregated text embeddings shape:", aggregated_text_embeddings_np.shape)
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)
    
    # Process Structured Data: Convert categorical columns to codes.
    for col in ["GENDER", "ETHNICITY", "INSURANCE"]:
        if col not in df_filtered.columns:
            df_filtered[col] = 0
        elif df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].astype("category").cat.codes
    for col in ["FIRST_WARDID", "LAST_WARDID"]:
        if col not in df_filtered.columns:
            df_filtered[col] = 0
        elif df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].astype("category").cat.codes
    if "age" not in df_filtered.columns:
        df_filtered["age"] = 0
    elif df_filtered["age"].dtype == object:
        df_filtered["age"] = df_filtered["age"].astype("category").cat.codes
    if "segment" not in df_filtered.columns:
        df_filtered["segment"] = 0
    elif df_filtered["segment"].dtype == object:
        df_filtered["segment"] = df_filtered["segment"].astype("category").cat.codes

    num_samples = len(df_filtered)
    # Create dummy inputs for BEHRT.
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    
    age_ids = torch.tensor(df_filtered["age"].values, dtype=torch.long)
    segment_ids = torch.tensor(df_filtered["segment"].values, dtype=torch.long)
    adm_loc_ids = torch.tensor(df_filtered["FIRST_WARDID"].values, dtype=torch.long)
    disch_loc_ids = torch.tensor(df_filtered["LAST_WARDID"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_filtered["GENDER"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_filtered["ETHNICITY"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_filtered["INSURANCE"].values, dtype=torch.long)
    
    labels = torch.tensor(df_filtered["mechanical_ventilation"].values, dtype=torch.float32)
    
    # Compute class weights for mechanical ventilation.
    class_weights_ventilation = compute_class_weights(df_filtered, 'mechanical_ventilation')
    ventilation_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    
    # Create TensorDataset and DataLoader.
    dataset = TensorDataset(
        dummy_input_ids, dummy_attn_mask,
        age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
        gender_ids, ethnicity_ids, insurance_ids,
        aggregated_text_embeddings_t,
        labels
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Set hyperparameters based on processed data.
    disease_mapping = {d: i for i, d in enumerate(df_filtered["hadm_id"].unique())}
    NUM_DISEASES = len(disease_mapping)
    NUM_AGES = df_filtered["age"].nunique()
    NUM_SEGMENTS = 2
    NUM_ADMISSION_LOCS = df_filtered["FIRST_WARDID"].nunique()
    NUM_DISCHARGE_LOCS = df_filtered["LAST_WARDID"].nunique()
    NUM_GENDERS = df_filtered["GENDER"].nunique()
    NUM_ETHNICITIES = df_filtered["ETHNICITY"].nunique()
    NUM_INSURANCES = df_filtered["INSURANCE"].nunique()
    
    print("\n--- Hyperparameters based on processed data ---")
    print("NUM_DISEASES:", NUM_DISEASES)
    print("NUM_AGES:", NUM_AGES)
    print("NUM_SEGMENTS:", NUM_SEGMENTS)
    print("NUM_ADMISSION_LOCS:", NUM_ADMISSION_LOCS)
    print("NUM_DISCHARGE_LOCS:", NUM_DISCHARGE_LOCS)
    print("NUM_GENDERS:", NUM_GENDERS)
    print("NUM_ETHNICITIES:", NUM_ETHNICITIES)
    print("NUM_INSURANCES:", NUM_INSURANCES)
    
    # Initialize the BEHRT model.
    behrt_model = BEHRTModel(
        num_diseases=NUM_DISEASES,
        num_ages=NUM_AGES,
        num_segments=NUM_SEGMENTS,
        num_admission_locs=NUM_ADMISSION_LOCS,
        num_discharge_locs=NUM_DISCHARGE_LOCS,
        num_genders=NUM_GENDERS,
        num_ethnicities=NUM_ETHNICITIES,
        num_insurances=NUM_INSURANCES,
        hidden_size=768
    ).to(device)
    
    # Initialize the Multimodal Transformer.
    multimodal_model = MultimodalTransformer(
        text_embed_size=768,
        BEHRT=behrt_model,
        device=device,
        hidden_size=512
    ).to(device)
    
    optimizer = AdamW(multimodal_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = FocalLoss(gamma=2, pos_weight=ventilation_pos_weight, reduction='mean')
    
    # Training Loop.
    num_epochs = 20
    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = train_step(multimodal_model, dataloader, optimizer, device, criterion)
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)
    
    # Evaluation.
    metrics, probs, true_labels = evaluate_model(multimodal_model, dataloader, device, threshold=0.5)
    print("\nFinal Evaluation Metrics for Mechanical Ventilation Prediction:")
    print(f"AUC-ROC: {metrics['aucroc']:.4f}, AUPRC: {metrics['auprc']:.4f}, F1: {metrics['f1']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")
    
    # EDDI Computation for Fairness Analysis.
    def get_model_predictions(model, dataloader, device):
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                (dummy_input_ids, dummy_attn_mask,
                 age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                 gender_ids, ethnicity_ids, insurance_ids,
                 aggregated_text_embedding,
                 labels) = [x.to(device) for x in batch]
                logits = model(dummy_input_ids, dummy_attn_mask,
                               age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                               gender_ids, ethnicity_ids, insurance_ids,
                               aggregated_text_embedding)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                all_predictions.extend(preds)
                all_labels.append(labels.cpu().numpy())
        all_labels = np.concatenate(all_labels, axis=0)
        return np.array(all_predictions), all_labels

    preds, true_labels = get_model_predictions(multimodal_model, dataloader, device)
    
    # Create age bins: 15-29, 30-49, 50-69, 70-89.
    age_bins = [15, 30, 50, 70, 90]
    age_labels_bins = ['15-29', '30-49', '50-69', '70-89']
    df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=age_bins, labels=age_labels_bins, right=False)
    
    # Get sensitive attribute arrays.
    age_groups = df_filtered['age_group'].values
    ethnicity_groups = np.array([map_ethnicity(e) for e in df_filtered['ETHNICITY'].values])
    insurance_groups = np.array([map_insurance(i) for i in df_filtered['INSURANCE'].values])
    
    def print_subgroup_eddi(true, pred, sensitive_name, sensitive_values):
        eddi_attr, subgroup_d = compute_eddi(true, pred, sensitive_values)
        print(f"\nSensitive Attribute: {sensitive_name}")
        print(f"Overall Error Rate: {np.mean(true != pred):.4f}")
        for group, d in subgroup_d.items():
            print(f"  Subgroup {group}: d(s) = {d:.4f}")
        print(f"Attribute-level EDDI for {sensitive_name}: {eddi_attr:.4f}\n")
        return subgroup_d, eddi_attr
    
    print("\n=== EDDI Calculation for Mechanical Ventilation ===")
    print_subgroup_eddi(true_labels, preds, "Age Groups", age_groups)
    print_subgroup_eddi(true_labels, preds, "Ethnicity Groups", ethnicity_groups)
    print_subgroup_eddi(true_labels, preds, "Insurance Groups", insurance_groups)
    
    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
