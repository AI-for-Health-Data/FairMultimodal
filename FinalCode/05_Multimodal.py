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

from transformers import BertModel, BertConfig, AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score


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

# Compute class weights using the Inverse of Number of Samples (INS)
def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

# Get positive weight for binary loss computation
def get_pos_weight(labels_series, device):
    positive = labels_series.sum()
    negative = len(labels_series) - positive
    if positive == 0:
        weight = torch.tensor(1.0, dtype=torch.float, device=device)
    else:
        weight = torch.tensor(negative / positive, dtype=torch.float, device=device)
    return weight

# BioClinicalBERT Fine-Tuning Wrapper
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding

# --- Function to Process and Aggregate Patient Notes ---
def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean"):
    """
    For each unique patient (by subject_id), extracts all non-null notes from the given note columns,
    tokenizes them, computes the CLS embedding using the provided model, and aggregates (by mean or max) them.
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
            # If no valid note is found, use a zero vector.
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
                    emb = model(input_ids, attn_mask)  # (1, hidden_size)
                embeddings.append(emb.cpu().numpy())
            embeddings = np.vstack(embeddings)
            if aggregation == "mean":
                agg_emb = np.mean(embeddings, axis=0)
            else:
                agg_emb = np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings

# --- BEHRT Model for Structured Data ---
class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs, 
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()
        # Here, we construct a vocabulary size for the underlying BERT (this example adds extra tokens)
        vocab_size = num_diseases + num_ages + num_segments + num_admission_locs + num_discharge_locs + 2
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(config)
        # Extra embeddings for structured variables
        self.age_embedding = nn.Embedding(num_ages, hidden_size)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.admission_loc_embedding = nn.Embedding(num_admission_locs, hidden_size)
        self.discharge_loc_embedding = nn.Embedding(num_discharge_locs, hidden_size)
        self.gender_embedding = nn.Embedding(num_genders, hidden_size)
        self.ethnicity_embedding = nn.Embedding(num_ethnicities, hidden_size)
        self.insurance_embedding = nn.Embedding(num_insurances, hidden_size)

    def forward(self, input_ids, attention_mask, age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids):
        # Clamp IDs to valid ranges
        age_ids = torch.clamp(age_ids, 0, self.age_embedding.num_embeddings - 1)
        segment_ids = torch.clamp(segment_ids, 0, self.segment_embedding.num_embeddings - 1)
        adm_loc_ids = torch.clamp(adm_loc_ids, 0, self.admission_loc_embedding.num_embeddings - 1)
        disch_loc_ids = torch.clamp(disch_loc_ids, 0, self.discharge_loc_embedding.num_embeddings - 1)
        gender_ids = torch.clamp(gender_ids, 0, self.gender_embedding.num_embeddings - 1)
        ethnicity_ids = torch.clamp(ethnicity_ids, 0, self.ethnicity_embedding.num_embeddings - 1)
        insurance_ids = torch.clamp(insurance_ids, 0, self.insurance_embedding.num_embeddings - 1)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_length, hidden_size)
        # Get extra embeddings from structured variables
        age_embeds = self.age_embedding(age_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        adm_embeds = self.admission_loc_embedding(adm_loc_ids)
        disch_embeds = self.discharge_loc_embedding(disch_loc_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        eth_embeds = self.ethnicity_embedding(ethnicity_ids)
        ins_embeds = self.insurance_embedding(insurance_ids)
        # Compute an “extra” embedding by averaging these additional features
        extra = (age_embeds + segment_embeds + adm_embeds + disch_embeds + gender_embeds + eth_embeds + ins_embeds) / 7.0
        # Combine the CLS token from BERT with the extra information.
        cls_token = sequence_output[:, 0, :]  # CLS token from BERT
        # If extra is a 3D tensor (e.g. if using sequence input), take the first token's extra embedding.
        if extra.dim() == 3:
            cls_embedding = cls_token + extra[:, 0, :]
        else:
            cls_embedding = cls_token + extra
        return cls_embedding  # (batch, hidden_size)

# --- Multimodal Transformer Model ---
# This model takes the CLS embedding from the BEHRT branch (structured data) and the aggregated
# CLS embedding from the BioClinicalBERT branch (unstructured patient notes), projects them to a common
# space, concatenates them, and then predicts the outcomes.
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, BEHRT, device, hidden_size=512):
        super(MultimodalTransformer, self).__init__()
        self.BEHRT = BEHRT
        self.device = device

        # Project the BEHRT (structured) embedding to 256 dimensions.
        self.ts_projector = nn.Sequential(
            nn.Linear(BEHRT.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        # Project the BioClinicalBERT (text) aggregated embedding to 256 dimensions.
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)  # Two outputs: one for mortality and one for readmission
        )

    def forward(self, dummy_input_ids, dummy_attn_mask, 
                age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding):
        # Get the CLS embedding from BEHRT (structured branch)
        structured_emb = self.BEHRT(dummy_input_ids, dummy_attn_mask,
                                    age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                                    gender_ids, ethnicity_ids, insurance_ids)
        ts_proj = self.ts_projector(structured_emb)
        # aggregated_text_embedding is assumed to be the (precomputed) CLS embedding from BioClinicalBERT
        text_proj = self.text_projector(aggregated_text_embedding)
        # Concatenate both projected embeddings
        combined = torch.cat((ts_proj, text_proj), dim=1)
        logits = self.classifier(combined)
        # Split the logits for the two tasks
        mortality_logits = logits[:, 0].unsqueeze(1)
        readmission_logits = logits[:, 1].unsqueeze(1)
        return mortality_logits, readmission_logits


# Training and Evaluation Functions 

def train_step(model, dataloader, optimizer, device, criterion_mortality, criterion_readmission):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (dummy_input_ids, dummy_attn_mask,
         age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
         gender_ids, ethnicity_ids, insurance_ids,
         aggregated_text_embedding,
         labels_mortality, labels_readmission) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        mortality_logits, readmission_logits = model(
            dummy_input_ids, dummy_attn_mask,
            age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
            gender_ids, ethnicity_ids, insurance_ids,
            aggregated_text_embedding
        )
        loss_mortality = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
        loss_readmission = criterion_readmission(readmission_logits, labels_readmission.unsqueeze(1))
        loss = loss_mortality + loss_readmission
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_mort_logits = []
    all_readm_logits = []
    all_labels_mort = []
    all_labels_readm = []
    with torch.no_grad():
        for batch in dataloader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_readmission) = [x.to(device) for x in batch]
            mort_logits, readm_logits = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_readm_logits.append(readm_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_readm.append(labels_readmission.cpu())
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_readm_logits = torch.cat(all_readm_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_readm = torch.cat(all_labels_readm, dim=0)

    mort_probs = torch.sigmoid(all_mort_logits).numpy()
    readm_probs = torch.sigmoid(all_readm_logits).numpy()
    labels_mort_np = all_labels_mort.numpy()
    labels_readm_np = all_labels_readm.numpy()

    metrics = {}
    for task, probs, labels in zip(["mortality", "readmission"],
                                   [mort_probs, readm_probs],
                                   [labels_mort_np, labels_readm_np]):
        try:
            aucroc = roc_auc_score(labels, probs)
        except Exception:
            aucroc = float('nan')
        try:
            auprc = average_precision_score(labels, probs)
        except Exception:
            auprc = float('nan')
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1, "recall": recall, "precision": precision}
    return metrics, all_mort_logits, all_readm_logits


## Training Step Function
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Merge Structured and Unstructured Data -----
    structured_data = pd.read_csv('filtered_structured_first_icu_stays.csv')
    unstructured_data = pd.read_csv("filtered_unstructured.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)
    
    # Drop duplicate columns from unstructured data if they exist.
    unstructured_data.drop(
        columns=["short_term_mortality", "readmission_within_30_days", "age",
                 "segment", "admission_loc", "discharge_loc", "gender", "ethnicity", "insurance"],
        errors='ignore',
        inplace=True
    )

    merged_df = pd.merge(
        structured_data,
        unstructured_data,
        on=["subject_id", "hadm_id"],
        how="inner",
        suffixes=("_struct", "_unstruct")
    )
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check your data and merge keys.")
    
    # Ensure target columns are integers.
    merged_df["short_term_mortality"] = merged_df["short_term_mortality"].astype(int)
    merged_df["readmission_within_30_days"] = merged_df["readmission_within_30_days"].astype(int)

    # Identify note columns (those starting with "note_")
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False

    # Filter to rows that have at least one valid note.
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    # ----- Compute Aggregated Text Embeddings -----
    print("Computing aggregated text embeddings for each patient...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)

    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    print("Aggregated text embeddings shape:", aggregated_text_embeddings_np.shape)
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    # ----- Process Structured Data Columns -----
    # For categorical columns, ensure they exist and are numeric (if needed, convert using .cat.codes).
    categorical_columns = [
        "GENDER_struct",
        "ETHNICITY_struct",
        "INSURANCE_struct",
        "FIRST_WARDID_struct",
        "LAST_WARDID_struct"
    ]
    for col in categorical_columns:
        if col not in df_filtered.columns:
            print(f"Column {col} not found; creating default values.")
            df_filtered[col] = 0
        elif df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].astype("category").cat.codes

    # Process age and segment columns.
    if "age" in df_filtered.columns and df_filtered["age"].dtype == object:
        df_filtered["age"] = df_filtered["age"].astype("category").cat.codes
    if "segment" not in df_filtered.columns:
        df_filtered["segment"] = 0
    elif df_filtered["segment"].dtype == object:
        df_filtered["segment"] = df_filtered["segment"].astype("category").cat.codes

    num_samples = len(df_filtered)
    # Create dummy inputs for BEHRT’s underlying BERT (which expects token ids and attention masks).
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)

    # Create tensors for structured variables.
    age_ids = torch.tensor(df_filtered["age"].values, dtype=torch.long)
    segment_ids = torch.tensor(df_filtered["segment"].values, dtype=torch.long)
    admission_loc_ids = torch.tensor(df_filtered["FIRST_WARDID_struct"].values, dtype=torch.long)
    discharge_loc_ids = torch.tensor(df_filtered["LAST_WARDID_struct"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_filtered["GENDER_struct"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_filtered["ETHNICITY_struct"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_filtered["INSURANCE_struct"].values, dtype=torch.long)

    labels_mortality = torch.tensor(df_filtered["short_term_mortality"].values, dtype=torch.float32)
    labels_readmission = torch.tensor(df_filtered["readmission_within_30_days"].values, dtype=torch.float32)

    # ----- Compute Class Weights (INS) for Each Task -----
    class_weights_mortality = compute_class_weights(df_filtered, 'short_term_mortality')
    class_weights_readmission = compute_class_weights(df_filtered, 'readmission_within_30_days')
    class_weights_tensor_mortality = torch.tensor(class_weights_mortality.values, dtype=torch.float).to('cpu')
    class_weights_tensor_readmission = torch.tensor(class_weights_readmission.values, dtype=torch.float).to('cpu')

    # ----- Create Dataset and DataLoader -----
    dataset = TensorDataset(
        dummy_input_ids, dummy_attn_mask,
        age_ids, segment_ids, admission_loc_ids, discharge_loc_ids,
        gender_ids, ethnicity_ids, insurance_ids,
        aggregated_text_embeddings_t,
        labels_mortality, labels_readmission
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # ----- Hyperparameters Based on Processed Data -----
    # Here we use the number of unique values in each column to set the embedding sizes.
    disease_mapping = {d: i for i, d in enumerate(df_filtered["hadm_id"].unique())}
    NUM_DISEASES = len(disease_mapping)
    NUM_AGES = df_filtered["age"].nunique()
    NUM_SEGMENTS = 2
    NUM_ADMISSION_LOCS = df_filtered["FIRST_WARDID_struct"].nunique()  
    NUM_DISCHARGE_LOCS = df_filtered["LAST_WARDID_struct"].nunique()    
    NUM_GENDERS = df_filtered["GENDER_struct"].nunique()     
    NUM_ETHNICITIES = df_filtered["ETHNICITY_struct"].nunique()       
    NUM_INSURANCES = df_filtered["INSURANCE_struct"].nunique()

    print("\n--- Hyperparameters based on processed data ---")
    print("NUM_DISEASES:", NUM_DISEASES)
    print("NUM_AGES:", NUM_AGES)
    print("NUM_SEGMENTS:", NUM_SEGMENTS)
    print("NUM_ADMISSION_LOCS:", NUM_ADMISSION_LOCS)
    print("NUM_DISCHARGE_LOCS:", NUM_DISCHARGE_LOCS)
    print("NUM_GENDERS:", NUM_GENDERS)
    print("NUM_ETHNICITIES:", NUM_ETHNICITIES)
    print("NUM_INSURANCES:", NUM_INSURANCES)

    # ----- Initialize the BEHRT and Multimodal Models -----
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

    multimodal_model = MultimodalTransformer(
        text_embed_size=768,  # BioClinicalBERT outputs 768-dim embeddings
        BEHRT=behrt_model,
        device=device,
        hidden_size=512
    ).to(device)

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    readmission_pos_weight = get_pos_weight(df_filtered["readmission_within_30_days"], device)
    criterion_mortality = FocalLoss(gamma=2, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_readmission = FocalLoss(gamma=2, pos_weight=readmission_pos_weight, reduction='mean')

    # ----- Training Loop -----
    num_epochs = 5
    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = train_step(multimodal_model, dataloader, optimizer, device,
                                  criterion_mortality, criterion_readmission)
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)
        metrics, _, _ = evaluate_model(multimodal_model, dataloader, device, threshold=0.5)
        print(f"Metrics at threshold=0.5 after epoch {epoch+1}: {metrics}")

    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
