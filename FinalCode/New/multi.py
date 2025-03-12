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

DEBUG = True

#############################################
# Loss Functions and Class Weight Computation
#############################################

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

#############################################
# BioClinicalBERT Fine-Tuning and Text Aggregation
#############################################

class BioClinicalBERT_FT(nn.Module):
    """
    A fine-tuning wrapper for BioClinicalBERT.
    Returns the CLS token embedding.
    """
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean"):
    """
    For each unique patient (by subject_id), extracts all non-null notes from the given note columns,
    tokenizes them, computes the CLS embedding via BioClinicalBERT_FT,
    and aggregates them (using mean or max).
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

#############################################
# BEHRT Models for Structured Data
#############################################

# Demographics Branch: uses recoded categorical values (age bucket, gender, ethnicity, insurance)
class BEHRTModel_Demo(nn.Module):
    def __init__(self, num_ages, num_genders, num_ethnicities, num_insurances, hidden_size=768):
        """
        A BEHRT-like model for demographic features.
        """
        super(BEHRTModel_Demo, self).__init__()
        vocab_size = num_ages + num_genders + num_ethnicities + num_insurances + 2
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=3072,
            max_position_embeddings=128,
            type_vocab_size=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(config)
        self.age_embedding = nn.Embedding(num_ages, hidden_size)
        self.gender_embedding = nn.Embedding(num_genders, hidden_size)
        self.ethnicity_embedding = nn.Embedding(num_ethnicities, hidden_size)
        self.insurance_embedding = nn.Embedding(num_insurances, hidden_size)

    def forward(self, input_ids, attention_mask, age_ids, gender_ids, ethnicity_ids, insurance_ids):
        age_ids = torch.clamp(age_ids, 0, self.age_embedding.num_embeddings - 1)
        gender_ids = torch.clamp(gender_ids, 0, self.gender_embedding.num_embeddings - 1)
        ethnicity_ids = torch.clamp(ethnicity_ids, 0, self.ethnicity_embedding.num_embeddings - 1)
        insurance_ids = torch.clamp(insurance_ids, 0, self.insurance_embedding.num_embeddings - 1)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]

        age_embeds = self.age_embedding(age_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        eth_embeds = self.ethnicity_embedding(ethnicity_ids)
        ins_embeds = self.insurance_embedding(insurance_ids)
        extra = (age_embeds + gender_embeds + eth_embeds + ins_embeds) / 4.0
        demo_embedding = cls_token + extra
        return demo_embedding

# Lab Features Branch: projects and processes lab feature tokens through a transformer encoder.
class BEHRTModel_Lab(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768, nhead=8, num_layers=2):
        """
        A BEHRT-like model for lab features.
        Each lab feature (a scalar) is projected and then passed through a Transformer encoder.
        """
        super(BEHRTModel_Lab, self).__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Linear(1, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(lab_token_count, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, lab_features):
        x = lab_features.unsqueeze(-1)
        x = self.token_embedding(x)
        x = x + self.pos_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        lab_embedding = x.mean(dim=1)
        return lab_embedding

#############################################
# Multimodal Transformer: Fusion for Three Outcomes
#############################################

class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device):
        super(MultimodalTransformer, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device

        # Separate projection networks for each outcome
        # Mortality projections
        self.demo_projector_mort = nn.Sequential(
            nn.Linear(behrt_demo.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        self.lab_projector_mort = nn.Sequential(
            nn.Linear(behrt_lab.hidden_size, 256),
            nn.ReLU()
        )
        self.text_projector_mort = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )
        # LOS projections
        self.demo_projector_los = nn.Sequential(
            nn.Linear(behrt_demo.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        self.lab_projector_los = nn.Sequential(
            nn.Linear(behrt_lab.hidden_size, 256),
            nn.ReLU()
        )
        self.text_projector_los = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )
        # Mechanical Ventilation projections
        self.demo_projector_mech = nn.Sequential(
            nn.Linear(behrt_demo.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        self.lab_projector_mech = nn.Sequential(
            nn.Linear(behrt_lab.hidden_size, 256),
            nn.ReLU()
        )
        self.text_projector_mech = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )

    def _compute_logit(self, demo_proj, lab_proj, text_proj, beta):
        demo_out = torch.sum(demo_proj, dim=1, keepdim=True)
        lab_out  = torch.sum(lab_proj, dim=1, keepdim=True)
        text_out = torch.sum(text_proj, dim=1, keepdim=True)
        modalities = torch.cat([demo_out, lab_out, text_out], dim=1)
        max_val, _ = torch.max(modalities, dim=1, keepdim=True)
        weight_demo = 0.33 + beta * (max_val - demo_out)
        weight_lab  = 0.33 + beta * (max_val - lab_out)
        weight_text = 0.33 + beta * (max_val - text_out)
        logit = demo_out * weight_demo + lab_out * weight_lab + text_out * weight_text
        return logit

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=1.0):
        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding  # pre-computed

        # Mortality prediction branch
        demo_proj_mort = self.demo_projector_mort(demo_embedding)
        lab_proj_mort = self.lab_projector_mort(lab_embedding)
        text_proj_mort = self.text_projector_mort(text_embedding)
        mortality_logit = self._compute_logit(demo_proj_mort, lab_proj_mort, text_proj_mort, beta)

        # LOS prediction branch
        demo_proj_los = self.demo_projector_los(demo_embedding)
        lab_proj_los = self.lab_projector_los(lab_embedding)
        text_proj_los = self.text_projector_los(text_embedding)
        los_logit = self._compute_logit(demo_proj_los, lab_proj_los, text_proj_los, beta)

        # Mechanical Ventilation prediction branch
        demo_proj_mech = self.demo_projector_mech(demo_embedding)
        lab_proj_mech = self.lab_projector_mech(lab_embedding)
        text_proj_mech = self.text_projector_mech(text_embedding)
        mechvent_logit = self._compute_logit(demo_proj_mech, lab_proj_mech, text_proj_mech, beta)

        return mortality_logit, los_logit, mechvent_logit

#############################################
# Training and Evaluation Functions
#############################################

def train_step(model, dataloader, optimizer, device, 
               criterion_mort, criterion_los, criterion_mech,
               beta=1.0, loss_gamma=1.0, target=1.0):
    """
    Training step with loss = BCE_mortality + BCE_los + BCE_mechvent + loss_gamma * EDDI_loss.
    EDDI_loss is computed on the mortality branch.
    """
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]

        optimizer.zero_grad()

        mortality_logit, los_logit, mechvent_logit = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding, beta=beta
        )

        loss_mort = criterion_mort(mortality_logit, labels_mortality.unsqueeze(1))
        loss_los  = criterion_los(los_logit, labels_los.unsqueeze(1))
        loss_mech = criterion_mech(mechvent_logit, labels_mechvent.unsqueeze(1))
        eddi_loss = ((mortality_logit - target) ** 2).mean()  
        loss = loss_mort + loss_los + loss_mech + loss_gamma * eddi_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def compute_eddi(true_labels, predicted_labels, sensitive_labels):
    """
    Computes the EDDI for a given sensitive attribute.
    Returns attribute-level EDDI and subgroup EDDI values.
    """
    unique_groups = np.unique(sensitive_labels)
    subgroup_eddi = {}
    overall_error = np.mean(predicted_labels != true_labels)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            group_error = np.mean(predicted_labels[mask] != true_labels[mask])
            subgroup_eddi[group] = (group_error - overall_error) / denom
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values()))**2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

def evaluate_model(model, dataloader, device, threshold=0.5, print_eddi=False):
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_mech_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_labels_mech = []
    all_age = []
    all_ethnicity = []
    all_insurance = []
    
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]
            mort_logits, los_logits, mech_logits = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_mech_logits.append(mech_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_mech.append(labels_mechvent.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_los_logits  = torch.cat(all_los_logits, dim=0)
    all_mech_logits = torch.cat(all_mech_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_los  = torch.cat(all_labels_los, dim=0)
    all_labels_mech = torch.cat(all_labels_mech, dim=0)
    
    mort_probs = torch.sigmoid(all_mort_logits).numpy().squeeze()
    los_probs  = torch.sigmoid(all_los_logits).numpy().squeeze()
    mech_probs = torch.sigmoid(all_mech_logits).numpy().squeeze()
    labels_mort_np = all_labels_mort.numpy().squeeze()
    labels_los_np  = all_labels_los.numpy().squeeze()
    labels_mech_np = all_labels_mech.numpy().squeeze()
    
    metrics = {}
    for task, probs, labels in zip(["mortality", "los", "mechanical_ventilation"],
                                    [mort_probs, los_probs, mech_probs],
                                    [labels_mort_np, labels_los_np, labels_mech_np]):
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
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall, "precision": precision}
    
    # For EDDI analysis, use sensitive attribute codes (converted to strings for simplicity)
    ages = torch.cat(all_age, dim=0).numpy().squeeze()
    ethnicities = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    insurances = torch.cat(all_insurance, dim=0).numpy().squeeze()
    
    age_groups = np.array([str(a) for a in ages])
    ethnicity_groups = np.array([str(e) for e in ethnicities])
    insurance_groups = np.array([str(i) for i in insurances])
    
    eddi_stats = {}
    for task, probs, labels in zip(["mortality", "los", "mechanical_ventilation"],
                                    [mort_probs, los_probs, mech_probs],
                                    [labels_mort_np, labels_los_np, labels_mech_np]):
        eddi_age, age_eddi_sub = compute_eddi(labels.astype(int), (probs > threshold).astype(int), age_groups)
        eddi_eth, eth_eddi_sub = compute_eddi(labels.astype(int), (probs > threshold).astype(int), ethnicity_groups)
        eddi_ins, ins_eddi_sub = compute_eddi(labels.astype(int), (probs > threshold).astype(int), insurance_groups)
        eddi_stats[task] = {
            "age_eddi": eddi_age,
            "age_subgroup_eddi": age_eddi_sub,
            "ethnicity_eddi": eddi_eth,
            "ethnicity_subgroup_eddi": eth_eddi_sub,
            "insurance_eddi": eddi_ins,
            "insurance_subgroup_eddi": ins_eddi_sub
        }
    metrics["eddi_stats"] = eddi_stats
    
    if print_eddi:
        for task in ["mortality", "los", "mechanical_ventilation"]:
            print(f"\n--- EDDI Calculation for {task.capitalize()} Outcome ---")
            eddi = eddi_stats[task]
            print("  Age EDDI:", eddi["age_eddi"])
            print("  Ethnicity EDDI:", eddi["ethnicity_eddi"])
            print("  Insurance EDDI:", eddi["insurance_eddi"])
    
    unique_subgroups = {"age": list(np.unique(age_groups)),
                        "ethnicity": list(np.unique(ethnicity_groups)),
                        "insurance": list(np.unique(insurance_groups))}
    
    return metrics, all_mort_logits, unique_subgroups

#############################################
# Main Training Pipeline
#############################################

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load merged datasets (as created by your pre-processing pipeline)
    structured_data = pd.read_csv('final_structured_common.csv')
    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)

    # Drop duplicate outcome/demographic columns from unstructured data if present.
    unstructured_data.drop(
        columns=["short_term_mortality", "los_binary", "mechanical_ventilation", "age",
                 "gender", "ethnicity_category", "insurance_category"],
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
        raise ValueError("Merged DataFrame is empty. Check your merge keys.")

    merged_df["short_term_mortality"] = merged_df["short_term_mortality"].astype(int)
    merged_df["los_binary"] = merged_df["los_binary"].astype(int)
    merged_df["mechanical_ventilation"] = merged_df["mechanical_ventilation"].astype(int)

    # Identify note columns (columns that start with "note_")
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    # Compute Aggregated Text Embeddings (Text Branch)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    # Process Structured Data: Demographics and Lab Features
    # Use recoded columns: age_bucket, gender, ethnicity_category, insurance_category.
    demographics_cols = ["age_bucket", "gender", "ethnicity_category", "insurance_category"]
    for col in demographics_cols:
        if col not in df_filtered.columns:
            print(f"Column {col} not found; creating default values.")
            df_filtered[col] = "unknown"
        df_filtered[col] = df_filtered[col].astype("category")
        df_filtered[col + "_code"] = df_filtered[col].cat.codes

    exclude_cols = set(["subject_id", "ROW_ID", "hadm_id", "ICUSTAY_ID",
                        "short_term_mortality", "los_binary", "mechanical_ventilation",
                        "age", "age_bucket", "gender", "ethnicity_category", "insurance_category"])
    lab_feature_columns = [col for col in df_filtered.columns 
                           if col not in exclude_cols and not col.startswith("note_") 
                           and pd.api.types.is_numeric_dtype(df_filtered[col])]
    print("Number of lab feature columns:", len(lab_feature_columns))
    df_filtered[lab_feature_columns] = df_filtered[lab_feature_columns].fillna(0)

    # Normalize Lab Features 
    lab_features_np = df_filtered[lab_feature_columns].values.astype(np.float32)
    lab_mean = np.mean(lab_features_np, axis=0)
    lab_std = np.std(lab_features_np, axis=0)
    lab_features_np = (lab_features_np - lab_mean) / (lab_std + 1e-6)

    # Create Inputs for Each Branch
    num_samples = len(df_filtered)
    demo_dummy_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    demo_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    age_ids = torch.tensor(df_filtered["age_bucket_code"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_filtered["gender_code"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_filtered["ethnicity_category_code"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_filtered["insurance_category_code"].values, dtype=torch.long)
    lab_features_t = torch.tensor(lab_features_np, dtype=torch.float32)
    labels_mortality = torch.tensor(df_filtered["short_term_mortality"].values, dtype=torch.float32)
    labels_los = torch.tensor(df_filtered["los_binary"].values, dtype=torch.float32)
    labels_mechvent = torch.tensor(df_filtered["mechanical_ventilation"].values, dtype=torch.float32)

    dataset = TensorDataset(
        demo_dummy_ids, demo_attn_mask,
        age_ids, gender_ids, ethnicity_ids, insurance_ids,
        lab_features_t,
        aggregated_text_embeddings_t,
        labels_mortality, labels_los, labels_mechvent
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Get number of unique codes for each demographic feature.
    NUM_AGES = df_filtered["age_bucket_code"].nunique()
    NUM_GENDERS = df_filtered["gender_code"].nunique()
    NUM_ETHNICITIES = df_filtered["ethnicity_category_code"].nunique()
    NUM_INSURANCES = df_filtered["insurance_category_code"].nunique()
    print("\n--- Demographics Hyperparameters ---")
    print("NUM_AGES:", NUM_AGES)
    print("NUM_GENDERS:", NUM_GENDERS)
    print("NUM_ETHNICITIES:", NUM_ETHNICITIES)
    print("NUM_INSURANCES:", NUM_INSURANCES)
    NUM_LAB_FEATURES = len(lab_feature_columns)
    print("NUM_LAB_FEATURES (tokens):", NUM_LAB_FEATURES)

    behrt_demo = BEHRTModel_Demo(
        num_ages=NUM_AGES,
        num_genders=NUM_GENDERS,
        num_ethnicities=NUM_ETHNICITIES,
        num_insurances=NUM_INSURANCES,
        hidden_size=768
    ).to(device)
    behrt_lab = BEHRTModel_Lab(
        lab_token_count=NUM_LAB_FEATURES,
        hidden_size=768,
        nhead=8,
        num_layers=2
    ).to(device)

    multimodal_model = MultimodalTransformer(
        text_embed_size=768,
        behrt_demo=behrt_demo,
        behrt_lab=behrt_lab,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Compute separate positive weights for each outcome.
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mechvent_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    
    # Create separate loss functions for each outcome.
    criterion_mort = FocalLoss(gamma=2, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los  = FocalLoss(gamma=2, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=2, pos_weight=mechvent_pos_weight, reduction='mean')

    max_epochs = 20
    patience_limit = 5  
    beta = 0.3  # Use a fixed beta value
    loss_gamma = 1

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        running_loss = train_step(multimodal_model, dataloader, optimizer, device,
                                  criterion_mort, criterion_los, criterion_mech,
                                  beta=beta, loss_gamma=loss_gamma, target=1.0)
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)
        metrics, _, _ = evaluate_model(multimodal_model, dataloader, device, threshold=0.5)
        print(f"Metrics at threshold=0.5 after epoch {epoch+1}: {metrics}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience_limit:
            print(f"No improvement for {patience_limit} consecutive epochs. Early stopping.")
            break

    print("Training complete.")
    
    # Final evaluation with detailed EDDI statistics.
    final_metrics, _, unique_subgroups = evaluate_model(multimodal_model, dataloader, device, threshold=0.5, print_eddi=True)
    print("\n--- Final Evaluation Metrics ---")
    for task in ["mortality", "los", "mechanical_ventilation"]:
        print(f"\n{task.capitalize()}:")
        print("  AUROC     : {:.4f}".format(final_metrics[task]["aucroc"]))
        print("  AUPRC     : {:.4f}".format(final_metrics[task]["auprc"]))
        print("  F1 Score  : {:.4f}".format(final_metrics[task]["f1"]))
        print("  Recall    : {:.4f}".format(final_metrics[task]["recall"]))
        print("  Precision : {:.4f}".format(final_metrics[task]["precision"]))

    print("\n--- Final EDDI Statistics ---")
    for task in ["mortality", "los", "mechanical_ventilation"]:
        eddi = final_metrics["eddi_stats"][task]
        print(f"\nTask: {task.capitalize()}")
        print("  Age subgroup EDDI      :", eddi["age_subgroup_eddi"])
        print("  Aggregated Age EDDI    : {:.4f}".format(eddi["age_eddi"]))
        print("  Ethnicity subgroup EDDI:", eddi["ethnicity_subgroup_eddi"])
        print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi["ethnicity_eddi"]))
        print("  Insurance subgroup EDDI:", eddi["insurance_subgroup_eddi"])
        print("  Aggregated Insurance EDDI: {:.4f}".format(eddi["insurance_eddi"]))

    print("\n--- Unique Subgroups ---")
    print("Age subgroups      :", unique_subgroups["age"])
    print("Ethnicity subgroups:", unique_subgroups["ethnicity"])
    print("Insurance subgroups:", unique_subgroups["insurance"])

    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
