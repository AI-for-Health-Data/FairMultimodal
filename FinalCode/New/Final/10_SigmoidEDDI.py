import os
import sys
# Remove current directory from sys.path to avoid conflicts with local modules.
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv

DEBUG = True

# -----------------------------
# Loss and Utility Functions
# -----------------------------
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

def get_age_bucket(age):
    if 15 <= age <= 29:
        return "15-29"
    elif 30 <= age <= 49:
        return "30-49"
    elif 50 <= age <= 69:
        return "50-69"
    elif 70 <= age <= 90:
        return "70-90"
    else:
        return "Other"

def map_ethnicity(e):
    mapping = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian"}
    return mapping.get(e, "Other")

def map_insurance(i):
    mapping = {0: "Government", 1: "Medicare", 2: "Medicaid", 3: "Private", 4: "Self Pay"}
    return mapping.get(i, "Other")

def compute_eddi(y_true, y_pred, sensitive_labels, threshold=0.5):
    """
    Computes overall and subgroup EDDI.
    """
    y_pred_bin = (y_pred > threshold).astype(int)
    unique_groups = np.unique(sensitive_labels)
    subgroup_eddi = {}
    overall_error = np.mean(y_pred_bin != y_true)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            er_group = np.mean(y_pred_bin[mask] != y_true[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

# -----------------------------
# Models and Fusion
# -----------------------------
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean"):
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
                    max_length=512,
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
            agg_emb = np.mean(embeddings, axis=0) if aggregation=="mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings

class BEHRTModel_Demo(nn.Module):
    def __init__(self, num_ages, num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel_Demo, self).__init__()
        vocab_size = num_ages + num_genders + num_ethnicities + num_insurances + 2
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

class BEHRTModel_Lab(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768, nhead=8, num_layers=2):
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

class MultimodalTransformer_EDDI_Sigmoid(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device, fusion_hidden=512, beta=1.0):
        """
        Fusion Process:
          1. Each modality is projected into a 256-D space.
          2. The three modality projections are dynamically weighted and concatenated into a 768-D vector.
          3. A shared 768-D sigmoid gating vector is applied elementwise.
          4. The gated representation is fed through a fusion MLP to produce 3 logits.
        """
        super(MultimodalTransformer_EDDI_Sigmoid, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device
        self.beta = beta

        # Project each modality into 256-D.
        self.demo_projector = nn.Sequential(
            nn.Linear(behrt_demo.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        self.lab_projector = nn.Sequential(
            nn.Linear(behrt_lab.hidden_size, 256),
            nn.ReLU()
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )

        # Modality-specific classifiers for EDDI computation.
        self.classifier_demo = nn.Linear(256, 1)
        self.classifier_lab = nn.Linear(256, 1)
        self.classifier_text = nn.Linear(256, 1)

        # Shared sigmoid gating vector (768-D).
        self.sig_weights = nn.Parameter(torch.randn(768))

        # Fusion MLP mapping the 768-D gated vector to 3 logits.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden, 3)
        )

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding,
                beta=None, old_eddi_weights=None, return_modality_logits=False):
        if beta is None:
            beta = self.beta

        # Obtain unimodal embeddings.
        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        # Project each modality.
        demo_proj = self.demo_projector(demo_embedding)
        lab_proj = self.lab_projector(lab_embedding)
        text_proj = self.text_projector(text_embedding)

        # Retrieve dynamic EDDI weights (default to equal if not provided).
        if old_eddi_weights is None:
            weight_demo = 0.33
            weight_lab = 0.33
            weight_text = 0.33
        else:
            weight_demo = old_eddi_weights.get("demo", 0.33)
            weight_lab = old_eddi_weights.get("lab", 0.33)
            weight_text = old_eddi_weights.get("text", 0.33)

        # Apply dynamic weighting.
        weighted_demo = weight_demo * demo_proj
        weighted_lab = weight_lab * lab_proj
        weighted_text = weight_text * text_proj

        fused_vector = torch.cat([weighted_demo, weighted_lab, weighted_text], dim=1)  # shape (B, 768)
        # Apply shared sigmoid gating.
        sig_weights = torch.sigmoid(self.sig_weights)  # shape (768,)
        gated_vector = fused_vector * sig_weights  # element-wise modulation
        print(gated_vector.shape)
        # Final fusion MLP.
        fused_logits = self.fusion_mlp(gated_vector)
        

        outputs = {
            "fused_logits": fused_logits,
            "dynamic_weights": {"demo": weight_demo, "lab": weight_lab, "text": weight_text},
            "sigmoid_weights": sig_weights
        }

        if return_modality_logits:
            outputs["modality_logits"] = {
                'demo': self.classifier_demo(demo_proj),
                'lab': self.classifier_lab(lab_proj),
                'text': self.classifier_text(text_proj)
            }

        return outputs

# -----------------------------
# Weight Update and Training Steps
# -----------------------------
def update_dynamic_weights(model, dataloader, device, old_eddi_weights, beta, threshold=0.5):
    model.eval()
    all_demo_preds, all_lab_preds, all_text_preds, all_labels = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding, labels) = [x.to(device) for x in batch]
            outputs = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding,
                beta=model.beta, old_eddi_weights=old_eddi_weights, return_modality_logits=True
            )
            modality_logits = outputs["modality_logits"]
            demo_probs = torch.sigmoid(modality_logits["demo"]).squeeze().cpu().numpy()
            lab_probs = torch.sigmoid(modality_logits["lab"]).squeeze().cpu().numpy()
            text_probs = torch.sigmoid(modality_logits["text"]).squeeze().cpu().numpy()
            # Use the first outcome (e.g., mortality) for error calculation.
            batch_labels = labels[:, 0].squeeze().cpu().numpy()
            all_demo_preds.append(demo_probs)
            all_lab_preds.append(lab_probs)
            all_text_preds.append(text_probs)
            all_labels.append(batch_labels)
    
    all_demo_preds = np.concatenate(all_demo_preds)
    all_lab_preds = np.concatenate(all_lab_preds)
    all_text_preds = np.concatenate(all_text_preds)
    all_labels_np = np.concatenate(all_labels)
    
    error_demo = np.mean(np.abs(all_demo_preds - all_labels_np))
    error_lab = np.mean(np.abs(all_lab_preds - all_labels_np))
    error_text = np.mean(np.abs(all_text_preds - all_labels_np))
    overall_error = (error_demo + error_lab + error_text) / 3.0

    def compute_eddi_mod(error_mod, overall_error):
        denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
        return (error_mod - overall_error) / denom

    eddi_demo = compute_eddi_mod(error_demo, overall_error)
    eddi_lab = compute_eddi_mod(error_lab, overall_error)
    eddi_text = compute_eddi_mod(error_text, overall_error)
    eddi_max = max(eddi_demo, eddi_lab, eddi_text)

    print(f"[Weight Update] Computed EDDI: demo={eddi_demo:.4f}, lab={eddi_lab:.4f}, text={eddi_text:.4f}, eddi_max={eddi_max:.4f}")

    old_demo = old_eddi_weights.get("demo", 0.33) if old_eddi_weights is not None else 0.33
    old_lab = old_eddi_weights.get("lab", 0.33) if old_eddi_weights is not None else 0.33
    old_text = old_eddi_weights.get("text", 0.33) if old_eddi_weights is not None else 0.33

    new_weight_demo = old_demo + beta * (eddi_max - eddi_demo)
    new_weight_lab = old_lab + beta * (eddi_max - eddi_lab)
    new_weight_text = old_text + beta * (eddi_max - eddi_text)

    total = new_weight_demo + new_weight_lab + new_weight_text
    new_weight_demo /= total
    new_weight_lab /= total
    new_weight_text /= total

    print(f"[Weight Update] New dynamic weights (normalized): demo={new_weight_demo:.4f}, lab={new_weight_lab:.4f}, text={new_weight_text:.4f}")
    return {"demo": new_weight_demo, "lab": new_weight_lab, "text": new_weight_text}

def train_step(model, dataloader, optimizer, device, criterion, beta=1.0, lambda_edd=1.0, lambda_l1=0.01, target=1.0, threshold=0.5, old_eddi_weights=None):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features, aggregated_text_embedding,
         labels) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        outputs = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding,
            beta=beta, old_eddi_weights=old_eddi_weights, return_modality_logits=True
        )
        fused_logits = outputs["fused_logits"]
        modality_logits = outputs["modality_logits"]

        bce_loss = criterion(fused_logits, labels)

        eddi_losses = []
        for modality in ['demo', 'lab', 'text']:
            eddi_loss_mod = ((modality_logits[modality].squeeze() - target) ** 2).mean()
            eddi_losses.append(eddi_loss_mod)
        eddi_loss = torch.stack(eddi_losses).mean()

        # L1 regularization on the sigmoid gating vector.
        l1_reg = lambda_l1 * torch.sum(torch.abs(model.sig_weights))

        loss = bce_loss + lambda_edd * eddi_loss + l1_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

# -----------------------------
# Evaluation Functions (with Sensitive Data Return)
# -----------------------------
def calibrate_thresholds(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding,
             labels) = [x.to(device) for x in batch]
            outputs = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            logits = outputs["fused_logits"]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    outcome_names = ["mortality", "los", "mechanical_ventilation"]
    thresholds = {}
    for i, outcome in enumerate(outcome_names):
        probs = torch.sigmoid(all_logits[:, i]).numpy().squeeze()
        labels_np = all_labels[:, i].numpy().squeeze()
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.linspace(0, 1, 101):
            preds = (probs > t).astype(int)
            f1 = f1_score(labels_np, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        thresholds[outcome] = best_thresh
    return thresholds

def evaluate_model_multi(model, dataloader, device, thresholds, print_eddi=False):
    model.eval()
    all_logits = []
    all_labels = []
    all_age = []
    all_ethnicity = []
    all_insurance = []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding,
             labels) = [x.to(device) for x in batch]
            outputs = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            logits = outputs["fused_logits"]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_age = torch.cat(all_age, dim=0).numpy().squeeze()
    all_ethnicity = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    all_insurance = torch.cat(all_insurance, dim=0).numpy().squeeze()

    outcome_names = ["mortality", "los", "mechanical_ventilation"]
    metrics = {}
    for i, outcome in enumerate(outcome_names):
        thresh = thresholds[outcome] if isinstance(thresholds, dict) else thresholds
        probs = torch.sigmoid(all_logits[:, i]).numpy().squeeze()
        labels_np = all_labels[:, i].numpy().squeeze()
        preds = (probs > thresh).astype(int)
        try:
            aucroc = roc_auc_score(labels_np, probs)
        except Exception:
            aucroc = float('nan')
        try:
            auprc = average_precision_score(labels_np, probs)
        except Exception:
            auprc = float('nan')
        f1 = f1_score(labels_np, preds, zero_division=0)
        recall_val = recall_score(labels_np, preds, zero_division=0)
        precision_val = precision_score(labels_np, preds, zero_division=0)
        cm = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()
        if cm.size == 4:
            tn, fp, fn, tp = cm
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tpr = 0
            fpr = 0
        metrics[outcome] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                            "recall (TPR)": recall_val, "TPR": tpr,
                            "precision": precision_val, "fpr": fpr,
                            "optimal_threshold": thresh}
    # Return both metrics and sensitive attribute arrays.
    return metrics, all_logits.numpy(), all_labels.numpy(), all_age, all_ethnicity, all_insurance

def evaluate_model(model, dataloader, device, threshold=0.5, old_eddi_weights=None):
    metrics, logits_all, labels_all, age_all, ethnicity_all, insurance_all = evaluate_model_multi(model, dataloader, device, thresholds=threshold, print_eddi=True)
    return metrics, logits_all, labels_all, age_all, ethnicity_all, insurance_all

# -----------------------------
# Training Pipeline
# -----------------------------
def run_experiment(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    structured_data = pd.read_csv("final_structured_common.csv", low_memory=False)
    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)
    
    unstructured_data.drop(
        columns=["short_term_mortality", "los_binary", "mechanical_ventilation", "age",
                 "GENDER", "ETHNICITY", "INSURANCE"],
        errors='ignore',
        inplace=True
    )
    merged_df = pd.merge(
        structured_data,
        unstructured_data,
        on=["subject_id", "hadm_id"],
        how="inner",
        suffixes=("_struct", "_unstruct")
    ).head(10)
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check your merge keys.")

    merged_df["short_term_mortality"] = merged_df["short_term_mortality"].astype(int)
    merged_df["los_binary"] = merged_df["los_binary"].astype(int)
    merged_df["mechanical_ventilation"] = merged_df["mechanical_ventilation"].astype(int)
    
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    if "age" not in df_filtered.columns:
        if "Age" in df_filtered.columns:
            df_filtered.rename(columns={"Age": "age"}, inplace=True)
        else:
            df_filtered["age"] = 0

    print("Computing aggregated text embeddings for each patient...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    print("Aggregated text embeddings shape:", aggregated_text_embeddings_np.shape)
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    # Process demographic columns.
    demographics_cols = ["age", "GENDER", "ETHNICITY", "INSURANCE"]
    for col in demographics_cols:
        if col not in df_filtered.columns:
            print(f"Column {col} not found; creating default values.")
            df_filtered[col] = 0
        elif df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].astype("category").cat.codes

    exclude_cols = set(["subject_id", "ROW_ID", "hadm_id", "ICUSTAY_ID",
                        "short_term_mortality", "los_binary", "mechanical_ventilation",
                        "age", "GENDER", "ETHNICITY", "INSURANCE"])
    lab_feature_columns = [col for col in df_filtered.columns 
                           if col not in exclude_cols and not col.startswith("note_") 
                           and pd.api.types.is_numeric_dtype(df_filtered[col])]
    print("Number of lab feature columns:", len(lab_feature_columns))
    df_filtered[lab_feature_columns] = df_filtered[lab_feature_columns].fillna(0)

    lab_features_np = df_filtered[lab_feature_columns].values.astype(np.float32)
    lab_mean = np.mean(lab_features_np, axis=0)
    lab_std = np.std(lab_features_np, axis=0)
    lab_features_np = (lab_features_np - lab_mean) / (lab_std + 1e-6)

    num_samples = len(df_filtered)
    demo_dummy_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    demo_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    age_ids = torch.tensor(df_filtered["age"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_filtered["GENDER"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_filtered["ETHNICITY"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_filtered["INSURANCE"].values, dtype=torch.long)
    lab_features_t = torch.tensor(lab_features_np, dtype=torch.float32)
    labels_np = df_filtered[["short_term_mortality", "los_binary", "mechanical_ventilation"]].values.astype(np.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32)

    # Extract multilabel targets.
    labels_multilabel = df_filtered[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values

    # First split: train+validation vs. test (80/20 split).
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    for train_val_idx, test_idx in msss.split(df_filtered, labels_multilabel):
        train_val_df = df_filtered.iloc[train_val_idx]
        test_df = df_filtered.iloc[test_idx]

    # Second split: train vs. validation (e.g., 95/5 split on train+validation).
    labels_train_val = train_val_df[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    for train_idx, val_idx in msss_val.split(train_val_df, labels_train_val):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    def create_dataset(indices):
        return TensorDataset(
            demo_dummy_ids[indices], demo_attn_mask[indices],
            age_ids[indices], gender_ids[indices], ethnicity_ids[indices], insurance_ids[indices],
            lab_features_t[indices],
            aggregated_text_embeddings_t[indices],
            labels[indices]
        )

    train_dataset = create_dataset(train_idx)
    val_dataset = create_dataset(val_idx)
    test_dataset = create_dataset(test_idx)

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)

    train_df = df_filtered.iloc[train_idx]
    pos_weight_mort = compute_class_weights(train_df, "short_term_mortality")[1]
    pos_weight_los = compute_class_weights(train_df, "los_binary")[1]
    pos_weight_mech = compute_class_weights(train_df, "mechanical_ventilation")[1]
    pos_weight = torch.tensor([pos_weight_mort, pos_weight_los, pos_weight_mech], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    NUM_AGES = df_filtered["age"].nunique()
    NUM_GENDERS = df_filtered["GENDERS"].nunique() if "GENDERS" in df_filtered.columns else df_filtered["GENDER"].nunique()
    NUM_ETHNICITIES = df_filtered["ETHNICITY"].nunique()
    NUM_INSURANCES = df_filtered["INSURANCE"].nunique()
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

    beta_value = hparams['beta']

    multimodal_model = MultimodalTransformer_EDDI_Sigmoid(
        text_embed_size=768, 
        behrt_demo=behrt_demo,
        behrt_lab=behrt_lab,
        device=device,
        fusion_hidden=512,
        beta=beta_value
    ).to(device)

    optimizer = AdamW(multimodal_model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    old_eddi_weights = {"demo": 0.33, "lab": 0.33, "text": 0.33}
    
    tracked_dynamic_weights = []  # list of [demo, lab, text] per epoch
    tracked_sigmoid_weights = []    # shared 768-D vector per epoch

    dynamic_weights_csv = "dynamic_weights_per_epoch.csv"
    with open(dynamic_weights_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "demo_weight", "lab_weight", "text_weight"])

    max_epochs = hparams['num_epochs']
    for epoch in range(max_epochs):
        train_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                criterion, beta=beta_value, lambda_edd=hparams['lambda_edd'],
                                lambda_l1=hparams['lambda_l1'], target=1.0, threshold=hparams['threshold'],
                                old_eddi_weights=old_eddi_weights)
        avg_train_loss = train_loss / len(train_loader)
        
        multimodal_model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (demo_dummy_ids, demo_attn_mask,
                 age_ids, gender_ids, ethnicity_ids, insurance_ids,
                 lab_features, aggregated_text_embedding,
                 labels) = [x.to(device) for x in batch]
                outputs = multimodal_model(
                    demo_dummy_ids, demo_attn_mask,
                    age_ids, gender_ids, ethnicity_ids, insurance_ids,
                    lab_features, aggregated_text_embedding,
                    beta=beta_value, old_eddi_weights=old_eddi_weights
                )
                logits = outputs["fused_logits"]
                loss = criterion(logits, labels)
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = multimodal_model.state_dict()
            print("Validation loss improved. Saving model...")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} consecutive epochs.")
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

        new_weights = update_dynamic_weights(multimodal_model, train_loader, device, old_eddi_weights, beta=beta_value, threshold=hparams['threshold'])
        old_eddi_weights = new_weights
        print("Updated dynamic EDDI weights:", old_eddi_weights)
        
        tracked_dynamic_weights.append([old_eddi_weights["demo"], old_eddi_weights["lab"], old_eddi_weights["text"]])
        tracked_sigmoid_weights.append(torch.sigmoid(multimodal_model.sig_weights).detach().cpu().numpy())
        with open("eddi_sigmoid_weights.log", "a") as f:
            f.write(f"Epoch {epoch+1}: dynamic_weights = {old_eddi_weights}, "
                    f"sigmoid_weights = {tracked_sigmoid_weights[-1].tolist()}\n")
        
        with open(dynamic_weights_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, old_eddi_weights["demo"], old_eddi_weights["lab"], old_eddi_weights["text"]])
    
    print("Training complete.\n")
    
    if best_model_state is not None:
        multimodal_model.load_state_dict(best_model_state)

    val_thresholds = calibrate_thresholds(multimodal_model, val_loader, device)
    print("\nOptimal thresholds from validation:")
    for outcome, thresh in val_thresholds.items():
        print(f"{outcome}: {thresh:.2f}")

    # Evaluate model on test set and also retrieve sensitive attribute arrays.
    final_metrics, logits_all, labels_all, age_all, ethnicity_all, insurance_all = evaluate_model(multimodal_model, test_loader, device, threshold=val_thresholds)
    
    print("\n--- Final Evaluation Metrics on Test Set ---")
    for outcome, m in final_metrics.items():
        if outcome == "eddi_stats":
            continue
        print(f"\nOutcome: {outcome}")
        print("  AUROC     : {:.4f}".format(m["aucroc"]))
        print("  AUPRC     : {:.4f}".format(m["auprc"]))
        print("  F1 Score  : {:.4f}".format(m["f1"]))
        print("  Recall    : {:.4f}".format(m["recall (TPR)"]))
        print("  Precision : {:.4f}".format(m["precision"]))
        print("  TPR       : {:.4f}".format(m["TPR"]))
        print("  FPR       : {:.4f}".format(m["fpr"]))
        print("  Optimal Thresh: {:.2f}".format(m["optimal_threshold"]))
    
    # For each outcome, compute EDDI for each sensitive attribute.
    outcome_names = ["mortality", "los", "mechanical_ventilation"]
    print("\n--- Sensitive Subgroup EDDI Statistics ---")
    combined_eddi = {}
    for i, outcome in enumerate(outcome_names):
        # Get predicted probabilities for this outcome.
        probs = torch.sigmoid(torch.tensor(logits_all))[:, i].numpy().squeeze()
        true_vals = labels_all[:, i]
        # Use the calibrated threshold for binarization.
        thresh = val_thresholds[outcome]
        print(f"\nOutcome: {outcome} (Threshold: {thresh:.2f})")
        eddi_age, subgroup_age = compute_eddi(true_vals, probs, age_all, threshold=thresh)
        eddi_ethnicity, subgroup_ethnicity = compute_eddi(true_vals, probs, ethnicity_all, threshold=thresh)
        eddi_insurance, subgroup_insurance = compute_eddi(true_vals, probs, insurance_all, threshold=thresh)
        # Combine by averaging the overall EDDI for the three sensitive attributes.
        overall_combined = np.mean([eddi_age, eddi_ethnicity, eddi_insurance])
        combined_eddi[outcome] = overall_combined
        print(" Age EDDI:")
        print("  Overall:", eddi_age)
        print("  Subgroups:", subgroup_age)
        print(" Ethnicity EDDI:")
        print("  Overall:", eddi_ethnicity)
        print("  Subgroups:", subgroup_ethnicity)
        print(" Insurance EDDI:")
        print("  Overall:", eddi_insurance)
        print("  Subgroups:", subgroup_insurance)
        print(" Combined EDDI (average):", overall_combined)
    
    # Optionally, compute an overall combined EDDI across outcomes.
    overall_combined_eddi = np.mean(list(combined_eddi.values()))
    print("\n--- Overall Combined EDDI across outcomes ---")
    print("Overall Combined EDDI:", overall_combined_eddi)
    
    # Save tracked weight arrays.
    np.save("tracked_dynamic_weights.npy", np.array(tracked_dynamic_weights))
    np.save("tracked_sigmoid_weights.npy", np.array(tracked_sigmoid_weights))
    
    # Visualize dynamic weights heatmap.
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(tracked_dynamic_weights), annot=True, cmap="viridis", xticklabels=["demo", "lab", "text"])
    plt.title("Dynamic EDDI Weights per Epoch")
    plt.xlabel("Modality")
    plt.ylabel("Epoch")
    plt.tight_layout()
    plt.savefig("dynamic_weights_heatmap.png")
    plt.close()
    
    # Visualize sigmoid weights heatmap.
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.array(tracked_sigmoid_weights), cmap="viridis")
    plt.title("Sigmoid Weights per Epoch")
    plt.xlabel("Sigmoid Dimension")
    plt.ylabel("Epoch")
    plt.tight_layout()
    plt.savefig("sigmoid_weights_heatmap.png")
    plt.close()
    
    print("Heatmaps and weight arrays saved to disk.")
    return final_metrics

if __name__ == "__main__":
    hyperparameter_grid = [
        {'lr': 1e-5, 'num_epochs': 50, 'lambda_edd': 1.0, 'lambda_l1': 0.01, 'batch_size': 16, 'threshold': 0.50, 'weight_decay': 0.01, 'beta': 1.0},
    ]
    results = {}
    for idx, hparams in enumerate(hyperparameter_grid):
        print("\n==============================")
        print(f"Running experiment {idx+1} with hyperparameters: {hparams}")
        metrics = run_experiment(hparams)
        results[f"experiment_{idx+1}"] = metrics

    print("\n=== Hyperparameter Search Complete ===")
