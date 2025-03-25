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

DEBUG = True

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

def compute_eddi(y_true, y_pred, sensitive_labels):
    """
    Computes subgroup-level EDDI and returns overall fairness metric.
    """
    unique_groups = np.unique(sensitive_labels)
    subgroup_eddi = {}
    overall_error = np.mean(y_pred != y_true)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            er_group = np.mean(y_pred[mask] != y_true[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values()))**2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

# BioClinicalBERT Fine-Tuning for Clinical Notes
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
            agg_emb = np.mean(embeddings, axis=0) if aggregation == "mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings

# BEHRT Model for Demographics (Structured Data)
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

# BEHRT Model for Lab Data
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

# Multimodal Fusion Model with Dynamic EDDI+Sigmoid Fusion
class MultimodalTransformer_EDDI_Sigmoid(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device, fusion_hidden=512):
        """
        Fusion process:
          1. Project each modality (demo, lab, text) into 256-D.
          2. Compute a scalar (sum) for each modality.
          3. Compute dynamic weights: weight = 0.33 + beta * (max_scalar - modality_scalar).
          4. Multiply each unimodal projection by its dynamic weight.
          5. Concatenate all three weighted 256-D vectors to form a 768-D vector.
          6. Modulate the concatenated vector elementwise using a single sigmoid layer whose weights are L1-regularized.
          7. Pass the result through a fusion MLP to output three logits.
        """
        super(MultimodalTransformer_EDDI_Sigmoid, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device

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
        
        # One sigmoid layer for the concatenated 768-D vector.
        self.sig_weights = nn.Parameter(torch.randn(768))
        
        # Initialize dynamic scalar weights for each modality.
        self.register_buffer('dynamic_weight_demo', torch.tensor(0.33))
        self.register_buffer('dynamic_weight_lab', torch.tensor(0.33))
        self.register_buffer('dynamic_weight_text', torch.tensor(0.33))
        
        # Modality-specific classifiers (for dynamic weight update via fairness).
        self.classifier_demo = nn.Linear(256, 3)
        self.classifier_lab = nn.Linear(256, 3)
        self.classifier_text = nn.Linear(256, 3)
        
        # Fusion MLP mapping 768-D input to 3 logits.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden, 3)
        )

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=1.0, return_modality_logits=False):
        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        # Project each modality into 256-D.
        demo_proj = self.demo_projector(demo_embedding)  
        lab_proj = self.lab_projector(lab_embedding)       
        text_proj = self.text_projector(text_embedding)    

        # Compute scalar (sum) for each modality.
        demo_scalar = torch.sum(demo_proj, dim=1, keepdim=True)  
        lab_scalar = torch.sum(lab_proj, dim=1, keepdim=True)    
        text_scalar = torch.sum(text_proj, dim=1, keepdim=True)  
        modality_scalars = torch.cat([demo_scalar, lab_scalar, text_scalar], dim=1)  
        max_scalar, _ = torch.max(modality_scalars, dim=1, keepdim=True)  

        # Compute dynamic weights based on the scalar values.
        weight_demo = 0.33 + beta * (max_scalar - demo_scalar)
        weight_lab  = 0.33 + beta * (max_scalar - lab_scalar)
        weight_text = 0.33 + beta * (max_scalar - text_scalar)

        # Multiply unimodal projections by their dynamic weights.
        weighted_demo = weight_demo * demo_proj  
        weighted_lab = weight_lab * lab_proj      
        weighted_text = weight_text * text_proj   

        # For dynamic weight update, compute modality-specific predictions.
        modality_logits = {
            'demo': self.classifier_demo(demo_proj),
            'lab': self.classifier_lab(lab_proj),
            'text': self.classifier_text(text_proj)
        }

        # Concatenate weighted vectors → 768-D vector.
        concat_vec = torch.cat([weighted_demo, weighted_lab, weighted_text], dim=1)  

        # Apply a single sigmoid layer elementwise (modulation with L1 reg on weights).
        adjusted = concat_vec * torch.sigmoid(self.sig_weights)  

        # Fusion MLP to obtain final 3 logits.
        fused_logits = self.fusion_mlp(adjusted)  

        if return_modality_logits:
            return fused_logits, modality_logits, (demo_scalar, lab_scalar, text_scalar, max_scalar)
        else:
            return fused_logits, adjusted

# Calibration and Evaluation Functions
def calibrate_thresholds(model, dataloader, device):
    """
    Calibrates an optimal threshold for each outcome based on F1 score using the validation set.
    """
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels) = [x.to(device) for x in batch]
            logits, _ = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
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
             lab_features,
             aggregated_text_embedding,
             labels) = [x.to(device) for x in batch]
            logits, _ = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
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
        cm = confusion_matrix(labels_np, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tpr = 0
            fpr = 0
        precision_val = precision_score(labels_np, preds, zero_division=0)
        metrics[outcome] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                            "recall (TPR)": recall_val, "TPR": tpr,
                            "precision": precision_val, "fpr": fpr,
                            "optimal_threshold": thresh}
    if print_eddi:
        ages = torch.cat(all_age, dim=0).numpy().squeeze()
        ethnicities = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
        insurances = torch.cat(all_insurance, dim=0).numpy().squeeze()
        age_groups = np.array([get_age_bucket(a) for a in ages])
        eth_groups = np.array([map_ethnicity(e) for e in ethnicities])
        ins_groups = np.array([map_insurance(i) for i in insurances])
        eddi_stats_all = {}
        for i, outcome in enumerate(outcome_names):
            labels_np = all_labels[:, i].numpy().astype(int)
            preds_outcome = (torch.sigmoid(all_logits[:, i]).numpy() > thresholds[outcome]).astype(int)
            overall_age, age_eddi_sub = compute_eddi(labels_np, preds_outcome, age_groups)
            overall_eth, eth_eddi_sub = compute_eddi(labels_np, preds_outcome, eth_groups)
            overall_ins, ins_eddi_sub = compute_eddi(labels_np, preds_outcome, ins_groups)
            total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
            eddi_stats_all[outcome] = {
                "age_eddi": overall_age,
                "age_subgroup_eddi": age_eddi_sub,
                "ethnicity_eddi": overall_eth,
                "ethnicity_subgroup_eddi": eth_eddi_sub,
                "insurance_eddi": overall_ins,
                "insurance_subgroup_eddi": ins_eddi_sub,
                "final_EDDI": total_eddi
            }
            print(f"\n--- Fairness (EDDI) Statistics for {outcome} ---")
            print("Final Overall EDDI:", total_eddi)
        metrics["eddi_stats"] = eddi_stats_all
    return metrics

# Training and Evaluation Functions
def train_step(model, dataloader, optimizer, device, criterion, beta=1.0, lambda_edd=1.0, lambda_l1=0.01, target=1.0, threshold=0.5):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features,
         aggregated_text_embedding,
         labels) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        fused_logits, modality_logits, _ = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding,
            beta=1.0, return_modality_logits=True
        )
        # Calculate BCE loss per label (criterion handles the multi-label setting)
        bce_loss = criterion(fused_logits, labels)
        
        # Compute EDDI loss per label 
        eddi_losses = []
        num_labels = fused_logits.shape[1]
        for i in range(num_labels):
            eddi_loss_i = ((fused_logits[:, i] - target) ** 2).mean()
            eddi_losses.append(eddi_loss_i)
        eddi_loss = torch.stack(eddi_losses).mean()
        
        # L1 regularization on the sigmoid layer's weights.
        l1_reg = lambda_l1 * torch.sum(torch.abs(model.sig_weights))
        loss = bce_loss + lambda_edd * eddi_loss + l1_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate_model_multi_wrapper(model, dataloader, device, thresholds, print_eddi=False):
    return evaluate_model_multi(model, dataloader, device, thresholds, print_eddi)

# Training Pipeline with Hyperparameter Grid and Modified Training Procedure
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
    )
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

    # Process demographic columns
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

    # Create a combined stratification key for multi-label stratification
    df_filtered["label_combo"] = (df_filtered["short_term_mortality"].astype(str) + "_" +
                                  df_filtered["los_binary"].astype(str) + "_" +
                                  df_filtered["mechanical_ventilation"].astype(str))
    stratify_col = df_filtered["label_combo"]

    indices = np.arange(num_samples)
    # Split into 80% train and 20% test using stratification on the combined label
    train_idx, test_idx = train_test_split(indices, test_size=0.20, stratify=stratify_col, random_state=42)
    # From the train set, split 5% as validation (relative to train)
    stratify_train = stratify_col.iloc[train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=0.05, stratify=stratify_train, random_state=42)

    # Create TensorDatasets for each split
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

    # Compute class weights on the training set using INS method for each outcome
    train_df = df_filtered.iloc[train_idx]
    pos_weight_mort = compute_class_weights(train_df, "short_term_mortality")[1]
    pos_weight_los = compute_class_weights(train_df, "los_binary")[1]
    pos_weight_mech = compute_class_weights(train_df, "mechanical_ventilation")[1]
    pos_weight = torch.tensor([pos_weight_mort, pos_weight_los, pos_weight_mech], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    NUM_AGES = df_filtered["age"].nunique()
    NUM_GENDERS = df_filtered["GENDER"].nunique()
    NUM_ETHNICITIES = df_filtered["ETHNICITY"].nunique()
    NUM_INSURANCES = df_filtered["INSURANCE"].nunique()
    print("\n--- Demographics Hyperparameters ---")
    print("NUM_AGES:", NUM_AGES)
    print("NUM_GENDERS:", NUM_GENDERS)
    print("NUM_ETHNICITIES:", NUM_ETHNICITIES)
    print("NUM_INSURANCES:", NUM_INSURANCES)
    NUM_LAB_FEATURES = len(lab_feature_columns)
    print("NUM_LAB_FEATURES (tokens):", NUM_LAB_FEATURES)

    # Instantiate models
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

    multimodal_model = MultimodalTransformer_EDDI_Sigmoid(
        text_embed_size=768, 
        behrt_demo=behrt_demo,
        behrt_lab=behrt_lab,
        device=device,
        fusion_hidden=512
    ).to(device)

    optimizer = AdamW(multimodal_model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(hparams['num_epochs']):
        train_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                  criterion, lambda_edd=hparams['lambda_edd'], lambda_l1=hparams['lambda_l1'],
                                  target=1.0, threshold=hparams['threshold'])
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation set
        multimodal_model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (demo_dummy_ids, demo_attn_mask,
                 age_ids, gender_ids, ethnicity_ids, insurance_ids,
                 lab_features,
                 aggregated_text_embedding,
                 labels) = [x.to(device) for x in batch]
                logits, _ = multimodal_model(
                    demo_dummy_ids, demo_attn_mask,
                    age_ids, gender_ids, ethnicity_ids, insurance_ids,
                    lab_features, aggregated_text_embedding
                )
                loss = criterion(logits, labels)
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        # Early stopping check: if validation loss doesn't improve for 5 consecutive epochs, stop training.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = multimodal_model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

    # Load best model state before evaluation
    if best_model_state is not None:
        multimodal_model.load_state_dict(best_model_state)

    # Calibrate thresholds using the validation set.
    val_thresholds = calibrate_thresholds(multimodal_model, val_loader, device)
    print("\nOptimal thresholds from validation:")
    for outcome, thresh in val_thresholds.items():
        print(f"{outcome}: {thresh:.2f}")

    # Final evaluation on test set using calibrated thresholds.
    metrics = evaluate_model_multi_wrapper(multimodal_model, test_loader, device, thresholds=val_thresholds, print_eddi=True)
    
    print("\n--- Final Evaluation Metrics on Test Set ---")
    for outcome, metric_vals in metrics.items():
        if outcome == "eddi_stats":
            continue
        print(f"\nOutcome: {outcome}")
        print("AUROC           : {:.4f}".format(metric_vals["aucroc"]))
        print("AUPRC           : {:.4f}".format(metric_vals["auprc"]))
        print("F1 Score        : {:.4f}".format(metric_vals["f1"]))
        print("Recall (TPR)    : {:.4f}".format(metric_vals["recall (TPR)"]))
        print("Calculated TPR  : {:.4f}".format(metric_vals["TPR"]))
        print("Precision       : {:.4f}".format(metric_vals["precision"]))
        print("FPR             : {:.4f}".format(metric_vals["fpr"]))
        print("Optimal thresh  : {:.2f}".format(metric_vals["optimal_threshold"]))
    
    if "eddi_stats" in metrics:
        print("\n--- Detailed Fairness (EDDI) Statistics ---")
        for outcome, eddi_dict in metrics["eddi_stats"].items():
            print(f"\nOutcome: {outcome}")
            for key, value in eddi_dict.items():
                print(f"{key}: {value}")
    
    return metrics

if __name__ == "__main__":
    # Define a grid of hyperparameters.
    hyperparameter_grid = [
        {'lr': 1e-5, 'num_epochs': 50, 'lambda_edd': 1.0, 'lambda_l1': 0.01, 'batch_size': 16, 'threshold': 0.50, 'weight_decay': 0.01},
    ]
    
    results = {}
    for idx, hparams in enumerate(hyperparameter_grid):
        print("\n==============================")
        print(f"Running experiment {idx+1} with hyperparameters: {hparams}")
        metrics = run_experiment(hparams)
        results[f"experiment_{idx+1}"] = metrics

    print("\n=== Hyperparameter Search Complete ===")
