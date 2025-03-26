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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit 

DEBUG = True

# Focal Loss definition
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

# Fairness EDDI function used in evaluation
def compute_eddi(y_true, y_pred, sensitive_labels, threshold=0.5):
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
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values()))**2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

# Utility functions for positive weighting and subgroup mapping
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
        return "Other"

def map_ethnicity(code):
    mapping = {0: "white", 1: "black", 2: "asian", 3: "hispanic"}
    return mapping.get(code, "other")

def map_insurance(code):
    mapping = {0: "government", 1: "medicare", 2: "Medicaid", 3: "private", 4: "self pay"}
    return mapping.get(code, "other")

# BioClinicalBERT Fine-Tuning for Note Aggregation
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
            agg_emb = np.mean(embeddings, axis=0) if aggregation=="mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings

# BEHRT Models for Structured Data
class BEHRTModel_Demo(nn.Module):
    def __init__(self, num_ages, num_genders, num_ethnicities, num_insurances, hidden_size=768):
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

# Updated Multimodal Transformer for Fusion with Modality Weighting Using EDDI
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device):
        super(MultimodalTransformer, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device

        # Projectors: map each modality embedding to a 256-dimensional space.
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

        # Classification layers for each modality (for obtaining y_pred per modality).
        self.classifier_demo = nn.Linear(256, 1)
        self.classifier_lab = nn.Linear(256, 1)
        self.classifier_text = nn.Linear(256, 1)
 
    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=1.0,
                y_true=None, sensitive_labels=None):

        # Get modality embeddings.
        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        # Project each modality to 256-dim.
        demo_proj = self.demo_projector(demo_embedding)
        lab_proj  = self.lab_projector(lab_embedding)
        text_proj = self.text_projector(text_embedding)

        # Obtain modality-specific classification logits.
        demo_logit = self.classifier_demo(demo_proj)  # shape: [batch, 1]
        lab_logit  = self.classifier_lab(lab_proj)    
        text_logit = self.classifier_text(text_proj)    

        # Convert logits to probabilities.
        demo_prob = torch.sigmoid(demo_logit)
        lab_prob  = torch.sigmoid(lab_logit)
        text_prob = torch.sigmoid(text_logit)

        if (y_true is not None) and (sensitive_labels is not None):
            # Convert tensors to numpy arrays.
            demo_prob_np = demo_prob.detach().cpu().numpy().squeeze()
            lab_prob_np  = lab_prob.detach().cpu().numpy().squeeze()
            text_prob_np = text_prob.detach().cpu().numpy().squeeze()
            # Compute fairness EDDI for each modality 
            eddi_demo_val, _ = compute_eddi(y_true, demo_prob_np, sensitive_labels, threshold=0.5)
            eddi_lab_val, _  = compute_eddi(y_true, lab_prob_np, sensitive_labels, threshold=0.5)
            eddi_text_val, _ = compute_eddi(y_true, text_prob_np, sensitive_labels, threshold=0.5)
        else:
            eddi_demo_val = 0.0
            eddi_lab_val = 0.0
            eddi_text_val = 0.0

        # Determine the maximum EDDI across modalities.
        eddi_max_val = max(eddi_demo_val, eddi_lab_val, eddi_text_val)

        #   weight_mod = 0.33 + beta * (eddi_max_val - eddi_mod)
        weight_demo = 0.33 + beta * (eddi_max_val - eddi_demo_val)
        weight_lab  = 0.33 + beta * (eddi_max_val - eddi_lab_val)
        weight_text = 0.33 + beta * (eddi_max_val - eddi_text_val)

        # Fuse the modality logits.
        final_logit = demo_logit * weight_demo + lab_logit * weight_lab + text_logit * weight_text

        eddi_details = {
            "eddi_demo": eddi_demo_val,
            "eddi_lab": eddi_lab_val,
            "eddi_text": eddi_text_val,
            "eddi_max": eddi_max_val,
            "weight_demo": weight_demo,
            "weight_lab": weight_lab,
            "weight_text": weight_text,
            "demo_prob": demo_prob,   
            "lab_prob": lab_prob,
            "text_prob": text_prob
        }
        return final_logit, eddi_details

# Training and Evaluation Functions
def train_step(model, dataloader, optimizer, device, beta=1.0, loss_gamma=1.0, target=1.0):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        mortality_logit, los_logit, mechvent_logit, _ = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding, beta=beta
        )
        loss_mort = criterion_mortality(mortality_logit, labels_mortality.unsqueeze(1))
        loss_los = criterion_los(los_logit, labels_los.unsqueeze(1))
        loss_mech = criterion_mech(mechvent_logit, labels_mechvent.unsqueeze(1))
        eddi_loss = ((mortality_logit - target) ** 2).mean()
        loss = loss_mort + loss_los + loss_mech + loss_gamma * eddi_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def validate_step(model, dataloader, device, beta=1.0, loss_gamma=1.0, target=1.0):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]

            mortality_logit, los_logit, mechvent_logit, _ = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=beta
            )
            loss_mort = criterion_mortality(mortality_logit, labels_mortality.unsqueeze(1))
            loss_los = criterion_los(los_logit, labels_los.unsqueeze(1))
            loss_mech = criterion_mech(mechvent_logit, labels_mechvent.unsqueeze(1))
            eddi_loss = ((mortality_logit - target) ** 2).mean()
            loss = loss_mort + loss_los + loss_mech + loss_gamma * eddi_loss
            running_loss += loss.item()
    return running_loss

def evaluate_model_with_confusion(model, dataloader, device, threshold=0.5):
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
            mort_logits, los_logits, mech_logits, _ = model(
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
        recall_val = recall_score(labels, preds, zero_division=0)
        precision_val = precision_score(labels, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall_val, "precision": precision_val,
                         "tpr": tpr, "fpr": fpr}
    
    # Compute fairness EDDI for sensitive subgroups using the provided function.
    ages = torch.cat(all_age, dim=0).numpy().squeeze()
    ethnicities = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    insurances = torch.cat(all_insurance, dim=0).numpy().squeeze()
    
    age_groups = np.array([get_age_bucket(a) for a in ages])
    ethnicity_groups = np.array([map_ethnicity(e) for e in ethnicities])
    insurance_groups = np.array([map_insurance(i) for i in insurances])
    
    eddi_stats = {}
    for task, labels_np, probs in zip(["mortality", "los", "mechanical_ventilation"],
                                      [labels_mort_np, labels_los_np, labels_mech_np],
                                      [mort_probs, los_probs, mech_probs]):
        overall_age, age_eddi_sub = compute_eddi(labels_np.astype(int), probs, age_groups, threshold)
        overall_eth, eth_eddi_sub = compute_eddi(labels_np.astype(int), probs, ethnicity_groups, threshold)
        overall_ins, ins_eddi_sub = compute_eddi(labels_np.astype(int), probs, insurance_groups, threshold)
        total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
        eddi_stats[task] = {"age_eddi": overall_age,
                            "age_subgroup_eddi": age_eddi_sub,
                            "ethnicity_eddi": overall_eth,
                            "ethnicity_subgroup_eddi": eth_eddi_sub,
                            "insurance_eddi": overall_ins,
                            "insurance_subgroup_eddi": ins_eddi_sub,
                            "final_EDDI": total_eddi}
    
    metrics["eddi_stats"] = eddi_stats
    return metrics

def evaluate_model(model, dataloader, device, threshold=0.5):
    metrics = evaluate_model_with_confusion(model, dataloader, device, threshold)
    return metrics

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    structured_data = pd.read_csv("final_structured_common.csv")
    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)

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

    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    # Multi-label stratification based on the three outcomes
    labels = df_filtered[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_val_idx, test_idx in msss.split(df_filtered, labels):
        train_val_df = df_filtered.iloc[train_val_idx]
        test_df = df_filtered.iloc[test_idx]
    
    # Further stratify the train_val set into train and validation (5% for validation)
    labels_train_val = train_val_df[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    for train_idx, val_idx in msss_val.split(train_val_df, labels_train_val):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    # Compute Aggregated Text Embeddings for each split 
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)

    print("\nProcessing training text embeddings...")
    agg_text_train = apply_bioclinicalbert_on_patient_notes(train_df, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")
    print("Processing validation text embeddings...")
    agg_text_val = apply_bioclinicalbert_on_patient_notes(val_df, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")
    print("Processing test text embeddings...")
    agg_text_test = apply_bioclinicalbert_on_patient_notes(test_df, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")

    # Process demographics and lab features for each split 
    demographics_cols = ["age", "GENDER", "ETHNICITY", "INSURANCE"]
    for col in demographics_cols:
        for df in [train_df, val_df, test_df]:
            if col not in df.columns:
                print(f"Column {col} not found in dataframe; creating default values.")
                df[col] = 0
            elif df[col].dtype == object:
                df[col] = df[col].astype("category").cat.codes

    exclude_cols = set(["subject_id", "ROW_ID", "hadm_id", "ICUSTAY_ID",
                        "short_term_mortality", "los_binary", "mechanical_ventilation",
                        "age", "age_bucket", "gender", "ethnicity_category", "insurance_category"])
    lab_feature_columns = [col for col in df_filtered.columns 
                           if col not in exclude_cols and not col.startswith("note_") 
                           and pd.api.types.is_numeric_dtype(df_filtered[col])]
    print("Number of lab feature columns:", len(lab_feature_columns))
    
    for df in [train_df, val_df, test_df]:
        df[lab_feature_columns] = df[lab_feature_columns].fillna(0)

    # Compute normalization parameters from training data
    lab_features_train = train_df[lab_feature_columns].values.astype(np.float32)
    lab_mean = np.mean(lab_features_train, axis=0)
    lab_std = np.std(lab_features_train, axis=0) + 1e-6

    def process_lab_features(df):
        lab_features = df[lab_feature_columns].values.astype(np.float32)
        lab_features = (lab_features - lab_mean) / lab_std
        return lab_features

    lab_train = process_lab_features(train_df)
    lab_val = process_lab_features(val_df)
    lab_test = process_lab_features(test_df)

    # Create tensor datasets for each split
    def create_dataset(df, agg_text_np, lab_features_np):
        num_samples = len(df)
        demo_dummy_ids = torch.zeros((num_samples, 1), dtype=torch.long)
        demo_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
        age_ids = torch.tensor(df["age"].values, dtype=torch.long)
        gender_ids = torch.tensor(df["GENDER"].values, dtype=torch.long)
        ethnicity_ids = torch.tensor(df["ETHNICITY"].values, dtype=torch.long)
        insurance_ids = torch.tensor(df["INSURANCE"].values, dtype=torch.long)
        lab_features_t = torch.tensor(lab_features_np, dtype=torch.float32)
        aggregated_text_embedding = torch.tensor(agg_text_np, dtype=torch.float32)
        labels_mortality = torch.tensor(df["short_term_mortality"].values, dtype=torch.float32)
        labels_los = torch.tensor(df["los_binary"].values, dtype=torch.float32)
        labels_mechvent = torch.tensor(df["mechanical_ventilation"].values, dtype=torch.float32)
        dataset = TensorDataset(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features_t,
            aggregated_text_embedding,
            labels_mortality, labels_los, labels_mechvent
        )
        return dataset

    train_dataset = create_dataset(train_df, agg_text_train, lab_train)
    val_dataset   = create_dataset(val_df, agg_text_val, lab_val)
    test_dataset  = create_dataset(test_df, agg_text_test, lab_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    NUM_AGES = train_df["age"].nunique()
    NUM_GENDERS = train_df["GENDER"].nunique()
    NUM_ETHNICITIES = train_df["ETHNICITY"].nunique()
    NUM_INSURANCES = train_df["INSURANCE"].nunique()
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

    # Compute positive weights using the entire merged dataset (df_filtered)
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)

    # Define separate FocalLoss criteria for each task with gamma set to 1
    global criterion_mortality, criterion_los, criterion_mech
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')

    max_epochs = 20
    patience_limit = 5  
    beta_value = 0.3   
    loss_gamma = 1.0

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training Loop with Early Stopping and detailed evaluation metrics printing per epoch
    for epoch in range(max_epochs):
        train_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                beta=beta_value, loss_gamma=loss_gamma, target=1.0)
        train_loss_epoch = train_loss / len(train_loader)
        
        val_loss = validate_step(multimodal_model, val_loader, device,
                                 beta=beta_value, loss_gamma=loss_gamma, target=1.0)
        val_loss_epoch = val_loss / len(val_loader)
        
        # Evaluate on validation set and get detailed metrics.
        val_metrics = evaluate_model_with_confusion(multimodal_model, val_loader, device, threshold=0.5)
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")
        # Print metrics for each outcome.
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            m = val_metrics[outcome]
            print(f"  {outcome.capitalize()}: AUROC={m['aucroc']:.4f}, AUPRC={m['auprc']:.4f}, F1={m['f1']:.4f}, "
                  f"Recall={m['recall']:.4f}, Precision={m['precision']:.4f}, TPR={m['tpr']:.4f}, FPR={m['fpr']:.4f}")
        # Print fairness EDDI for each outcome.
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            eddi = val_metrics["eddi_stats"][outcome]["final_EDDI"]
            print(f"  {outcome.capitalize()} Final EDDI: {eddi:.4f}")
        
        scheduler.step(val_loss_epoch)
        
        # Early stopping check
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            torch.save(multimodal_model.state_dict(), "best_model.pth")
            print("Validation loss improved. Saving model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} consecutive epochs.")
        if epochs_no_improve >= patience_limit:
            print(f"Early stopping triggered after {patience_limit} epochs with no improvement.")
            break

    print("Training complete.\n")
    
    # Load the best model and evaluate on the test set.
    multimodal_model.load_state_dict(torch.load("best_model.pth"))
    final_metrics = evaluate_model(multimodal_model, test_loader, device, threshold=0.5)
    
    print("\n--- Unique Subgroups (Fixed Order) ---")
    print("Age subgroups      :", ["15-29", "30-49", "50-69", "70-89", "Other"])
    print("Ethnicity subgroups:", ["white", "black", "asian", "hispanic", "other"])
    print("Insurance subgroups:", ["government", "medicare", "Medicaid", "private", "self pay", "other"])

    print("\n--- Final Evaluation Metrics on Test Set ---")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = final_metrics[outcome]
        print(f"\n{outcome.capitalize()}:")
        print("  AUROC     : {:.4f}".format(m["aucroc"]))
        print("  AUPRC     : {:.4f}".format(m["auprc"]))
        print("  F1 Score  : {:.4f}".format(m["f1"]))
        print("  Recall    : {:.4f}".format(m["recall"]))
        print("  Precision : {:.4f}".format(m["precision"]))

    print("\n--- Final Detailed EDDI Statistics ---")
    for task in ["mortality", "los", "mechanical_ventilation"]:
        print(f"\nTask: {task.capitalize()}")
        eddi_stats = final_metrics["eddi_stats"][task]
        print("  Aggregated Age EDDI    : {:.4f}".format(eddi_stats["age_eddi"]))
        print("  Age Subgroup EDDI:")
        for bucket in ["15-29", "30-49", "50-69", "70-89", "Other"]:
            score = eddi_stats["age_subgroup_eddi"].get(bucket, 0)
            print(f"    {bucket}: {score:.4f}")
        print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi_stats["ethnicity_eddi"]))
        print("  Ethnicity Subgroup EDDI:")
        for group in ["white", "black", "asian", "hispanic", "other"]:
            score = eddi_stats["ethnicity_subgroup_eddi"].get(group, 0)
            print(f"    {group}: {score:.4f}")
        print("  Aggregated Insurance EDDI: {:.4f}".format(eddi_stats["insurance_eddi"]))
        print("  Insurance Subgroup EDDI:")
        for group in ["government", "medicare", "Medicaid", "private", "self pay", "other"]:
            score = eddi_stats["insurance_subgroup_eddi"].get(group, 0)
            print(f"    {group}: {score:.4f}")
        print("  Final Overall {} EDDI: {:.4f}".format(task.capitalize(), eddi_stats["final_EDDI"]))
    
    print("\nTesting complete.")

if __name__ == "__main__":
    train_pipeline()
