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


# EDDI Calculation Function
def compute_eddi(true_labels, predicted_labels, sensitive_labels, threshold=0.5):
    """
    Computes the EDDI for a given sensitive attribute.
    It first converts predicted probabilities to binary predictions using the given threshold,
    then computes the overall error rate and for each subgroup computes the subgroup error rate.
    The subgroup disparity is the difference between subgroup error and overall error.
    Finally, the attribute-level EDDI is computed as the square root of the sum of the squared subgroup
    disparities divided by the number of groups.
    
    Returns:
      eddi_attr: the attribute-level EDDI.
      subgroup_eddi: dictionary mapping subgroup -> disparity value.
    """
    # Convert predicted probabilities to binary predictions.
    preds = (predicted_labels > threshold).astype(int)
    overall_error = np.mean(preds != true_labels)
    unique_groups = np.unique(sensitive_labels)
    subgroup_eddi = {}
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            group_error = np.mean(preds[mask] != true_labels[mask])
            subgroup_eddi[group] = group_error - overall_error
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values()))**2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi
    

# BioClinicalBERT Fine-Tuning and Note Aggregation
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
# Demographics Branch
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

# Lab Features Branch
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


# Multimodal Transformer (Fusion of Branches)
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device):
        super(MultimodalTransformer, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device

        # We use the same projector for all outcomes in this example.
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

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=1.0):
        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        # Outcome 1: Mortality
        demo_proj_mort = self.demo_projector(demo_embedding)
        lab_proj_mort = self.lab_projector(lab_embedding)
        text_proj_mort = self.text_projector(text_embedding)
        demo_EDDI_mort = torch.sum(demo_proj_mort, dim=1, keepdim=True)
        lab_EDDI_mort  = torch.sum(lab_proj_mort, dim=1, keepdim=True)
        text_EDDI_mort = torch.sum(text_proj_mort, dim=1, keepdim=True)
        modality_EDDI_mort = torch.cat([demo_EDDI_mort, lab_EDDI_mort, text_EDDI_mort], dim=1)
        EDDI_max_mort, _ = torch.max(modality_EDDI_mort, dim=1, keepdim=True)
        weight_demo_mort = 0.33 + beta * (EDDI_max_mort - demo_EDDI_mort)
        weight_lab_mort  = 0.33 + beta * (EDDI_max_mort - lab_EDDI_mort)
        weight_text_mort = 0.33 + beta * (EDDI_max_mort - text_EDDI_mort)
        mortality_logit = demo_EDDI_mort * weight_demo_mort + lab_EDDI_mort * weight_lab_mort + text_EDDI_mort * weight_text_mort

        # Outcome 2: LOS
        demo_proj_los = self.demo_projector(demo_embedding)
        lab_proj_los = self.lab_projector(lab_embedding)
        text_proj_los = self.text_projector(text_embedding)
        demo_EDDI_los = torch.sum(demo_proj_los, dim=1, keepdim=True)
        lab_EDDI_los  = torch.sum(lab_proj_los, dim=1, keepdim=True)
        text_EDDI_los = torch.sum(text_proj_los, dim=1, keepdim=True)
        modality_EDDI_los = torch.cat([demo_EDDI_los, lab_EDDI_los, text_EDDI_los], dim=1)
        EDDI_max_los, _ = torch.max(modality_EDDI_los, dim=1, keepdim=True)
        weight_demo_los = 0.33 + beta * (EDDI_max_los - demo_EDDI_los)
        weight_lab_los  = 0.33 + beta * (EDDI_max_los - lab_EDDI_los)
        weight_text_los = 0.33 + beta * (EDDI_max_los - text_EDDI_los)
        los_logit = demo_EDDI_los * weight_demo_los + lab_EDDI_los * weight_lab_los + text_EDDI_los * weight_text_los

        # Outcome 3: Mechanical Ventilation
        demo_proj_mv = self.demo_projector(demo_embedding)
        lab_proj_mv = self.lab_projector(lab_embedding)
        text_proj_mv = self.text_projector(text_embedding)
        demo_EDDI_mv = torch.sum(demo_proj_mv, dim=1, keepdim=True)
        lab_EDDI_mv  = torch.sum(lab_proj_mv, dim=1, keepdim=True)
        text_EDDI_mv = torch.sum(text_proj_mv, dim=1, keepdim=True)
        modality_EDDI_mv = torch.cat([demo_EDDI_mv, lab_EDDI_mv, text_EDDI_mv], dim=1)
        EDDI_max_mv, _ = torch.max(modality_EDDI_mv, dim=1, keepdim=True)
        weight_demo_mv = 0.33 + beta * (EDDI_max_mv - demo_EDDI_mv)
        weight_lab_mv  = 0.33 + beta * (EDDI_max_mv - lab_EDDI_mv)
        weight_text_mv = 0.33 + beta * (EDDI_max_mv - text_EDDI_mv)
        mechvent_logit = demo_EDDI_mv * weight_demo_mv + lab_EDDI_mv * weight_lab_mv + text_EDDI_mv * weight_text_mv

        return mortality_logit, los_logit, mechvent_logit, (demo_EDDI_mort, lab_EDDI_mort, text_EDDI_mort, EDDI_max_mort)


# Training and Evaluation Functions
def train_step(model, dataloader, optimizer, device, criterion, beta=1.0, loss_gamma=1.0, target=1.0):
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
        loss_mort = criterion(mortality_logit, labels_mortality.unsqueeze(1))
        loss_los = criterion(los_logit, labels_los.unsqueeze(1))
        loss_mech = criterion(mechvent_logit, labels_mechvent.unsqueeze(1))
        eddi_loss = ((mortality_logit - target) ** 2).mean()
        loss = loss_mort + loss_los + loss_mech + loss_gamma * eddi_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

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
        recall = recall_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall, "precision": precision}
    
    # Get sensitive attribute groups.
    ages = torch.cat(all_age, dim=0).numpy().squeeze()
    ethnicities = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    insurances = torch.cat(all_insurance, dim=0).numpy().squeeze()
    
    age_groups = np.array([get_age_bucket(a) for a in ages])
    ethnicity_groups = np.array([map_ethnicity(e) for e in ethnicities])
    insurance_groups = np.array([map_insurance(i) for i in insurances])
    
    # Fixed subgroup orders.
    age_order = ["15-29", "30-49", "50-69", "70-89", "Other"]
    ethnicity_order = ["white", "black", "asian", "hispanic", "other"]
    insurance_order = ["government", "medicare", "Medicaid", "private", "self pay", "other"]
    
    # Compute EDDI for each outcome.
    eddi_stats = {}
    for task, labels_np, probs in zip(["mortality", "los", "mechanical_ventilation"],
                                      [labels_mort_np, labels_los_np, labels_mech_np],
                                      [mort_probs, los_probs, mech_probs]):
        overall_age, age_eddi_sub = compute_eddi(labels_np.astype(int), probs, age_groups, threshold)
        overall_eth, eth_eddi_sub = compute_eddi(labels_np.astype(int), probs, ethnicity_groups, threshold)
        overall_ins, ins_eddi_sub = compute_eddi(labels_np.astype(int), probs, insurance_groups, threshold)
        total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
        eddi_stats[task] = {
            "age_eddi": overall_age,
            "age_subgroup_eddi": age_eddi_sub,
            "ethnicity_eddi": overall_eth,
            "ethnicity_subgroup_eddi": eth_eddi_sub,
            "insurance_eddi": overall_ins,
            "insurance_subgroup_eddi": ins_eddi_sub,
            "final_EDDI": total_eddi
        }
    
    metrics["eddi_stats"] = eddi_stats
    
    unique_subgroups = {
        "age": list(np.unique(age_groups)),
        "ethnicity": list(np.unique(ethnicity_groups)),
        "insurance": list(np.unique(insurance_groups))
    }
    
    return metrics, all_mort_logits, unique_subgroups


def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load merged datasets (Structured and Unstructured)
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

    # Ensure outcomes are integers.
    merged_df["short_term_mortality"] = merged_df["short_term_mortality"].astype(int)
    merged_df["los_binary"] = merged_df["los_binary"].astype(int)
    merged_df["mechanical_ventilation"] = merged_df["mechanical_ventilation"].astype(int)

    # Filter rows with at least one valid note.
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    # Compute aggregated text embeddings.
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    # Process demographics and lab features.
    demographics_cols = ["age", "GENDER", "ETHNICITY", "INSURANCE"]
    for col in demographics_cols:
        if col not in df_filtered.columns:
            print(f"Column {col} not found; creating default values.")
            df_filtered[col] = 0
        elif df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].astype("category").cat.codes

    exclude_cols = set(["subject_id", "ROW_ID", "hadm_id", "ICUSTAY_ID",
                        "short_term_mortality", "los_binary", "mechanical_ventilation",
                        "age", "age_bucket", "gender", "ethnicity_category", "insurance_category"])
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

    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    criterion = FocalLoss(gamma=2, pos_weight=mortality_pos_weight, reduction='mean')

    max_epochs = 20
    patience_limit = 5  
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    loss_gamma = 1

    for beta_value in beta_values:
        print(f"\nTraining with beta = {beta_value}")
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            running_loss = train_step(multimodal_model, dataloader, optimizer, device,
                                      criterion, beta=beta_value, loss_gamma=loss_gamma, target=1.0)
            epoch_loss = running_loss / len(dataloader)
            print(f"[Beta {beta_value} | Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
            scheduler.step(epoch_loss)
            metrics, _, _ = evaluate_model(multimodal_model, dataloader, device, threshold=0.5)
            # Print evaluation metrics for each outcome.
            print(f"[Epoch {epoch+1}] Mortality - AUROC: {metrics['mortality']['aucroc']:.4f}, AUPRC: {metrics['mortality']['auprc']:.4f}, F1: {metrics['mortality']['f1']:.4f}, Recall: {metrics['mortality']['recall']:.4f}, Precision: {metrics['mortality']['precision']:.4f}")
            print(f"[Epoch {epoch+1}] LOS - AUROC: {metrics['los']['aucroc']:.4f}, AUPRC: {metrics['los']['auprc']:.4f}, F1: {metrics['los']['f1']:.4f}, Recall: {metrics['los']['recall']:.4f}, Precision: {metrics['los']['precision']:.4f}")
            print(f"[Epoch {epoch+1}] Mechanical Ventilation - AUROC: {metrics['mechanical_ventilation']['aucroc']:.4f}, AUPRC: {metrics['mechanical_ventilation']['auprc']:.4f}, F1: {metrics['mechanical_ventilation']['f1']:.4f}, Recall: {metrics['mechanical_ventilation']['recall']:.4f}, Precision: {metrics['mechanical_ventilation']['precision']:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience_limit:
                print(f"No improvement for {patience_limit} consecutive epochs. Early stopping for beta = {beta_value}.")
                break

    print("Training complete.\n")
    
    final_metrics, mort_logits, unique_subgroups = evaluate_model(multimodal_model, dataloader, device, threshold=0.5, print_eddi=True)
    
    print("\n--- Unique Subgroups (Fixed Order) ---")
    print("Age subgroups      :", ["15-29", "30-49", "50-69", "70-89", "Other"])
    print("Ethnicity subgroups:", ["white", "black", "asian", "hispanic", "other"])
    print("Insurance subgroups:", ["government", "medicare", "Medicaid", "private", "self pay", "other"])

    print("\n--- Final Evaluation Metrics ---")
    print("Mortality:")
    print("  AUROC     : {:.4f}".format(final_metrics["mortality"]["aucroc"]))
    print("  AUPRC     : {:.4f}".format(final_metrics["mortality"]["auprc"]))
    print("  F1 Score  : {:.4f}".format(final_metrics["mortality"]["f1"]))
    print("  Recall    : {:.4f}".format(final_metrics["mortality"]["recall"]))
    print("  Precision : {:.4f}".format(final_metrics["mortality"]["precision"]))
    print("LOS:")
    print("  AUROC     : {:.4f}".format(final_metrics["los"]["aucroc"]))
    print("  AUPRC     : {:.4f}".format(final_metrics["los"]["auprc"]))
    print("  F1 Score  : {:.4f}".format(final_metrics["los"]["f1"]))
    print("  Recall    : {:.4f}".format(final_metrics["los"]["recall"]))
    print("  Precision : {:.4f}".format(final_metrics["los"]["precision"]))
    print("Mechanical Ventilation:")
    print("  AUROC     : {:.4f}".format(final_metrics["mechanical_ventilation"]["aucroc"]))
    print("  AUPRC     : {:.4f}".format(final_metrics["mechanical_ventilation"]["auprc"]))
    print("  F1 Score  : {:.4f}".format(final_metrics["mechanical_ventilation"]["f1"]))
    print("  Recall    : {:.4f}".format(final_metrics["mechanical_ventilation"]["recall"]))
    print("  Precision : {:.4f}".format(final_metrics["mechanical_ventilation"]["precision"]))

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
    
    print("\nTraining complete.")

if __name__ == "__main__":
    train_pipeline()
