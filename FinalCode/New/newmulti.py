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
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from scipy.special import expit  # logistic sigmoid
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, ttest_ind

########################################
# 1. Loss and Class Weight Functions
########################################

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

def get_pos_weight(labels_series, device):
    positive = labels_series.sum()
    negative = len(labels_series) - positive
    if positive == 0:
        weight = torch.tensor(1.0, dtype=torch.float, device=device)
    else:
        weight = torch.tensor(negative / positive, dtype=torch.float, device=device)
    return weight

########################################
# 2. Helper Functions for Bucketing and Mapping
########################################

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

def map_ethnicity(e):
    mapping = {0: "white", 1: "black", 2: "hispanic", 3: "asian"}
    return mapping.get(e, "others")

def map_insurance(i):
    mapping = {0: "government", 1: "medicare", 2: "Medicaid", 3: "private", 4: "self pay"}
    return mapping.get(i, "others")

########################################
# 3. EDDI Computation Functions
########################################

def compute_eddi(y_true, y_pred, sensitive_labels):
    """
    Computes the Error Distribution Disparity Index (EDDI) for one outcome.
    For each subgroup s, compute the subgroup error rate (ER_s) and overall error rate (OER).
    Then, for each subgroup, disparity = (ER_s - OER) / max(OER, 1 - OER).
    Finally, attribute-level EDDI is computed as the root mean square of subgroup disparities.
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
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

########################################
# 4. BioClinicalBERT Fine-Tuning Wrapper & Text Embedding Aggregation
########################################

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
    return aggregated_embeddings, patient_ids

########################################
# 5. Structured Data: BEHRT Model Definition
########################################

class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs, 
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()
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
        self.age_embedding = nn.Embedding(num_ages, hidden_size)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.admission_loc_embedding = nn.Embedding(num_admission_locs, hidden_size)
        self.discharge_loc_embedding = nn.Embedding(num_discharge_locs, hidden_size)
        self.gender_embedding = nn.Embedding(num_genders, hidden_size)
        self.ethnicity_embedding = nn.Embedding(num_ethnicities, hidden_size)
        self.insurance_embedding = nn.Embedding(num_insurances, hidden_size)

    def forward(self, input_ids, attention_mask, age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids):
        age_ids = torch.clamp(age_ids, 0, self.age_embedding.num_embeddings - 1)
        segment_ids = torch.clamp(segment_ids, 0, self.segment_embedding.num_embeddings - 1)
        adm_loc_ids = torch.clamp(adm_loc_ids, 0, self.admission_loc_embedding.num_embeddings - 1)
        disch_loc_ids = torch.clamp(disch_loc_ids, 0, self.discharge_loc_embedding.num_embeddings - 1)
        gender_ids = torch.clamp(gender_ids, 0, self.gender_embedding.num_embeddings - 1)
        ethnicity_ids = torch.clamp(ethnicity_ids, 0, self.ethnicity_embedding.num_embeddings - 1)
        insurance_ids = torch.clamp(insurance_ids, 0, self.insurance_embedding.num_embeddings - 1)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        age_embeds = self.age_embedding(age_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        adm_embeds = self.admission_loc_embedding(adm_loc_ids)
        disch_embeds = self.discharge_loc_embedding(disch_loc_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        eth_embeds = self.ethnicity_embedding(ethnicity_ids)
        ins_embeds = self.insurance_embedding(insurance_ids)
        extra = (age_embeds + segment_embeds + adm_embeds + disch_embeds +
                 gender_embeds + eth_embeds + ins_embeds) / 7.0
        cls_embedding = cls_token + extra
        return cls_embedding

########################################
# 6. Multimodal Transformer Model Definition
########################################

class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, BEHRT, device, hidden_size=512):
        super(MultimodalTransformer, self).__init__()
        self.BEHRT = BEHRT
        self.device = device
        self.ts_projector = nn.Sequential(
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
            nn.Linear(hidden_size, 2)  # Outputs: [mortality_logit, los_logit]
        )

    def forward(self, dummy_input_ids, dummy_attn_mask, age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids, aggregated_text_embedding):
        structured_emb = self.BEHRT(dummy_input_ids, dummy_attn_mask,
                                    age_ids, segment_ids, adm_loc_ids, disch_loc_ids,
                                    gender_ids, ethnicity_ids, insurance_ids)
        ts_proj = self.ts_projector(structured_emb)
        text_proj = self.text_projector(aggregated_text_embedding)
        combined = torch.cat((ts_proj, text_proj), dim=1)
        logits = self.classifier(combined)
        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits = logits[:, 1].unsqueeze(1)
        return mortality_logits, los_logits

########################################
# 7. Training and Evaluation Functions for Multimodal Model
########################################

def train_step(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (dummy_input_ids, dummy_attn_mask,
         age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
         gender_ids, ethnicity_ids, insurance_ids,
         aggregated_text_embedding,
         labels_mortality, labels_los) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        mortality_logits, los_logits = model(
            dummy_input_ids, dummy_attn_mask,
            age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
            gender_ids, ethnicity_ids, insurance_ids,
            aggregated_text_embedding
        )
        loss_mort = criterion(mortality_logits, labels_mortality.unsqueeze(1))
        loss_los = criterion(los_logits, labels_los.unsqueeze(1))
        loss = loss_mort + loss_los
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, device, threshold=0.5, print_eddi=False):
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_age = []
    all_ethnicity = []
    all_insurance = []
    with torch.no_grad():
        for batch in dataloader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los) = [x.to(device) for x in batch]
            mort_logits, los_logits = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    all_mort_logits = torch.cat(all_mort_logits, dim=0).numpy().squeeze()
    all_los_logits = torch.cat(all_los_logits, dim=0).numpy().squeeze()
    all_labels_mort = torch.cat(all_labels_mort, dim=0).numpy().squeeze()
    all_labels_los = torch.cat(all_labels_los, dim=0).numpy().squeeze()
    ages = torch.cat(all_age, dim=0).numpy().squeeze()
    ethnicities = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    insurances = torch.cat(all_insurance, dim=0).numpy().squeeze()

    mort_probs = expit(all_mort_logits)
    los_probs = expit(all_los_logits)
    
    metrics = {}
    for task, probs, labels in zip(["mortality", "los"],
                                   [mort_probs, los_probs],
                                   [all_labels_mort, all_labels_los]):
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
    
    # Compute EDDI for fairness evaluation.
    y_pred_mort = (mort_probs > threshold).astype(int)
    y_true_mort = all_labels_mort.astype(int)
    y_pred_los = (los_probs > threshold).astype(int)
    y_true_los = all_labels_los.astype(int)
    
    age_groups = np.array([get_age_bucket(a) for a in ages])
    ethnicity_groups = np.array([map_ethnicity(e) for e in ethnicities])
    insurance_groups = np.array([map_insurance(i) for i in insurances])
    
    # Define fixed subgroup orders.
    age_order = ["15-29", "30-49", "50-69", "70-89", "others"]
    ethnicity_order = ["white", "black", "hispanic", "asian", "others"]
    insurance_order = ["government", "medicare", "Medicaid", "private", "self pay", "others"]
    
    # Mortality outcome EDDI.
    eddi_age_mort, age_eddi_sub_mort = compute_eddi(y_true_mort, y_pred_mort, age_groups)
    eddi_eth_mort, eth_eddi_sub_mort = compute_eddi(y_true_mort, y_pred_mort, ethnicity_groups)
    eddi_ins_mort, ins_eddi_sub_mort = compute_eddi(y_true_mort, y_pred_mort, insurance_groups)
    age_scores = [age_eddi_sub_mort.get(bucket, 0) for bucket in age_order]
    eth_scores = [eth_eddi_sub_mort.get(group, 0) for group in ethnicity_order]
    ins_scores = [ins_eddi_sub_mort.get(group, 0) for group in insurance_order]
    overall_age_eddi_mort = np.sqrt(np.sum(np.square(age_scores))) / len(age_order)
    overall_eth_eddi_mort = np.sqrt(np.sum(np.square(eth_scores))) / len(ethnicity_order)
    overall_ins_eddi_mort = np.sqrt(np.sum(np.square(ins_scores))) / len(insurance_order)
    total_eddi_mort = np.sqrt((overall_age_eddi_mort**2 + overall_eth_eddi_mort**2 + overall_ins_eddi_mort**2)) / 3

    # LOS outcome EDDI.
    eddi_age_los, age_eddi_sub_los = compute_eddi(y_true_los, y_pred_los, age_groups)
    eddi_eth_los, eth_eddi_sub_los = compute_eddi(y_true_los, y_pred_los, ethnicity_groups)
    eddi_ins_los, ins_eddi_sub_los = compute_eddi(y_true_los, y_pred_los, insurance_groups)
    age_scores_los = [age_eddi_sub_los.get(bucket, 0) for bucket in age_order]
    eth_scores_los = [eth_eddi_sub_los.get(group, 0) for group in ethnicity_order]
    ins_scores_los = [ins_eddi_sub_los.get(group, 0) for group in insurance_order]
    overall_age_eddi_los = np.sqrt(np.sum(np.square(age_scores_los))) / len(age_order)
    overall_eth_eddi_los = np.sqrt(np.sum(np.square(eth_scores_los))) / len(ethnicity_order)
    overall_ins_eddi_los = np.sqrt(np.sum(np.square(ins_scores_los))) / len(insurance_order)
    total_eddi_los = np.sqrt((overall_age_eddi_los**2 + overall_eth_eddi_los**2 + overall_ins_eddi_los**2)) / 3

    metrics["eddi_stats"] = {
        "mortality": {
            "age_subgroup_eddi": age_eddi_sub_mort,
            "age_eddi": overall_age_eddi_mort,
            "ethnicity_subgroup_eddi": eth_eddi_sub_mort,
            "ethnicity_eddi": overall_eth_eddi_mort,
            "insurance_subgroup_eddi": ins_eddi_sub_mort,
            "insurance_eddi": overall_ins_eddi_mort,
            "final_EDDI": total_eddi_mort
        },
        "los": {
            "age_subgroup_eddi": age_eddi_sub_los,
            "age_eddi": overall_age_eddi_los,
            "ethnicity_subgroup_eddi": eth_eddi_sub_los,
            "ethnicity_eddi": overall_eth_eddi_los,
            "insurance_subgroup_eddi": ins_eddi_sub_los,
            "insurance_eddi": overall_ins_eddi_los,
            "final_EDDI": total_eddi_los
        }
    }
    
    if print_eddi:
        print("\n--- EDDI Calculation for Mortality Outcome ---")
        for bucket in age_order:
            print(f"Age bucket {bucket}: {age_eddi_sub_mort.get(bucket, np.nan):.4f}")
        for group in ethnicity_order:
            print(f"Ethnicity group {group}: {eth_eddi_sub_mort.get(group, np.nan):.4f}")
        for group in insurance_order:
            print(f"Insurance group {group}: {ins_eddi_sub_mort.get(group, np.nan):.4f}")
        print("Final Overall Mortality EDDI: {:.4f}".format(total_eddi_mort))
        
        print("\n--- EDDI Calculation for LOS Outcome ---")
        for bucket in age_order:
            print(f"Age bucket {bucket}: {age_eddi_sub_los.get(bucket, np.nan):.4f}")
        for group in ethnicity_order:
            print(f"Ethnicity group {group}: {eth_eddi_sub_los.get(group, np.nan):.4f}")
        for group in insurance_order:
            print(f"Insurance group {group}: {ins_eddi_sub_los.get(group, np.nan):.4f}")
        print("Final Overall LOS EDDI: {:.4f}".format(total_eddi_los))
    
    metrics["eddi_stats"]["overall"] = {"mortality": total_eddi_mort, "los": total_eddi_los}
    return metrics

########################################
# 8. Main Training and Evaluation Pipeline
########################################

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Read structured data.
    structured_data = pd.read_csv('filtered_structured_output.csv')
    keep_cols = {"subject_id", "hadm_id", "short_term_mortality", "readmission_within_30_days", "age",
                 "FIRST_WARDID", "LAST_WARDID", "ETHNICITY", "INSURANCE", "GENDER"}
    new_columns = {col: f"{col}_struct" for col in structured_data.columns if col not in keep_cols}
    structured_data.rename(columns=new_columns, inplace=True)

    # Read unstructured data.
    unstructured_data = pd.read_csv("filtered_unstructured.csv", low_memory=False)
    unstructured_data.drop(
        columns=["short_term_mortality", "readmission_within_30_days", "age", "segment", 
                 "admission_loc", "discharge_loc", "gender", "ethnicity", "insurance"],
        errors='ignore',
        inplace=True
    )
    merged_df = pd.merge(structured_data, unstructured_data, on=["subject_id", "hadm_id"], how="inner")
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check your data and merge keys.")
    
    # Update outcomes: use 'short_term_mortality' and 'los_gt_3' for mortality and LOS respectively.
    merged_df["short_term_mortality"] = merged_df["short_term_mortality"].astype(int)
    merged_df["los_gt_3"] = merged_df["los_gt_3"].astype(int)
    
    # Identify note columns.
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))
    
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
    if "age" in df_filtered.columns and df_filtered["age"].dtype == object:
        df_filtered["age"] = df_filtered["age"].astype("float")
    if "segment" not in df_filtered.columns:
        df_filtered["segment"] = 0
    elif df_filtered["segment"].dtype == object:
        df_filtered["segment"] = df_filtered["segment"].astype("category").cat.codes
    
    num_samples = len(df_filtered)
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    
    age_ids = torch.tensor(df_filtered["age"].values, dtype=torch.long)
    segment_ids = torch.tensor(df_filtered["segment"].values, dtype=torch.long)
    admission_loc_ids = torch.tensor(df_filtered["FIRST_WARDID"].values, dtype=torch.long)
    discharge_loc_ids = torch.tensor(df_filtered["LAST_WARDID"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_filtered["GENDER"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_filtered["ETHNICITY"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_filtered["INSURANCE"].values, dtype=torch.long)
    
    labels_mortality = torch.tensor(df_filtered["short_term_mortality"].values, dtype=torch.float32)
    labels_los = torch.tensor(df_filtered["los_gt_3"].values, dtype=torch.float32)
    
    class_weights_mortality = compute_class_weights(df_filtered, 'short_term_mortality')
    class_weights_los = compute_class_weights(df_filtered, 'los_gt_3')
    
    dataset = TensorDataset(
        dummy_input_ids, dummy_attn_mask,
        age_ids, segment_ids, admission_loc_ids, discharge_loc_ids,
        gender_ids, ethnicity_ids, insurance_ids,
        aggregated_text_embeddings_t,
        labels_mortality, labels_los
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Hyperparameters for structured data.
    disease_mapping = {d: i for i, d in enumerate(df_filtered["hadm_id"].unique())}
    NUM_DISEASES = len(disease_mapping)
    NUM_AGES = int(df_filtered["age"].nunique())
    NUM_SEGMENTS = 2
    NUM_ADMISSION_LOCS = int(df_filtered["FIRST_WARDID"].nunique())
    NUM_DISCHARGE_LOCS = int(df_filtered["LAST_WARDID"].nunique())
    NUM_GENDERS = int(df_filtered["GENDER"].nunique())
    NUM_ETHNICITIES = int(df_filtered["ETHNICITY"].nunique())
    NUM_INSURANCES = int(df_filtered["INSURANCE"].nunique())
    
    print("\n--- Hyperparameters based on processed data ---")
    print("NUM_DISEASES:", NUM_DISEASES)
    print("NUM_AGES:", NUM_AGES)
    print("NUM_SEGMENTS:", NUM_SEGMENTS)
    print("NUM_ADMISSION_LOCS:", NUM_ADMISSION_LOCS)
    print("NUM_DISCHARGE_LOCS:", NUM_DISCHARGE_LOCS)
    print("NUM_GENDERS:", NUM_GENDERS)
    print("NUM_ETHNICITIES:", NUM_ETHNICITIES)
    print("NUM_INSURANCES:", NUM_INSURANCES)
    
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
        text_embed_size=768,
        BEHRT=behrt_model,
        device=device,
        hidden_size=512
    ).to(device)
    
    optimizer = AdamW(multimodal_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    criterion = FocalLoss(gamma=2, pos_weight=mortality_pos_weight, reduction='mean')
    
    num_epochs = 20
    for epoch in range(num_epochs):
        loss = train_step(multimodal_model, dataloader, optimizer, device, criterion)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {loss:.4f}")
        scheduler.step(loss)
    
    metrics = evaluate_model(multimodal_model, dataloader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation Metrics:")
    for outcome in ["mortality", "los"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")
    
    print("\nFinal Detailed EDDI Statistics:")
    eddi_stats = metrics["eddi_stats"]
    print("\nMortality EDDI Stats:")
    print("  Age subgroup EDDI:", eddi_stats["mortality"]["age_subgroup_eddi"])
    print("  Aggregated Age EDDI: {:.4f}".format(eddi_stats["mortality"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI:", eddi_stats["mortality"]["ethnicity_subgroup_eddi"])
    print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi_stats["mortality"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI:", eddi_stats["mortality"]["insurance_subgroup_eddi"])
    print("  Aggregated Insurance EDDI: {:.4f}".format(eddi_stats["mortality"]["insurance_eddi"]))
    print("  Final Overall Mortality EDDI: {:.4f}".format(eddi_stats["mortality"]["final_EDDI"]))
    
    print("\nLOS EDDI Stats:")
    print("  Age subgroup EDDI:", eddi_stats["los"]["age_subgroup_eddi"])
    print("  Aggregated Age EDDI: {:.4f}".format(eddi_stats["los"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI:", eddi_stats["los"]["ethnicity_subgroup_eddi"])
    print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi_stats["los"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI:", eddi_stats["los"]["insurance_subgroup_eddi"])
    print("  Aggregated Insurance EDDI: {:.4f}".format(eddi_stats["los"]["insurance_eddi"]))
    print("  Final Overall LOS EDDI: {:.4f}".format(eddi_stats["los"]["final_EDDI"]))
    
    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
