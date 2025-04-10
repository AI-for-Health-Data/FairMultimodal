import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, Subset
from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Iterative stratification for multi-label splitting
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

DEBUG = True

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
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

def compute_frequency_dict(tensor):
    """Computes a dictionary mapping unique values to their counts."""
    arr = tensor.cpu().numpy()
    unique, counts = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}

def compute_sample_weights(sensitive, freq_dict):
    """For each sample, weight = 1 / frequency of its group."""
    weights = [1.0 / freq_dict[int(s.item())] for s in sensitive]
    return torch.tensor(weights, dtype=torch.float32, device=sensitive.device)

def weighted_reconstruction_loss(x, x_recon, sample_weights):
    """Computes the weighted mean squared error loss."""
    mse_per_sample = ((x - x_recon) ** 2).mean(dim=1)
    loss = (sample_weights * mse_per_sample).mean()
    return loss

def compute_eddi(y_true, y_pred, sensitive_labels, threshold=0.5):
    y_pred_binary = (np.array(y_pred) > threshold).astype(int)
    unique_groups = np.unique(sensitive_labels)
    subgroup_eddi = {}
    overall_error = np.mean(y_pred_binary != y_true)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = 0.0
        else:
            er_group = np.mean(y_pred_binary[mask] != y_true[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

# Updated sensitive attribute mapping functions
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
    mapping = {0: "white", 1: "black", 2: "hispanic", 3: "asian"}
    return mapping.get(code, "other")

def map_insurance(code):
    mapping = {0: "government", 1: "medicare", 2: "medicaid", 3: "self pay", 4: "private"}
    return mapping.get(code, "other")

def calculate_predictive_parity(y_true, y_pred, sensitive_attrs):
    """
    Calculate the predictive parity (precision equality) for each group defined by a sensitive attribute.
    """
    unique_groups = np.unique(sensitive_attrs)
    precision_scores = {}
    for group in unique_groups:
        group_indices = sensitive_attrs == group
        precision = precision_score(y_true[group_indices], y_pred[group_indices], zero_division=0, average='weighted')
        precision_scores[group] = precision
    return precision_scores

def calculate_tpr_and_fpr(y_true, y_pred, group_mask):
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for a given group.
    """
    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask], labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

def calculate_subgroup_tpr_fpr(y_true, y_pred, sensitive_array):
    """
    For a given sensitive attribute array, calculate TPR and FPR for each subgroup.
    Returns a dictionary mapping subgroup to its {TPR, FPR} and overall averages.
    """
    unique_groups = np.unique(sensitive_array)
    results = {}
    tpr_list = []
    fpr_list = []
    for group in unique_groups:
        mask = (sensitive_array == group)
        if np.sum(mask) == 0:
            continue
        tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred, mask)
        results[group] = {'TPR': tpr, 'FPR': fpr}
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    avg_tpr = np.mean(tpr_list) if tpr_list else 0
    avg_fpr = np.mean(fpr_list) if fpr_list else 0
    return results, avg_tpr, avg_fpr

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

# BEHRT Model for Structured Data
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

# Stacked Denoising Autoencoder (FPM Baseline)
class StackedDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, noise_factor=0.05):
        super(StackedDenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor        
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for h in hidden_dims[::-1]:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        noise = self.noise_factor * torch.randn_like(x)
        x_noisy = x + noise
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        return x_recon, z

# FPM Pretraining Routine
def pretrain_fpm_autoencoder(model, data_loader, optimizer, device, freq_dict):
    model.train()
    running_loss = 0.0
    for batch in data_loader:
        x_batch, sensitive_batch = [b.to(device) for b in batch]
        optimizer.zero_grad()
        x_recon, _ = model(x_batch)
        sample_weights = compute_sample_weights(sensitive_batch, freq_dict)
        loss = weighted_reconstruction_loss(x_batch, x_recon, sample_weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)

# Downstream Evaluation Functions
def train_downstream_classifier(representations, labels):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(representations, labels)
    return clf

def evaluate_downstream(clf, representations, labels):
    y_prob = clf.predict_proba(representations)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    aucroc = roc_auc_score(labels, y_prob)
    auprc = average_precision_score(labels, y_prob)
    f1 = f1_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    precision = precision_score(labels, y_pred, zero_division=0)
    return {"aucroc": aucroc, "auprc": auprc, "f1": f1, "recall": recall, "precision": precision}, y_prob, y_pred

def compute_demographic_parity(y_pred, sensitive):
    groups = np.unique(sensitive)
    rates = {}
    for g in groups:
        idx = (sensitive == g)
        rates[g] = np.mean(y_pred[idx])
    return rates

def evaluate_model_loss(model, dataloader, device, crit_mort, crit_los, crit_vent):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
            mort_logits, los_logits, mech_logits = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            loss_mort = crit_mort(mort_logits, labels_mortality.unsqueeze(1))
            loss_los = crit_los(los_logits, labels_los.unsqueeze(1))
            loss_vent = crit_vent(mech_logits, labels_vent.unsqueeze(1))
            loss = loss_mort + loss_los + loss_vent
            running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model_metrics(model, dataloader, device, threshold=0.5, print_eddi=False):
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
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
            mort_logits, los_logits, mech_logits = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_mech_logits.append(mech_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_mech.append(labels_vent.cpu())
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
        TP = ((preds == 1) & (labels == 1)).sum()
        FP = ((preds == 1) & (labels == 0)).sum()
        TN = ((preds == 0) & (labels == 0)).sum()
        FN = ((preds == 0) & (labels == 1)).sum()
        TPR = TP / (TP + FN + 1e-8)
        FPR = FP / (FP + TN + 1e-8)
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall, "precision": precision,
                         "tpr": TPR, "fpr": FPR}
    
    if print_eddi:
        # Standard subgroup orders
        age_order_list = ["15-29", "30-49", "50-69", "70-89", "Other"]
        ethnicity_order_list = ["white", "black", "asian", "hispanic", "other"]
        insurance_order_list = ["government", "medicare", "medicaid", "self pay", "private", "other"]
        
        eddi_stats = {}
        for task, labels_np, probs in zip(["mortality", "los", "mechanical_ventilation"],
                                          [labels_mort_np, labels_los_np, labels_mech_np],
                                          [mort_probs, los_probs, mech_probs]):
            overall_age, age_eddi_sub = compute_eddi(labels_np.astype(int), probs, 
                                                     np.array([get_age_bucket(a) for a in torch.cat(all_age, dim=0).numpy().squeeze()]), 
                                                     threshold)
            overall_eth, eth_eddi_sub = compute_eddi(labels_np.astype(int), probs, 
                                                     np.array([map_ethnicity(e) for e in torch.cat(all_ethnicity, dim=0).numpy().squeeze()]), 
                                                     threshold)
            overall_ins, ins_eddi_sub = compute_eddi(labels_np.astype(int), probs, 
                                                     np.array([map_insurance(i) for i in torch.cat(all_insurance, dim=0).numpy().squeeze()]), 
                                                     threshold)
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
        
        print("\n--- EDDI Calculation for Each Outcome ---")
        for task in ["mortality", "los", "mechanical_ventilation"]:
            print(f"\nTask: {task.capitalize()}")
            eddi = eddi_stats[task]
            print("  Aggregated Age EDDI    : {:.4f}".format(eddi["age_eddi"]))
            print("  Age Subgroup EDDI:")
            for bucket in age_order_list:
                score = eddi["age_subgroup_eddi"].get(bucket, 0)
                print(f"    {bucket}: {score:.4f}")
            print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi["ethnicity_eddi"]))
            print("  Ethnicity Subgroup EDDI:")
            for group in ethnicity_order_list:
                score = eddi["ethnicity_subgroup_eddi"].get(group, 0)
                print(f"    {group}: {score:.4f}")
            print("  Aggregated Insurance EDDI: {:.4f}".format(eddi["insurance_eddi"]))
            print("  Insurance Subgroup EDDI:")
            for group in insurance_order_list:
                score = eddi["insurance_subgroup_eddi"].get(group, 0)
                print(f"    {group}: {score:.4f}")
            print("  Final Overall {} EDDI: {:.4f}".format(task.capitalize(), eddi["final_EDDI"]))
    
    return metrics

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load structured and unstructured data
    keep_cols = {"subject_id", "hadm_id", "short_term_mortality", "los_binary", 
                 "mechanical_ventilation", "age", "FIRST_WARDID", "LAST_WARDID", "ETHNICITY", "INSURANCE", "GENDER"}
    
    structured_data = pd.read_csv('final_structured_common.csv')
    new_columns = {col: f"{col}_struct" for col in structured_data.columns if col not in keep_cols}
    structured_data.rename(columns=new_columns, inplace=True)

    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    unstructured_data.drop(
        columns=["short_term_mortality", "los_binary", "mechanical_ventilation", "age", "segment", 
                 "admission_loc", "discharge_loc", "gender", "ethnicity", "insurance"],
        errors='ignore',
        inplace=True
    )
    
    merged_df = pd.merge(
        structured_data,
        unstructured_data,
        on=["subject_id", "hadm_id"],
        how="inner"
    )
    
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check your data and merge keys.")

    merged_df.columns = [col.lower().strip() for col in merged_df.columns]
    if "age_struct" in merged_df.columns:
        merged_df.rename(columns={"age_struct": "age"}, inplace=True)
    if "age" not in merged_df.columns:
        print("Column 'age' not found in merged dataframe; creating default 'age' column with zeros.")
        merged_df["age"] = 0

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

    required_cols = ["age", "first_wardid", "last_wardid", "gender", "ethnicity", "insurance"]
    for col in required_cols:
        if col not in df_filtered.columns:
            print(f"Column {col} not found in filtered dataframe; creating default values.")
            df_filtered[col] = 0

    df_unique = df_filtered.groupby("subject_id", as_index=False).first()
    print("Number of unique patients:", len(df_unique))
    
    if "segment" not in df_unique.columns:
        df_unique["segment"] = 0

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_unique, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    demographics_cols = ["age", "gender", "ethnicity", "insurance"]
    for col in demographics_cols:
        if col not in df_unique.columns:
            print(f"Column {col} not found; creating default values.")
            df_unique[col] = 0
        elif df_unique[col].dtype == object:
            df_unique[col] = df_unique[col].astype("category").cat.codes

    exclude_cols = set(["subject_id", "row_id", "hadm_id", "icustay_id",
                        "short_term_mortality", "los_binary", "mechanical_ventilation",
                        "age", "first_wardid", "last_wardid", "ethnicity", "insurance", "gender"])
    lab_feature_columns = [col for col in df_unique.columns 
                           if col not in exclude_cols and not col.startswith("note_") 
                           and pd.api.types.is_numeric_dtype(df_unique[col])]
    print("Number of lab feature columns:", len(lab_feature_columns))
    df_unique[lab_feature_columns] = df_unique[lab_feature_columns].fillna(0)

    num_samples = len(df_unique)
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)

    age_ids = torch.tensor(df_unique["age"].values, dtype=torch.long)
    segment_ids = torch.tensor(df_unique["segment"].values, dtype=torch.long)
    admission_loc_ids = torch.tensor(df_unique["first_wardid"].values, dtype=torch.long)
    discharge_loc_ids = torch.tensor(df_unique["last_wardid"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_unique["gender"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_unique["ethnicity"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_unique["insurance"].values, dtype=torch.long)

    labels_mortality = torch.tensor(df_unique["short_term_mortality"].values, dtype=torch.float32)
    labels_los = torch.tensor(df_unique["los_binary"].values, dtype=torch.float32)
    labels_vent = torch.tensor(df_unique["mechanical_ventilation"].values, dtype=torch.float32)

    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')
    
    dataset = TensorDataset(
        dummy_input_ids, dummy_attn_mask,
        age_ids, segment_ids, admission_loc_ids, discharge_loc_ids,
        gender_ids, ethnicity_ids, insurance_ids,
        aggregated_text_embeddings_t,
        labels_mortality, labels_los, labels_vent
    )
    
    labels_array = df_unique[["short_term_mortality", "los_binary", "mechanical_ventilation"]].values
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(msss.split(df_unique, labels_array))
    print("Train/Val samples:", len(train_val_idx), "Test samples:", len(test_idx))
    
    train_val_dataset = Subset(dataset, train_val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    labels_train_val = labels_array[train_val_idx]
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    train_idx_rel, val_idx_rel = next(msss_val.split(np.zeros(len(train_val_idx)), labels_train_val))
    train_idx = [train_val_idx[i] for i in train_idx_rel]
    val_idx = [train_val_idx[i] for i in val_idx_rel]
    print(f"Final split -> Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    disease_mapping = {d: i for i, d in enumerate(df_unique["hadm_id"].unique())}
    NUM_DISEASES = len(disease_mapping)
    NUM_AGES = df_unique["age"].nunique()
    NUM_SEGMENTS = 2
    NUM_ADMISSION_LOCS = df_unique["first_wardid"].nunique()
    NUM_DISCHARGE_LOCS = df_unique["last_wardid"].nunique()
    NUM_GENDERS = df_unique["gender"].nunique()
    NUM_ETHNICITIES = df_unique["ethnicity"].nunique()
    NUM_INSURANCES = df_unique["insurance"].nunique()
    
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
    
    print("Number of lab feature columns:", len(lab_feature_columns))
    df_unique[lab_feature_columns] = df_unique[lab_feature_columns].fillna(0)
    lab_features_np = df_unique[lab_feature_columns].values.astype(np.float32)
    lab_features_tensor = torch.tensor(lab_features_np, dtype=torch.float32)
    
    sensitive_attribute = torch.tensor(df_unique["ethnicity"].values, dtype=torch.long)
    freq_dict = compute_frequency_dict(sensitive_attribute)
    input_dim = lab_features_tensor.shape[1]
    hidden_dims = [500, 500, 500]
    noise_factor = 0.05
    fpm_model = StackedDenoisingAutoencoder(input_dim, hidden_dims, noise_factor=noise_factor).to(device)
    optimizer_fpm = Adam(fpm_model.parameters(), lr=1e-3)
    num_epochs = 20
    print("Pre-training Fair Patient Model (FPM)...")
    fpm_dataset = TensorDataset(lab_features_tensor, sensitive_attribute)
    fpm_loader = DataLoader(fpm_dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        loss = pretrain_fpm_autoencoder(fpm_model, fpm_loader, optimizer_fpm, device, freq_dict)
        print(f"[FPM Pretrain] Epoch {epoch+1}/{num_epochs} - Weighted Reconstruction Loss: {loss:.4f}")
    
    fpm_model.eval()
    with torch.no_grad():
        _, patient_repr = fpm_model(lab_features_tensor.to(device))
    print("FPM pretraining complete. Learned patient representations shape:", patient_repr.shape)
    
    X_repr = patient_repr.cpu().numpy()
    scaler = StandardScaler()
    X_repr_scaled = scaler.fit_transform(X_repr)
    labels_array = np.stack([df_unique["short_term_mortality"].values.astype(np.float32),
                             df_unique["los_binary"].values.astype(np.float32),
                             df_unique["mechanical_ventilation"].values.astype(np.float32)], axis=1)
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(msss.split(df_unique, labels_array))
    print("Train/Val samples:", len(train_val_idx), "Test samples:", len(test_idx))
    X_train = X_repr_scaled[train_val_idx]
    X_test = X_repr_scaled[test_idx]
    y_mort = df_unique["short_term_mortality"].values.astype(np.float32)
    y_los = df_unique["los_binary"].values.astype(np.float32)
    y_vent = df_unique["mechanical_ventilation"].values.astype(np.float32)
    y_mort_train = y_mort[train_val_idx]
    y_mort_test = y_mort[test_idx]
    y_los_train = y_los[train_val_idx]
    y_los_test = y_los[test_idx]
    y_vent_train = y_vent[train_val_idx]
    y_vent_test = y_vent[test_idx]
    
    age_attr = df_unique["age"].values.astype(np.int32)
    eth_attr = df_unique["ethnicity"].values.astype(np.int32)
    ins_attr = df_unique["insurance"].values.astype(np.int32)
    age_train = age_attr[train_val_idx]
    age_test = age_attr[test_idx]
    eth_train = eth_attr[train_val_idx]
    eth_test = eth_attr[test_idx]
    ins_train = ins_attr[train_val_idx]
    ins_test = ins_attr[test_idx]
    
    print("\n--- Downstream Evaluation ---")
    
    clf_mort = train_downstream_classifier(X_train, y_mort_train)
    metrics_mort, y_mort_prob, y_mort_pred = evaluate_downstream(clf_mort, X_test, y_mort_test)
    print("\nMortality Prediction Metrics:")
    print(f"  AUC-ROC: {metrics_mort['aucroc']:.4f}, AUPRC: {metrics_mort['auprc']:.4f}")
    print(f"  F1: {metrics_mort['f1']:.4f}, Recall (TPR): {metrics_mort['recall']:.4f}, Precision: {metrics_mort['precision']:.4f}")
    
    clf_los = train_downstream_classifier(X_train, y_los_train)
    metrics_los, y_los_prob, y_los_pred = evaluate_downstream(clf_los, X_test, y_los_test)
    print("\nLOS Prediction Metrics:")
    print(f"  AUC-ROC: {metrics_los['aucroc']:.4f}, AUPRC: {metrics_los['auprc']:.4f}")
    print(f"  F1: {metrics_los['f1']:.4f}, Recall (TPR): {metrics_los['recall']:.4f}, Precision: {metrics_los['precision']:.4f}")
    
    clf_vent = train_downstream_classifier(X_train, y_vent_train)
    metrics_vent, y_vent_prob, y_vent_pred = evaluate_downstream(clf_vent, X_test, y_vent_test)
    print("\nMechanical Ventilation Prediction Metrics:")
    print(f"  AUC-ROC: {metrics_vent['aucroc']:.4f}, AUPRC: {metrics_vent['auprc']:.4f}")
    print(f"  F1: {metrics_vent['f1']:.4f}, Recall (TPR): {metrics_vent['recall']:.4f}, Precision: {metrics_vent['precision']:.4f}")
    
    def print_fairness(y_true, y_prob, sens_array, threshold=0.5):
        dp_rate = np.mean((y_prob >= threshold).astype(int))
        eddi, subgroup_eddi = compute_eddi(y_true.astype(int), y_prob, sens_array, threshold)
        return dp_rate, eddi, subgroup_eddi

    print("\n--- Fairness Evaluation ---")
    outcomes = {
        "Mortality": (y_mort_test, y_mort_prob),
        "LOS": (y_los_test, y_los_prob),
        "Mechanical Ventilation": (y_vent_test, y_vent_prob)
    }
    age_order = ["15-29", "30-49", "50-69", "70-89", "Other"]
    ethnicity_order = ["white", "black", "asian", "hispanic", "other"]
    insurance_order = ["government", "medicare", "medicaid", "self pay", "private", "other"]
    sensitive_dict = {
        "Age": np.array([get_age_bucket(a) for a in age_test]),
        "Ethnicity": np.array([map_ethnicity(e) for e in eth_test]),
        "Insurance": np.array([map_insurance(i) for i in ins_test])
    }
    
    for outcome_name, (y_true, y_prob) in outcomes.items():
        print(f"\nOutcome: {outcome_name}")
        for sens_name, sens_array in sensitive_dict.items():
            dp, eddi, subgroup_eddi = print_fairness(y_true, y_prob, sens_array, threshold=0.5)
            print(f"  Sensitive Attribute: {sens_name}")
            print(f"    Demographic Parity (Positive Prediction Rate): {dp:.4f}")
            print(f"    Aggregated EDDI: {eddi:.4f}")
            if sens_name == "Age":
                order = age_order
            elif sens_name == "Ethnicity":
                order = ethnicity_order
            elif sens_name == "Insurance":
                order = insurance_order
            else:
                order = list(np.unique(sens_array))
            print("    Subgroup EDDI:")
            for subgroup in order:
                score = subgroup_eddi.get(subgroup, 0)
                print(f"      {subgroup}: {score:.4f}")
    
    print("\n--- Subgroup TPR and FPR Calculation for Each Outcome ---")
    all_mort_preds = []
    all_los_preds = []
    all_vent_preds = []
    all_mort_true = []
    # Accumulate raw sensitive values from the test set.
    all_age_sens_raw = []
    all_eth_sens_raw = []
    all_ins_sens_raw = []
    multimodal_model = None  

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
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
                nn.Linear(hidden_size, 3)
            )
    
        def forward(self, dummy_input_ids, dummy_attn_mask,
                    age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                    gender_ids, ethnicity_ids, insurance_ids,
                    aggregated_text_embedding):
            structured_emb = self.BEHRT(dummy_input_ids, dummy_attn_mask,
                                        age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                                        gender_ids, ethnicity_ids, insurance_ids)
            ts_proj = self.ts_projector(structured_emb)
            text_proj = self.text_projector(aggregated_text_embedding)
            combined = torch.cat((ts_proj, text_proj), dim=1)
            logits = self.classifier(combined)
            mortality_logits = logits[:, 0].unsqueeze(1)
            los_logits = logits[:, 1].unsqueeze(1)
            vent_logits = logits[:, 2].unsqueeze(1)
            return mortality_logits, los_logits, vent_logits
    
    multimodal_model = MultimodalTransformer(
        text_embed_size=768,
        BEHRT=behrt_model,
        device=device,
        hidden_size=512
    ).to(device)
    
    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    best_model_path = "best_multimodal_model.pt"
    
    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = 0.0
        for batch in train_loader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
    
            optimizer.zero_grad()
            mortality_logits, los_logits, vent_logits = multimodal_model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            loss_mort = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
            loss_los = criterion_los(los_logits, labels_los.unsqueeze(1))
            loss_vent = criterion_mech(vent_logits, labels_vent.unsqueeze(1))
            loss = loss_mort + loss_los + loss_vent
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model_loss(multimodal_model, val_loader, device,
                                       criterion_mortality, criterion_los, criterion_mech)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
    
        val_metrics = evaluate_model_metrics(multimodal_model, val_loader, device, threshold=0.5)
        print(f"--- Validation Metrics at Epoch {epoch+1} ---")
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            m = val_metrics[outcome]
            print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
                  f"F1: {m['f1']:.4f}, Recall (TPR): {m['recall']:.4f}, Precision: {m['precision']:.4f}, "
                  f"TPR: {m['tpr']:.4f}, FPR: {m['fpr']:.4f}")
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(multimodal_model.state_dict(), best_model_path)
            print("  Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"  No improvement in validation loss. Patience counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
    
    multimodal_model.load_state_dict(torch.load(best_model_path))
    print("\nEvaluating on test set...")
    metrics = evaluate_model_metrics(multimodal_model, test_loader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation Metrics on Test Set:")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
              f"F1: {m['f1']:.4f}, Recall (TPR): {m['recall']:.4f}, Precision: {m['precision']:.4f}, "
              f"TPR: {m['tpr']:.4f}, FPR: {m['fpr']:.4f}")
    
    if "eddi_stats" in metrics:
        print("\nDetailed EDDI Statistics on Test Set:")
        eddi_stats = metrics["eddi_stats"]
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            print(f"\n{outcome.capitalize()} EDDI Stats:")
            stats = eddi_stats[outcome]
            print("  Aggregated Age EDDI    : {:.4f}".format(stats["age_eddi"]))
            print("  Age Subgroup EDDI      :", stats["age_subgroup_eddi"])
            print("  Aggregated Ethnicity EDDI: {:.4f}".format(stats["ethnicity_eddi"]))
            print("  Ethnicity Subgroup EDDI:", stats["ethnicity_subgroup_eddi"])
            print("  Aggregated Insurance EDDI: {:.4f}".format(stats["insurance_eddi"]))
            print("  Insurance Subgroup EDDI:", stats["insurance_subgroup_eddi"])
            print("  Final Overall {} EDDI: {:.4f}".format(outcome.capitalize(), stats["final_EDDI"]))
    
    # Compute subgroup TPR/FPR for each outcome
    # Accumulate raw sensitive attributes from test loader
    all_mort_preds = []
    all_los_preds = []
    all_vent_preds = []
    all_mort_true = []
    all_age_raw = []
    all_eth_raw = []
    all_ins_raw = []
    
    with torch.no_grad():
        for batch in test_loader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
            mortality_logits, los_logits, vent_logits = multimodal_model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            mort_probs = torch.sigmoid(mortality_logits).cpu().numpy().squeeze()
            los_probs = torch.sigmoid(los_logits).cpu().numpy().squeeze()
            vent_probs = torch.sigmoid(vent_logits).cpu().numpy().squeeze()
            all_mort_preds.extend((mort_probs > 0.5).astype(int).tolist())
            all_los_preds.extend((los_probs > 0.5).astype(int).tolist())
            all_vent_preds.extend((vent_probs > 0.5).astype(int).tolist())
            all_mort_true.extend(labels_mortality.cpu().numpy().squeeze().tolist())
            all_age_raw.extend(age_ids.cpu().numpy().squeeze().tolist())
            all_eth_raw.extend(ethnicity_ids.cpu().numpy().squeeze().tolist())
            all_ins_raw.extend(insurance_ids.cpu().numpy().squeeze().tolist())
    
    # Map raw sensitive values to subgroup labels using the mapping functions.
    all_age_sens = [get_age_bucket(a) for a in all_age_raw]
    all_eth_sens = [map_ethnicity(e) for e in all_eth_raw]
    all_ins_sens = [map_insurance(i) for i in all_ins_raw]
    
    outcomes_pred = {
        "Mortality": (np.array(all_mort_true), np.array(all_mort_preds)),
        "LOS": (np.array(all_mort_true), np.array(all_los_preds)),  # Use same labels if separate ones aren't available.
        "Mechanical Ventilation": (np.array(all_mort_true), np.array(all_vent_preds))
    }
    
    overall_avg = {}
    for outcome_name, (y_true, y_pred) in outcomes_pred.items():
        print(f"\nOutcome: {outcome_name}")
        avg_tprs = []
        avg_fprs = []
        for sens_name, sens_array in zip(["Age", "Ethnicity", "Insurance"],
                                         [np.array(all_age_sens), np.array(all_eth_sens), np.array(all_ins_sens)]):
            subgroup_results, avg_tpr, avg_fpr = calculate_subgroup_tpr_fpr(y_true, y_pred, sens_array)
            print(f"  Sensitive Attribute: {sens_name}")
            for group, metrics_dict in subgroup_results.items():
                print(f"    Group {group}: TPR = {metrics_dict['TPR']:.3f}, FPR = {metrics_dict['FPR']:.3f}")
            print(f"    Average TPR for {sens_name}: {avg_tpr:.3f}")
            print(f"    Average FPR for {sens_name}: {avg_fpr:.3f}\n")
            avg_tprs.append(avg_tpr)
            avg_fprs.append(avg_fpr)
        overall_avg_tpr = np.mean(avg_tprs) if avg_tprs else 0
        overall_avg_fpr = np.mean(avg_fprs) if avg_fprs else 0
        overall_avg[outcome_name] = {"Overall Average TPR": overall_avg_tpr, "Overall Average FPR": overall_avg_fpr}
        print(f"Overall Average TPR for {outcome_name}: {overall_avg_tpr:.3f}")
        print(f"Overall Average FPR for {outcome_name}: {overall_avg_fpr:.3f}")
    
    print("FPM Baseline training, evaluation, and fairness analysis complete.")

if __name__ == "__main__":
    train_pipeline()
