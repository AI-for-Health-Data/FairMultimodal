import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, Subset
from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import (accuracy_score, recall_score, precision_score, roc_auc_score,
                             average_precision_score, f1_score, roc_curve, confusion_matrix)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pickle
import math
import matplotlib.pyplot as plt
import os

DEBUG = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_fairness_metrics(labels, predictions, demographics, sensitive_class):
    """
    Calculate fairness metrics such as Equal Opportunity and Equal Odds for a given sensitive_class.
    """
    sensitive_indices = demographics == sensitive_class
    non_sensitive_indices = ~sensitive_indices

    tn_s, fp_s, fn_s, tp_s = confusion_matrix(labels[sensitive_indices], predictions[sensitive_indices]).ravel()
    tn_ns, fp_ns, fn_ns, tp_ns = confusion_matrix(labels[non_sensitive_indices], predictions[non_sensitive_indices]).ravel()

    tpr_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) != 0 else 0
    fpr_s = fp_s / (fp_s + tn_s) if (fp_s + tn_s) != 0 else 0

    tpr_ns = tp_ns / (tp_ns + fn_ns) if (tp_ns + fn_ns) != 0 else 0
    fpr_ns = fp_ns / (fp_ns + tn_ns) if (fp_ns + tn_ns) != 0 else 0

    eod = tpr_s - tpr_ns
    eod_fpr = fpr_s - fpr_ns
    eod_tpr = tpr_s - tpr_ns
    avg_eod = (abs(eod_fpr) + abs(eod_tpr)) / 2

    return {
        "TPR Sensitive": tpr_s,
        "TPR Non-Sensitive": tpr_ns,
        "FPR Sensitive": fpr_s,
        "FPR Non-Sensitive": fpr_ns,
        "Equal Opportunity Difference": eod,
        "Equalized Odds Difference (FPR)": eod_fpr,
        "Equalized Odds Difference (TPR)": eod_tpr,
        "Average Equalized Odds Difference": avg_eod
    }

def calculate_multiclass_fairness_metrics(labels, predictions, demographics):
    """
    Calculate fairness metrics for all classes in the sensitive attribute.
    """
    unique_classes = np.unique(demographics)
    metrics = {}

    for sensitive_class in unique_classes:
        sensitive_indices = demographics == sensitive_class
        non_sensitive_indices = ~sensitive_indices

        tn_s, fp_s, fn_s, tp_s = confusion_matrix(labels[sensitive_indices], predictions[sensitive_indices]).ravel()
        tn_ns, fp_ns, fn_ns, tp_ns = confusion_matrix(labels[non_sensitive_indices], predictions[non_sensitive_indices]).ravel()

        tpr_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) != 0 else 0
        fpr_s = fp_s / (fp_s + tn_s) if (fp_s + tn_s) != 0 else 0

        tpr_ns = tp_ns / (tp_ns + fn_ns) if (tp_ns + fn_ns) != 0 else 0
        eod = tpr_s - tpr_ns
        avg_eod = (abs(fp_s / (fp_s + tn_s) - (fp_ns / (fp_ns + tn_ns))) + abs(eod)) / 2

        metrics[sensitive_class] = {
            "TPR": tpr_s,
            "FPR": fpr_s,
            "Equal Opportunity Difference": eod,
            "Average Equalized Odds Difference": avg_eod
        }

    return metrics

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
    cm = confusion_matrix(np.array(y_true)[group_mask], np.array(y_pred)[group_mask], labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

def calculate_sd_for_rates(y_true, y_pred, sensitive_attr):
    """
    Calculate the standard deviation of TPR and FPR across all classes of a sensitive attribute.
    """
    unique_classes = np.unique(sensitive_attr)
    tpr_values = []
    fpr_values = []

    for group in unique_classes:
        group_mask = sensitive_attr == group
        tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred, group_mask)
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    sd_tpr = np.std(tpr_values, ddof=1)  # Sample standard deviation
    sd_fpr = np.std(fpr_values, ddof=1)
    
    return sd_tpr, sd_fpr

def calculate_equalized_odds_difference(y_true, y_pred, sensitive_attr):
    """
    Calculate the average absolute differences in TPR and FPR over all pairs of sensitive groups.
    """
    unique_classes = np.unique(sensitive_attr)
    tpr_diffs = []
    fpr_diffs = []

    for i, group1 in enumerate(unique_classes):
        for group2 in unique_classes[i+1:]:
            group1_mask = sensitive_attr == group1
            group2_mask = sensitive_attr == group2
            tpr1, fpr1 = calculate_tpr_and_fpr(y_true, y_pred, group1_mask)
            tpr2, fpr2 = calculate_tpr_and_fpr(y_true, y_pred, group2_mask)

            tpr_diffs.append(abs(tpr1 - tpr2))
            fpr_diffs.append(abs(fpr1 - fpr2))

    avg_tpr_diff = np.mean(tpr_diffs)
    avg_fpr_diff = np.mean(fpr_diffs)

    return avg_tpr_diff, avg_fpr_diff

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
    mapping = {0: "government", 1: "medicare", 2: "medicaid", 3: "private", 4: "self pay"}
    return mapping.get(code, "other")

def compute_eddi(y_true, y_pred, sensitive_labels, threshold=0.5):
    y_pred_binary = (np.array(y_pred) > threshold).astype(int)
    unique_groups = np.unique(sensitive_labels)
    subgroup_eddi = {}
    overall_error = np.mean(y_pred_binary != y_true)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0

    for group in unique_groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            er_group = np.mean(y_pred_binary[mask] != y_true[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom

    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

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
        self.bert = BertModel(config).to(device)
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
            nn.Linear(hidden_size, 3)  # Predict three outcomes: mortality, LOS, mechanical ventilation.
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

def train_step(model, dataloader, optimizer, device, crit_mort, crit_los, crit_vent):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (dummy_input_ids, dummy_attn_mask,
         age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
         gender_ids, ethnicity_ids, insurance_ids,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]

        # Ensure target tensors have shape [batch, 1]
        if labels_mortality.dim() == 1:
            labels_mortality = labels_mortality.view(-1, 1)
        if labels_los.dim() == 1:
            labels_los = labels_los.view(-1, 1)
        if labels_vent.dim() == 1:
            labels_vent = labels_vent.view(-1, 1)

        optimizer.zero_grad()
        mortality_logits, los_logits, vent_logits = model(
            dummy_input_ids, dummy_attn_mask,
            age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
            gender_ids, ethnicity_ids, insurance_ids,
            aggregated_text_embedding
        )
        loss_mort = crit_mort(mortality_logits, labels_mortality)
        loss_los = crit_los(los_logits, labels_los)
        loss_vent = crit_vent(vent_logits, labels_vent)
        loss = loss_mort + loss_los + loss_vent
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

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
            if labels_mortality.dim() == 1:
                labels_mortality = labels_mortality.view(-1, 1)
            if labels_los.dim() == 1:
                labels_los = labels_los.view(-1, 1)
            if labels_vent.dim() == 1:
                labels_vent = labels_vent.view(-1, 1)
            mort_logits, los_logits, mech_logits = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            loss_mort = crit_mort(mort_logits, labels_mortality)
            loss_los = crit_los(los_logits, labels_los)
            loss_vent = crit_vent(mech_logits, labels_vent)
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
        recall_val = recall_score(labels, preds, zero_division=0)
        precision_val = precision_score(labels, preds, zero_division=0)
        TP = ((preds == 1) & (labels == 1)).sum()
        FP = ((preds == 1) & (labels == 0)).sum()
        TN = ((preds == 0) & (labels == 0)).sum()
        FN = ((preds == 0) & (labels == 1)).sum()
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall_val, "precision": precision_val,
                         "tpr": TPR, "fpr": FPR}

    if print_eddi:
        age_order_list = ["15-29", "30-49", "50-69", "70-89", "Other"]
        ethnicity_order_list = ["white", "black", "asian", "hispanic", "other"]
        insurance_order_list = ["government", "medicare", "medicaid", "private", "self pay", "other"]
        all_age_cat = torch.cat(all_age, dim=0).numpy().squeeze()
        all_eth_cat = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
        all_ins_cat = torch.cat(all_insurance, dim=0).numpy().squeeze()
        age_labels = np.array([get_age_bucket(a) for a in all_age_cat])
        eth_labels = np.array([map_ethnicity(e) for e in all_eth_cat])
        ins_labels = np.array([map_insurance(i) for i in all_ins_cat])
        eddi_stats = {}
        for task, labels_np, probs in zip(["mortality", "los", "mechanical_ventilation"],
                                          [labels_mort_np, labels_los_np, labels_mech_np],
                                          [mort_probs, los_probs, mech_probs]):
            overall_age, age_sub = compute_eddi(labels_np.astype(int), probs, age_labels, threshold)
            overall_eth, eth_sub = compute_eddi(labels_np.astype(int), probs, eth_labels, threshold)
            overall_ins, ins_sub = compute_eddi(labels_np.astype(int), probs, ins_labels, threshold)
            total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
            eddi_stats[task] = {
                "age_eddi": overall_age,
                "age_subgroup_eddi": age_sub,
                "ethnicity_eddi": overall_eth,
                "ethnicity_subgroup_eddi": eth_sub,
                "insurance_eddi": overall_ins,
                "insurance_subgroup_eddi": ins_sub,
                "final_EDDI": total_eddi
            }
        metrics["eddi_stats"] = eddi_stats
        print("\n--- Detailed EDDI Statistics on Test Set ---")
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

    # Calculate subgroup TPR and FPR for each sensitive attribute (age, ethnicity, insurance)
    sensitive_attrs = {
        "age": np.array([get_age_bucket(a) for a in torch.cat(all_age, dim=0).numpy().squeeze()]),
        "ethnicity": np.array([map_ethnicity(e) for e in torch.cat(all_ethnicity, dim=0).numpy().squeeze()]),
        "insurance": np.array([map_insurance(i) for i in torch.cat(all_insurance, dim=0).numpy().squeeze()])
    }
    outcome_names = ["mortality", "los", "mechanical_ventilation"]
    probs_list = [mort_probs, los_probs, mech_probs]
    labels_list = [labels_mort_np, labels_los_np, labels_mech_np]
    fairness_results = {}
    for outcome, probs, labels in zip(outcome_names, probs_list, labels_list):
        preds = (probs > threshold).astype(int)
        fairness_results[outcome] = {}
        for attr_name, attr_values in sensitive_attrs.items():
            unique_groups = np.unique(attr_values)
            fairness_results[outcome][attr_name] = {}
            tpr_list = []
            fpr_list = []
            for group in unique_groups:
                group_mask = attr_values == group
                tpr, fpr = calculate_tpr_and_fpr(labels, preds, group_mask)
                fairness_results[outcome][attr_name][group] = {"TPR": tpr, "FPR": fpr}
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            overall_tpr = np.mean(tpr_list)
            overall_fpr = np.mean(fpr_list)
            fairness_results[outcome][attr_name]["overall"] = {"average_TPR": overall_tpr, "average_FPR": overall_fpr}
    print("\n--- Fairness Metrics (TPR & FPR) per Sensitive Attribute and Outcome ---")
    for outcome in outcome_names:
        print(f"\nOutcome: {outcome.capitalize()}")
        for attr_name in sensitive_attrs.keys():
            print(f" Sensitive attribute: {attr_name.capitalize()}")
            for key, val in fairness_results[outcome][attr_name].items():
                if key == "overall":
                    print(f"   Overall - Average TPR: {val['average_TPR']:.3f}, Average FPR: {val['average_FPR']:.3f}")
                else:
                    print(f"   {key}: TPR: {val['TPR']:.3f}, FPR: {val['FPR']:.3f}")

    # Calculate overall Equalized Odds Metrics for each sensitive attribute
    print("\n--- Overall Equalized Odds Metrics for each Sensitive Attribute and Outcome ---")
    for outcome, probs, labels in zip(outcome_names, probs_list, labels_list):
        preds = (probs > threshold).astype(int)
        print(f"\nOutcome: {outcome.capitalize()}")
        for attr_name, attr_values in sensitive_attrs.items():
            avg_tpr_diff, avg_fpr_diff = calculate_equalized_odds_difference(labels, preds, attr_values)
            combined_metric = (avg_tpr_diff + avg_fpr_diff) / 2
            print(f" {attr_name.capitalize()} - Average TPR Difference: {avg_tpr_diff:.3f}, Average FPR Difference: {avg_fpr_diff:.3f}, Combined Metric: {combined_metric:.3f}")
    # ------------------ End New Fairness Metrics Section ------------------

    return metrics

import itertools

hyperparameter_list = ['learning_rate', 'num_iters', 'num_nodes', 'num_nodes_adv', 'dropout_rate', 'alpha']
get_new_control_indices = False
use_data_as_is = False

class Adv_Model(object):
    def __init__(self, params):
        self.params = params
        self.method = self.params['method']
        self.adversarial = self.method != 'basic'
        self.num_classes = self.params['num_classes']
        self.hyperparameters = self.params['hyperparameters']
        self.model = self.build_model()
        self.data = self.data_processing()

    def get_indexes(self):
        num_models = []
        for i in range(len(hyperparameter_list)):
            if i < 3 or i == 4 or self.adversarial:
                num_models.append(range(len(self.hyperparameters[hyperparameter_list[i]])))
            else:
                num_models.append([None])
        return itertools.product(*num_models)

    def get_hyperparameters(self, indexes):
        hyperparams = []
        for i in range(len(indexes)):
            if i < 3 or i == 4 or self.adversarial:
                hyperparams.append(self.hyperparameters[hyperparameter_list[i]][indexes[i]])
            else:
                hyperparams.append(None)
        return hyperparams

    def params_tostring(self, indexes):
        res = ''
        for i in range(len(hyperparameter_list)):
            if i > 0:
                res += '-'
            if i < 3 or i == 4 or self.adversarial:
                res += hyperparameter_list[i] + '_' + str(self.hyperparameters[hyperparameter_list[i]][indexes[i]])
        return res

    def create_dir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def data_processing(self):
        data = dict()
        data['Xtrain'] = self.params['Xtrain']
        data['ytrain'] = self.params['ytrain']
        data['Xvalid'] = self.params['Xvalid']
        data['yvalid'] = self.params['yvalid']
        if self.num_classes > 2:
            data['ztrain'] = self.params['ztrain']
            data['zvalid'] = self.params['zvalid']
        else:
            data['ztrain'] = self.params['ztrain']
            data['zvalid'] = self.params['zvalid']
        return data

    def build_model(self):
        models = {}
        for indexes in self.get_indexes():
            models[indexes] = self.build_single_model(indexes)
        return models

    def build_single_model(self, indexes):
        model = dict()
        num_nodes = self.hyperparameters['num_nodes'][indexes[2]]
        j = self.params['Xtrain'].shape[1]
        model['model'] = nn.Sequential(
            nn.Linear(j, num_nodes),
            nn.ReLU(),
            nn.Dropout(self.hyperparameters['dropout_rate'][indexes[4]]),
            nn.Linear(num_nodes, 1),
            nn.Sigmoid()
        )
        model['loss_function'] = nn.BCELoss()
        model['optimizer'] = Adam(model['model'].parameters(),
                                  lr=self.hyperparameters['learning_rate'][indexes[0]])
        if self.adversarial:
            num_nodes_adv = self.hyperparameters['num_nodes_adv'][indexes[3]]
            num_nodes_out = self.num_classes if self.num_classes > 2 else 1
            n_adv = 2  # predictor output and label concatenated.
            if self.num_classes > 2:
                model['adversarial_model'] = nn.Sequential(
                    nn.Linear(n_adv, num_nodes_adv),
                    nn.ReLU(),
                    nn.Dropout(self.hyperparameters['dropout_rate'][indexes[4]]),
                    nn.Linear(num_nodes_adv, num_nodes_out),
                    nn.Softmax(dim=1)
                )
                model['adversarial_loss_function'] = nn.CrossEntropyLoss()
            else:
                model['adversarial_model'] = nn.Sequential(
                    nn.Linear(n_adv, num_nodes_adv),
                    nn.ReLU(),
                    nn.Dropout(self.hyperparameters['dropout_rate'][indexes[4]]),
                    nn.Linear(num_nodes_adv, num_nodes_out),
                    nn.Sigmoid()
                )
                model['adversarial_loss_function'] = nn.BCELoss()
            model['adversarial_optimizer'] = Adam(model['adversarial_model'].parameters(),
                                                  lr=self.hyperparameters['learning_rate'][indexes[0]])
        return model

    def train(self):
        for indexes in self.get_indexes():
            self.train_single_model(indexes)

    def train_single_model(self, indexes):
        model_dict = self.model[indexes]
        model = model_dict['model']
        loss_function = model_dict['loss_function']
        optimizer = model_dict['optimizer']
        Xtrain = torch.tensor(self.params['Xtrain'].values).float()
        ytrain = torch.tensor(self.params['ytrain'].values).float().view(-1, 1)
        Xvalid = torch.tensor(self.params['Xvalid'].values).float()
        yvalid = torch.tensor(self.params['yvalid'].values).float().view(-1, 1)
        ztrain = torch.tensor(self.params['ztrain'].values).long().view(-1)

        if not use_data_as_is:
            idx_case = [i for i in range(len(ytrain)) if ytrain[i] == 1]
            idx_control = [i for i in range(len(ytrain)) if ytrain[i] == 0]
            match_number = 20
            if get_new_control_indices or not os.path.exists('control_indices.pkl'):
                matched_cohort_indices = []
                for i in idx_case:
                    matched = random.sample(idx_control, min(match_number, len(idx_control)))
                    matched_cohort_indices.extend(matched)
                with open('control_indices.pkl', 'wb') as f:
                    pickle.dump(matched_cohort_indices, f)
            else:
                with open('control_indices.pkl', 'rb') as f:
                    matched_cohort_indices = pickle.load(f)
            Xtrain = torch.cat((Xtrain[matched_cohort_indices, :], Xtrain[idx_case, :]), dim=0)
            ytrain = torch.cat((ytrain[matched_cohort_indices, :], ytrain[idx_case, :]), dim=0)
            ztrain = torch.cat((ztrain[matched_cohort_indices], ztrain[idx_case]), dim=0)

        # --- Resample X, y, and z together ---
        from imblearn.combine import SMOTEENN
        from imblearn.under_sampling import EditedNearestNeighbours
        resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'), random_state=25)
        # Concatenate ztrain as an extra column to Xtrain
        Xz = np.concatenate((Xtrain.numpy(), ztrain.numpy().reshape(-1, 1)), axis=1)
        Xz_res, ytrain_res = resample.fit_resample(Xz, ytrain.numpy())
        Xtrain = torch.tensor(Xz_res[:, :-1]).float()
        ztrain = torch.tensor(Xz_res[:, -1]).long()
        ytrain = torch.tensor(ytrain_res).float().view(-1, 1)

        if self.adversarial:
            adv_model = model_dict['adversarial_model']
            adv_loss_function = model_dict['adversarial_loss_function']
            adv_optimizer = model_dict['adversarial_optimizer']

        num_iters = self.hyperparameters['num_iters'][indexes[1]]
        train_loss_list = []
        valid_loss_list = []
        epoch_list = []
        for t in range(num_iters):
            ypred_train = model(Xtrain)
            loss_train = loss_function(ypred_train, ytrain)
            if self.adversarial:
                adv_input_train = torch.cat((ypred_train, ytrain), dim=1)
                zpred_train = adv_model(adv_input_train)
                adv_loss_train = adv_loss_function(zpred_train.squeeze(), ztrain.float())
                combined_loss_train = loss_train - self.hyperparameters['alpha'][indexes[5]] * adv_loss_train + loss_train/(adv_loss_train+1e-8)
            else:
                combined_loss_train = loss_train

            optimizer.zero_grad()
            if self.adversarial:
                adv_optimizer.zero_grad()
                adv_loss_train.backward(retain_graph=True)
            combined_loss_train.backward()
            optimizer.step()
            if self.adversarial:
                adv_optimizer.step()
            train_loss_list.append(combined_loss_train.item())

            ypred_valid = model(Xvalid)
            loss_valid = loss_function(ypred_valid, yvalid)
            if self.adversarial:
                adv_input_valid = torch.cat((ypred_valid, yvalid), dim=1)
                zpred_valid = adv_model(adv_input_valid)
                adv_loss_valid = adv_loss_function(zpred_valid.squeeze(), torch.tensor(self.params['zvalid'].values).long())
                combined_loss_valid = loss_valid - self.hyperparameters['alpha'][indexes[5]] * adv_loss_valid + loss_valid/(adv_loss_valid+1e-8)
            else:
                combined_loss_valid = loss_valid
            valid_loss_list.append(combined_loss_valid.item())

            if t % 100 == 0:
                print(f"Iteration: {t}, Train Loss: {combined_loss_train.item():.4f}, Valid Loss: {combined_loss_valid.item():.4f}")
                epoch_list.append(t)
            if t > 0 and t % 10000 == 0:
                torch.save(model, "model/model-basic.pth")
                if self.adversarial:
                    torch.save(adv_model, "adv/model-adv.pth")

        plt.plot(epoch_list, train_loss_list[:len(epoch_list)], color='blue', label="Train Loss")
        plt.plot(epoch_list, valid_loss_list[:len(epoch_list)], color='red', label="Valid Loss")
        plt.legend()
        plt.savefig("loss_metrics.png")
        plt.close()
        torch.save(model, "model/model-basic_final.pth")
        if self.adversarial:
            torch.save(adv_model, "adv/model-adv_final.pth")
        print("Training complete for hyperparameter setting:", self.params_tostring(indexes))

    def evaluate(self):
        eval_file = 'metrics.csv'
        all_metrics = []
        for indexes in self.get_indexes():
            all_metrics.append(self.evaluate_single_model(indexes))
        pd.concat(all_metrics).to_csv(eval_file)
        print("Evaluation metrics saved to", eval_file)

    def evaluate_single_model(self, indexes):
        model = self.model[indexes]['model']
        Xvalid = torch.tensor(self.params['Xvalid'].values).float()
        yvalid = torch.tensor(self.params['yvalid'].values).float().view(-1, 1)
        zvalid = torch.tensor(self.params['zvalid'].values).long().view(-1)
        ypred_valid = model(Xvalid)
        zpred_valid = None
        if self.adversarial:
            adv_model = self.model[indexes]['adversarial_model']
            adv_input_valid = torch.cat((ypred_valid, yvalid), dim=1)
            zpred_valid = adv_model(adv_input_valid)
        metrics = get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(),
                              self.get_hyperparameters(indexes), k=self.num_classes, eval_file='valid_set',
                              zpred=zpred_valid.data.numpy() if zpred_valid is not None else None)
        return pd.DataFrame(metrics, index=[0])

def get_metrics(ypred, y, z, hyperparameters, k=7, yselect=0, eval_file=None, zpred=None):
    metrics = dict()
    metrics['eval_file'] = eval_file
    hyperparameter_list = ['learning_rate', 'num_iters', 'num_nodes', 'num_nodes_adv', 'dropout_rate', 'alpha']
    for i in range(len(hyperparameter_list)):
        metrics[hyperparameter_list[i]] = hyperparameters[i]

    pred = (ypred >= 0.5)
    true_pos = np.sum((pred == 1) & (y == 1))
    true_neg = np.sum((pred == 0) & (y == 0))
    false_pos = np.sum((pred == 1) & (y == 0))
    false_neg = np.sum((pred == 0) & (y == 1))
    metrics['accuracy'] = accuracy_score(y, pred)
    metrics['recall'] = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    metrics['precision'] = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    metrics['specificity'] = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
    prev = 0.05
    metrics['ppv'] = (metrics['recall'] * prev) / (metrics['recall'] * prev + (1 - metrics['specificity']) * (1 - prev))
    metrics['npv'] = (metrics['specificity'] * (1 - prev)) / (metrics['specificity'] * (1 - prev) + (1 - metrics['recall']) * prev)
    metrics['f1score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-8)
    try:
        metrics['roc_auc'] = roc_auc_score(y, pred)
    except Exception:
        metrics['roc_auc'] = float('nan')
    return metrics

if __name__ == '__main__':
    structured_data = pd.read_csv('final_structured_common.csv')
    unstructured_data = pd.read_csv('final_unstructured_common.csv', low_memory=False)

    structured_data.columns = [col.lower().strip() for col in structured_data.columns]
    if "age_struct" in structured_data.columns:
        structured_data.rename(columns={"age_struct": "age"}, inplace=True)
    if "age" not in structured_data.columns:
        print("Column 'age' not found; creating default 'age' column with zeros.")
        structured_data["age"] = 0

    unstructured_data.drop(
        columns=["short_term_mortality", "los_binary", "mechanical_ventilation", "age",
                 "segment", "admission_loc", "discharge_loc", "gender", "ethnicity", "insurance"],
        errors='ignore',
        inplace=True
    )

    merged_df = pd.merge(
        structured_data,
        unstructured_data,
        on=["subject_id", "hadm_id"],
        how="inner"
    ).head(1000)
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check your data and merge keys.")

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df.columns = [col.lower().strip() for col in merged_df.columns]
    for outcome in ["short_term_mortality", "los_binary", "mechanical_ventilation"]:
        merged_df[outcome] = merged_df[outcome].astype(int)

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
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(df_unique, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")
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
    X_df = df_unique[lab_feature_columns].copy()
    y_df = df_unique[["short_term_mortality"]].copy()
    z_df = df_unique[["ethnicity"]].copy()  # sensitive attribute

    # Convert binary outcome into a multilabel indicator (one-hot encoded)
    y_values = y_df.values.astype(int).reshape(-1)
    labels_array = np.concatenate([(y_values == 0).astype(int).reshape(-1,1),
                                   (y_values == 1).astype(int).reshape(-1,1)], axis=1)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(msss.split(df_unique, labels_array))
    print("Train/Val samples:", len(train_val_idx), "Test samples:", len(test_idx))
    labels_train_val = labels_array[train_val_idx]
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    train_idx_rel, val_idx_rel = next(msss_val.split(np.zeros(len(train_val_idx)), labels_train_val))
    train_idx = [train_val_idx[i] for i in train_idx_rel]
    val_idx = [train_val_idx[i] for i in val_idx_rel]
    print(f"Final split -> Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = Subset(TensorDataset(
        torch.zeros((len(df_unique), 1), dtype=torch.long),  # dummy_input_ids
        torch.ones((len(df_unique), 1), dtype=torch.long),   # dummy_attn_mask
        torch.tensor(df_unique["age"].values, dtype=torch.long),
        torch.tensor(df_unique["segment"].values, dtype=torch.long),
        torch.tensor(df_unique["first_wardid"].values, dtype=torch.long),
        torch.tensor(df_unique["last_wardid"].values, dtype=torch.long),
        torch.tensor(df_unique["gender"].values, dtype=torch.long),
        torch.tensor(df_unique["ethnicity"].values, dtype=torch.long),
        torch.tensor(df_unique["insurance"].values, dtype=torch.long),
        aggregated_text_embeddings_t,
        torch.tensor(y_df.values.reshape(-1,1), dtype=torch.float32),
        torch.tensor(df_unique["los_binary"].values.reshape(-1,1), dtype=torch.float32),
        torch.tensor(df_unique["mechanical_ventilation"].values.reshape(-1,1), dtype=torch.float32)
    ), train_idx)
    val_dataset = Subset(TensorDataset(
        torch.zeros((len(df_unique), 1), dtype=torch.long),
        torch.ones((len(df_unique), 1), dtype=torch.long),
        torch.tensor(df_unique["age"].values, dtype=torch.long),
        torch.tensor(df_unique["segment"].values, dtype=torch.long),
        torch.tensor(df_unique["first_wardid"].values, dtype=torch.long),
        torch.tensor(df_unique["last_wardid"].values, dtype=torch.long),
        torch.tensor(df_unique["gender"].values, dtype=torch.long),
        torch.tensor(df_unique["ethnicity"].values, dtype=torch.long),
        torch.tensor(df_unique["insurance"].values, dtype=torch.long),
        aggregated_text_embeddings_t,
        torch.tensor(y_df.values.reshape(-1,1), dtype=torch.float32),
        torch.tensor(df_unique["los_binary"].values.reshape(-1,1), dtype=torch.float32),
        torch.tensor(df_unique["mechanical_ventilation"].values.reshape(-1,1), dtype=torch.float32)
    ), val_idx)
    test_dataset = Subset(TensorDataset(
        torch.zeros((len(df_unique), 1), dtype=torch.long),
        torch.ones((len(df_unique), 1), dtype=torch.long),
        torch.tensor(df_unique["age"].values, dtype=torch.long),
        torch.tensor(df_unique["segment"].values, dtype=torch.long),
        torch.tensor(df_unique["first_wardid"].values, dtype=torch.long),
        torch.tensor(df_unique["last_wardid"].values, dtype=torch.long),
        torch.tensor(df_unique["gender"].values, dtype=torch.long),
        torch.tensor(df_unique["ethnicity"].values, dtype=torch.long),
        torch.tensor(df_unique["insurance"].values, dtype=torch.long),
        aggregated_text_embeddings_t,
        torch.tensor(y_df.values.reshape(-1,1), dtype=torch.float32),
        torch.tensor(df_unique["los_binary"].values.reshape(-1,1), dtype=torch.float32),
        torch.tensor(df_unique["mechanical_ventilation"].values.reshape(-1,1), dtype=torch.float32)
    ), test_idx)

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

    multimodal_model = MultimodalTransformer(
        text_embed_size=768,
        BEHRT=behrt_model,
        device=device,
        hidden_size=512
    ).to(device)

    optimizer = Adam(multimodal_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    best_model_path = "best_multimodal_model.pt"

    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')

    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                  criterion_mortality, criterion_los, criterion_mech)
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model_loss(multimodal_model, val_loader, device,
                                       criterion_mortality, criterion_los, criterion_mech)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        val_metrics = evaluate_model_metrics(multimodal_model, val_loader, device, threshold=0.5)
        print(f"--- Validation Metrics at Epoch {epoch+1} ---")
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            m = val_metrics[outcome]
            print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, " +
                  f"F1: {m['f1']:.4f}, Recall (TPR): {m['recall']:.4f}, Precision: {m['precision']:.4f}, " +
                  f"TPR: {m['tpr']:.4f}, FPR: {m['fpr']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(multimodal_model.state_dict(), best_model_path)
            print("  Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"  No improvement in validation loss. Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    multimodal_model.load_state_dict(torch.load(best_model_path))
    print("\nEvaluating on test set...")
    metrics = evaluate_model_metrics(multimodal_model, test_loader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation Metrics on Test Set:")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, " +
              f"F1: {m['f1']:.4f}, Recall (TPR): {m['recall']:.4f}, Precision: {m['precision']:.4f}, " +
              f"TPR: {m['tpr']:.4f}, FPR: {m['fpr']:.4f}")

    if "eddi_stats" in metrics:
        print("\nDetailed EDDI Statistics on Test Set:")
        eddi_stats = metrics["eddi_stats"]
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            print(f"\n{outcome.capitalize()} EDDI Stats:")
            stats = eddi_stats[outcome]
            print("  Age subgroup EDDI      :", stats["age_subgroup_eddi"])
            print("  Aggregated Age EDDI    : {:.4f}".format(stats["age_eddi"]))
            print("  Ethnicity subgroup EDDI:", stats["ethnicity_subgroup_eddi"])
            print("  Aggregated Ethnicity EDDI: {:.4f}".format(stats["ethnicity_eddi"]))
            print("  Insurance subgroup EDDI:", stats["insurance_subgroup_eddi"])
            print("  Aggregated Insurance EDDI: {:.4f}".format(stats["insurance_eddi"]))
            print("  Final Overall {} EDDI: {:.4f}".format(outcome.capitalize(), stats["final_EDDI"]))

    print("Training complete.")

if __name__ == "__main__":
    params = {
        'Xtrain': X_df.iloc[train_idx].reset_index(drop=True),
        'ytrain': y_df.iloc[train_idx].reset_index(drop=True),
        'Xvalid': X_df.iloc[val_idx].reset_index(drop=True),
        'yvalid': y_df.iloc[val_idx].reset_index(drop=True),
        'ztrain': z_df.iloc[train_idx].reset_index(drop=True),
        'zvalid': z_df.iloc[val_idx].reset_index(drop=True),
        'method': 'adv',  # adversarial debiasing
        'num_classes': 2,
        'hyperparameters': {
            'learning_rate': [1e-4, 5e-5],
            'num_iters': [1000, 2000],
            'num_nodes': [64, 128],
            'num_nodes_adv': [32, 64],
            'dropout_rate': [0.3, 0.5],
            'alpha': [1, 2]
        }
    }
    for d in ['model', 'adv', 'metrics']:
        if not os.path.exists(d):
            os.makedirs(d)
    adv_model = Adv_Model(params)
    adv_model.train()
    adv_model.evaluate()
