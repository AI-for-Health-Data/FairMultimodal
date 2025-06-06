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
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from skmultilearn.model_selection import iterative_train_test_split

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

def map_ethnicity(val):
    if isinstance(val, str):
        val = val.lower().strip()
        allowed = ["white", "black", "asian", "hispanic"]
        return val if val in allowed else "other"
    mapping = {0: "white", 1: "black", 2: "asian", 3: "hispanic"}
    return mapping.get(val, "other")

def map_insurance(val):
    if isinstance(val, str):
        val = val.lower().strip()
        allowed = ["government", "medicare", "medicaid", "private", "self pay"]
        return val if val in allowed else "other"
    mapping = {0: "government", 1: "medicare", 2: "medicaid", 3: "private", 4: "self pay"}
    return mapping.get(val, "other")

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

# DfC Structured Branch (BEHRT without Demographics)
class BEHRTModel_DfC(nn.Module):
    def __init__(self, num_diseases, num_segments, num_admission_locs, num_discharge_locs, hidden_size=768):
        super(BEHRTModel_DfC, self).__init__()
        vocab_size = num_diseases + num_segments + num_admission_locs + num_discharge_locs + 1  # +1 for special token
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(config)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.admission_loc_embedding = nn.Embedding(num_admission_locs, hidden_size)
        self.discharge_loc_embedding = nn.Embedding(num_discharge_locs, hidden_size)

    def forward(self, input_ids, attention_mask, segment_ids, adm_loc_ids, disch_loc_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        seg_embeds = self.segment_embedding(segment_ids)
        adm_embeds = self.admission_loc_embedding(adm_loc_ids)
        disch_embeds = self.discharge_loc_embedding(disch_loc_ids)
        extra = (seg_embeds + adm_embeds + disch_embeds) / 3.0
        cls_embedding = cls_token + extra
        return cls_embedding

# Multimodal Transformer (DfC Version - Average Fusion)
class MultimodalTransformer_DfC(nn.Module):
    def __init__(self, text_embed_size, BEHRT, device, hidden_size=512):
        super(MultimodalTransformer_DfC, self).__init__()
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
                segment_ids, adm_loc_ids, disch_loc_ids,
                aggregated_text_embedding):
        structured_emb = self.BEHRT(dummy_input_ids, dummy_attn_mask,
                                      segment_ids, adm_loc_ids, disch_loc_ids)
        ts_proj = self.ts_projector(structured_emb)
        text_proj = self.text_projector(aggregated_text_embedding)
        combined = torch.cat((ts_proj, text_proj), dim=1)
        logits = self.classifier(combined)
        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits = logits[:, 1].unsqueeze(1)
        vent_logits = logits[:, 2].unsqueeze(1)
        return mortality_logits, los_logits, vent_logits

# Custom Dataset to Preserve Sensitive Info as Strings
class CustomDataset(Dataset):
    def __init__(self, dummy_input_ids, dummy_attn_mask, segment_ids, adm_loc_ids, disch_loc_ids,
                 aggregated_text_embeddings, labels_mortality, labels_los, labels_vent,
                 age_ids, ethnicity_list, insurance_list):
        self.dummy_input_ids = dummy_input_ids
        self.dummy_attn_mask = dummy_attn_mask
        self.segment_ids = segment_ids
        self.adm_loc_ids = adm_loc_ids
        self.disch_loc_ids = disch_loc_ids
        self.aggregated_text_embeddings = aggregated_text_embeddings
        self.labels_mortality = labels_mortality
        self.labels_los = labels_los
        self.labels_vent = labels_vent
        self.age_ids = age_ids
        self.ethnicity_list = ethnicity_list  
        self.insurance_list = insurance_list    
        self.length = dummy_input_ids.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.dummy_input_ids[idx],
                self.dummy_attn_mask[idx],
                self.segment_ids[idx],
                self.adm_loc_ids[idx],
                self.disch_loc_ids[idx],
                self.aggregated_text_embeddings[idx],
                self.labels_mortality[idx],
                self.labels_los[idx],
                self.labels_vent[idx],
                self.age_ids[idx],
                self.ethnicity_list[idx],
                self.insurance_list[idx])


def calculate_fairness_metrics(labels, predictions, demographics, sensitive_class):
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
    unique_classes = np.unique(demographics)
    metrics = {}
    for sensitive_class in unique_classes:
        sensitive_indices = demographics == sensitive_class
        non_sensitive_indices = ~sensitive_indices

        tn_s, fp_s, fn_s, tp_s = confusion_matrix(labels[sensitive_indices], predictions[sensitive_indices]).ravel()
        tn_ns, fp_ns, fn_ns, tp_ns = confusion_matrix(labels[non_sensitive_indices], predictions[non_sensitive_indices]).ravel()

        tpr_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) != 0 else 0
        fpr_s = fp_s / (fp_s + tn_s) if (fp_s + tn_s) != 0 else 0

        eod = tpr_s - (tp_ns / (tp_ns + fn_ns) if (tp_ns + fn_ns) != 0 else 0)
        avg_eod = (abs(fp_s / (fp_s + tn_s) - (fp_ns / (fp_ns + tn_ns)) if (fp_s+tn_s)!=0 and (fp_ns+tn_ns)!=0 else 0) + abs(eod)) / 2

        metrics[sensitive_class] = {
            "TPR": tpr_s,
            "FPR": fpr_s,
            "Equal Opportunity Difference": eod,
            "Average Equalized Odds Difference": avg_eod
        }
    return metrics

def calculate_predictive_parity(y_true, y_pred, sensitive_attrs):
    unique_groups = np.unique(sensitive_attrs)
    precision_scores = {}
    for group in unique_groups:
        group_indices = sensitive_attrs == group
        precision = precision_score(y_true[group_indices], y_pred[group_indices], zero_division=0, average='weighted')
        precision_scores[group] = precision
    return precision_scores

def calculate_tpr_and_fpr(y_true, y_pred, group_mask):
    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask], labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

def calculate_sd_for_rates(y_true, y_pred, sensitive_attr):
    unique_classes = np.unique(sensitive_attr)
    tpr_values = []
    fpr_values = []
    for group in unique_classes:
        group_mask = sensitive_attr == group
        tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred, group_mask)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    sd_tpr = np.std(tpr_values, ddof=1)
    sd_fpr = np.std(fpr_values, ddof=1)
    return sd_tpr, sd_fpr

def calculate_equalized_odds_difference(y_true, y_pred, sensitive_attr):
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
    avg_tpr_diff = np.mean(tpr_diffs) if tpr_diffs else 0
    avg_fpr_diff = np.mean(fpr_diffs) if fpr_diffs else 0
    return avg_tpr_diff, avg_fpr_diff


def train_step(model, dataloader, optimizer, device, crit_mort, crit_los, crit_vent):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        batch_moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        (dummy_input_ids, dummy_attn_mask,
         segment_ids, adm_loc_ids, disch_loc_ids,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_vent,
         _, _, _) = batch_moved  # sensitive strings are ignored during training
        optimizer.zero_grad()
        mortality_logits, los_logits, vent_logits = model(
            dummy_input_ids, dummy_attn_mask,
            segment_ids, adm_loc_ids, disch_loc_ids,
            aggregated_text_embedding
        )
        loss_mort = crit_mort(mortality_logits, labels_mortality.unsqueeze(1))
        loss_los = crit_los(los_logits, labels_los.unsqueeze(1))
        loss_vent = crit_vent(vent_logits, labels_vent.unsqueeze(1))
        loss = loss_mort + loss_los + loss_vent
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def validation_step(model, dataloader, device, crit_mort, crit_los, crit_vent):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            (dummy_input_ids, dummy_attn_mask,
             segment_ids, adm_loc_ids, disch_loc_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent,
             _, _, _) = batch_moved
            mortality_logits, los_logits, vent_logits = model(
                dummy_input_ids, dummy_attn_mask,
                segment_ids, adm_loc_ids, disch_loc_ids,
                aggregated_text_embedding
            )
            loss_mort = crit_mort(mortality_logits, labels_mortality.unsqueeze(1))
            loss_los = crit_los(los_logits, labels_los.unsqueeze(1))
            loss_vent = crit_vent(vent_logits, labels_vent.unsqueeze(1))
            loss = loss_mort + loss_los + loss_vent
            running_loss += loss.item()
    return running_loss / len(dataloader)

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
            batch_moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            (dummy_input_ids, dummy_attn_mask,
             segment_ids, adm_loc_ids, disch_loc_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent,
             age_ids, ethnicity_ids, insurance_ids) = batch_moved
            
            mort_logits, los_logits, mech_logits = model(
                dummy_input_ids, dummy_attn_mask,
                segment_ids, adm_loc_ids, disch_loc_ids,
                aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_mech_logits.append(mech_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_mech.append(labels_vent.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.extend(ethnicity_ids)
            all_insurance.extend(insurance_ids)
    
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
        tpr = recall_score(labels, preds, zero_division=0)
        precision_val = precision_score(labels, preds, zero_division=0)
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "tpr": tpr, "precision": precision_val, "fpr": fpr}
    
    ages = torch.cat(all_age, dim=0).numpy().squeeze()
    age_groups = np.array([get_age_bucket(a) for a in ages])
    ethnicity_groups = np.array([map_ethnicity(e) for e in all_ethnicity])
    insurance_groups = np.array([map_insurance(i) for i in all_insurance])
    
    age_order = ["15-29", "30-49", "50-69", "70-89", "Other"]
    ethnicity_order = ["white", "black", "asian", "hispanic", "other"]
    insurance_order = ["government", "medicare", "medicaid", "private", "self pay", "other"]
    
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

    if print_eddi:
        print("\n--- EDDI Calculation for Each Outcome ---")
        for task in ["mortality", "los", "mechanical_ventilation"]:
            print(f"\nTask: {task.capitalize()}")
            eddi = eddi_stats[task]
            print("  Aggregated Age EDDI    : {:.4f}".format(eddi["age_eddi"]))
            print("  Age Subgroup EDDI:")
            for bucket in age_order:
                score = eddi["age_subgroup_eddi"].get(bucket, 0)
                print(f"    {bucket}: {score:.4f}")
            print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi["ethnicity_eddi"]))
            print("  Ethnicity Subgroup EDDI:")
            for group in ethnicity_order:
                score = eddi["ethnicity_subgroup_eddi"].get(group, 0)
                print(f"    {group}: {score:.4f}")
            print("  Aggregated Insurance EDDI: {:.4f}".format(eddi["insurance_eddi"]))
            print("  Insurance Subgroup EDDI:")
            for group in insurance_order:
                score = eddi["insurance_subgroup_eddi"].get(group, 0)
                print(f"    {group}: {score:.4f}")
            print("  Final Overall {} EDDI: {:.4f}".format(task.capitalize(), eddi["final_EDDI"]))
    
    print("\n--- Fairness Metrics by Sensitive Attribute ---")
    for task, probs, labels in zip(["mortality", "los", "mechanical_ventilation"],
                                   [mort_probs, los_probs, mech_probs],
                                   [labels_mort_np, labels_los_np, labels_mech_np]):
        predictions = (probs > threshold).astype(int)
        print(f"\nOutcome: {task.capitalize()}")
        for attr_name, groups in zip(["Age", "Ethnicity", "Insurance"],
                                     [age_groups, ethnicity_groups, insurance_groups]):
            print(f"\nFairness metrics for sensitive attribute: {attr_name}")
            fm = calculate_multiclass_fairness_metrics(labels, predictions, groups)
            for group, metric_vals in fm.items():
                print(f" Group {group}: TPR: {metric_vals['TPR']:.3f}, FPR: {metric_vals['FPR']:.3f}, "
                      f"Equal Opportunity Diff: {metric_vals['Equal Opportunity Difference']:.3f}, "
                      f"Avg EO Diff: {metric_vals['Average Equalized Odds Difference']:.3f}")
            avg_tpr_diff, avg_fpr_diff = calculate_equalized_odds_difference(labels, predictions, groups)
            print(f" Overall Avg TPR Difference: {avg_tpr_diff:.3f}")
            print(f" Overall Avg FPR Difference: {avg_fpr_diff:.3f}")
        combined_metric = 0
        for groups in [age_groups, ethnicity_groups, insurance_groups]:
            avg_tpr, avg_fpr = calculate_equalized_odds_difference(labels, predictions, groups)
            combined_metric += (avg_tpr + avg_fpr) / 2
        combined_metric /= 3
        print(f"\nCombined Equalized Odds Metric (averaged over sensitive attributes): {combined_metric:.3f}")
    
    return metrics

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    keep_cols = {"subject_id", "hadm_id", "short_term_mortality", "los_binary", 
                 "mechanical_ventilation", "age", "first_wardid", "last_wardid", 
                 "ethnicity", "insurance", "gender"}
    
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
    if "ethnicity" not in merged_df.columns:
        if "ethnicity_struct" in merged_df.columns:
            merged_df.rename(columns={"ethnicity_struct": "ethnicity"}, inplace=True)
        else:
            raise ValueError("No ethnicity column found. Please provide ethnicity data.")
    if "insurance" not in merged_df.columns:
        if "insurance_struct" in merged_df.columns:
            merged_df.rename(columns={"insurance_struct": "insurance"}, inplace=True)
        else:
            raise ValueError("No insurance column found. Please provide insurance data.")
    for col, default in zip(["age", "gender"], [0, 0]):
        if col not in merged_df.columns:
            print(f"Column '{col}' not found; creating default '{default}' values.")
            merged_df[col] = default

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
            print(f"Column '{col}' not found in filtered dataframe; creating default values.")
            df_filtered[col] = 0

    df_unique = df_filtered.groupby("subject_id", as_index=False).first()
    print("Number of unique patients before sampling:", len(df_unique))
    print("Number of unique patients after sampling:", len(df_unique))
    if "segment" not in df_unique.columns:
        df_unique["segment"] = 0

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_unique, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    num_samples = len(df_unique)
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    segment_ids = torch.tensor(df_unique["segment"].values, dtype=torch.long)
    admission_loc_ids = torch.tensor(df_unique["first_wardid"].values, dtype=torch.long)
    discharge_loc_ids = torch.tensor(df_unique["last_wardid"].values, dtype=torch.long)
    labels_mortality = torch.tensor(df_unique["short_term_mortality"].values, dtype=torch.float32)
    labels_los = torch.tensor(df_unique["los_binary"].values, dtype=torch.float32)
    labels_vent = torch.tensor(df_unique["mechanical_ventilation"].values, dtype=torch.float32)
    age_ids = torch.tensor(df_unique["age"].values, dtype=torch.long)
    ethnicity_list = df_unique["ethnicity"].astype(str).tolist()
    insurance_list = df_unique["insurance"].astype(str).tolist()

    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')
    
    dataset = CustomDataset(
        dummy_input_ids, dummy_attn_mask,
        segment_ids, admission_loc_ids, discharge_loc_ids,
        aggregated_text_embeddings_t,
        labels_mortality, labels_los, labels_vent,
        age_ids, ethnicity_list, insurance_list
    )
    
    total_samples = len(dataset)
    Y = np.vstack([
        labels_mortality.numpy(),
        labels_los.numpy(),
        labels_vent.numpy()
    ]).T
    X = np.arange(total_samples).reshape(-1, 1)

    X_train_val, Y_train_val, X_test, Y_test = iterative_train_test_split(X, Y, test_size=0.2)
    val_fraction = 0.05
    X_train, Y_train, X_val, Y_val = iterative_train_test_split(X_train_val, Y_train_val, test_size=val_fraction)

    train_indices = X_train.flatten().tolist()
    val_indices = X_val.flatten().tolist()
    test_indices  = X_test.flatten().tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    disease_mapping = {d: i for i, d in enumerate(df_unique["hadm_id"].unique())}
    NUM_DISEASES = len(disease_mapping)
    NUM_SEGMENTS = df_unique["segment"].nunique() if df_unique["segment"].nunique() > 0 else 1
    NUM_ADMISSION_LOCS = df_unique["first_wardid"].nunique()
    NUM_DISCHARGE_LOCS = df_unique["last_wardid"].nunique()

    print("\n--- Hyperparameters based on processed data (DfC) ---")
    print("NUM_DISEASES:", NUM_DISEASES)
    print("NUM_SEGMENTS:", NUM_SEGMENTS)
    print("NUM_ADMISSION_LOCS:", NUM_ADMISSION_LOCS)
    print("NUM_DISCHARGE_LOCS:", NUM_DISCHARGE_LOCS)

    behrt_model_dfc = BEHRTModel_DfC(
        num_diseases=NUM_DISEASES,
        num_segments=NUM_SEGMENTS,
        num_admission_locs=NUM_ADMISSION_LOCS,
        num_discharge_locs=NUM_DISCHARGE_LOCS,
        hidden_size=768
    ).to(device)

    multimodal_model = MultimodalTransformer_DfC(
        text_embed_size=768,
        BEHRT=behrt_model_dfc,
        device=device,
        hidden_size=512
    ).to(device)

    
    optimizer = AdamW(multimodal_model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs = 20
    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                  criterion_mortality, criterion_los, criterion_mech)
        train_loss = running_loss / len(train_loader)
        
        val_loss = validation_step(multimodal_model, val_loader, device, criterion_mortality, criterion_los, criterion_mech)
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(multimodal_model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered. No improvement in validation loss for 5 consecutive epochs.")
                break

    state_dict = torch.load("best_model.pt", map_location=device)
    model_dict = multimodal_model.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("bert.") or key.startswith("segment_embedding") or key.startswith("admission_loc_embedding") or key.startswith("discharge_loc_embedding"):
            new_key = "BEHRT." + key
        else:
            new_key = key
        if new_key in model_dict:
            new_state_dict[new_key] = value
        else:
            print("Skipping key:", key)
    missing_keys, unexpected_keys = multimodal_model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys after loading:", missing_keys)
    print("Unexpected keys after loading:", unexpected_keys)
    
    metrics = evaluate_model(multimodal_model, test_loader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation Metrics on Test Set (DfC):")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
              f"F1: {m['f1']:.4f}, TPR: {m['tpr']:.4f}, Precision: {m['precision']:.4f}, FPR: {m['fpr']:.4f}")
    
    print("\nDetailed EDDI Statistics:")
    eddi_stats = metrics["eddi_stats"]
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        print(f"\n{outcome.capitalize()} EDDI Stats:")
        stats = eddi_stats[outcome]
        print("  Age Subgroup EDDI      :", stats["age_subgroup_eddi"])
        print("  Aggregated Age EDDI    : {:.4f}".format(stats["age_eddi"]))
        print("  Ethnicity Subgroup EDDI:", stats["ethnicity_subgroup_eddi"])
        print("  Aggregated Ethnicity EDDI: {:.4f}".format(stats["ethnicity_eddi"]))
        print("  Insurance Subgroup EDDI:", stats["insurance_subgroup_eddi"])
        print("  Aggregated Insurance EDDI: {:.4f}".format(stats["insurance_eddi"]))
        print("  Final Overall {} EDDI: {:.4f}".format(outcome.capitalize(), stats["final_EDDI"]))

    print("Training complete for the DfC pipeline.")

if __name__ == "__main__":
    train_pipeline()
