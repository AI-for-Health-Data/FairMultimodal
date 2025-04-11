import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, Subset
from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
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
    else:
        return "70-89"

def map_ethnicity(code, ethnicity_mapping):
    group = ethnicity_mapping.get(code, "others")
    group = group.lower()
    if "white" in group:
        return "white"
    elif "black" in group:
        return "black"
    elif "asian" in group:
        return "asian"
    elif "hispanic" in group or "latino" in group:
        return "hispanic"
    else:
        return "others"

def map_insurance(i, mapping=None):
    if mapping is None:
        default_mapping = {0: "Government", 1: "Medicare", 2: "Medicaid", 3: "Private", 4: "Self Pay"}
        return default_mapping.get(i, "Other")
    else:
        return mapping.get(i, "Other")

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

class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, BEHRT, device, hidden_size=512):
        super(MultimodalTransformer, self).__init__()
        self.BEHRT = BEHRT
        self.device = device

        self.ts_linear = nn.Linear(BEHRT.bert.config.hidden_size, 256)
        self.text_linear = nn.Linear(text_embed_size, 256)
        
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
        ts_pre = self.ts_linear(structured_emb)
        text_pre = self.text_linear(aggregated_text_embedding)
        ts_proj = F.relu(ts_pre)
        text_proj = F.relu(text_pre)
        combined_post = torch.cat((ts_proj, text_proj), dim=1)
        logits = self.classifier(combined_post)
        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits = logits[:, 1].unsqueeze(1)
        vent_logits = logits[:, 2].unsqueeze(1)
        fused_embedding_pre_relu = torch.cat((ts_pre, text_pre), dim=1)
        return mortality_logits, los_logits, vent_logits, fused_embedding_pre_relu

def train_step(model, dataloader, optimizer, device, crit_mort, crit_los, crit_vent):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (dummy_input_ids, dummy_attn_mask,
         age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
         gender_ids, ethnicity_ids, insurance_ids,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        mortality_logits, los_logits, vent_logits, _ = model(
            dummy_input_ids, dummy_attn_mask,
            age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
            gender_ids, ethnicity_ids, insurance_ids,
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
            
            mort_logits, los_logits, vent_logits, _ = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            loss_mort = crit_mort(mort_logits, labels_mortality.unsqueeze(1))
            loss_los = crit_los(los_logits, labels_los.unsqueeze(1))
            loss_vent = crit_vent(vent_logits, labels_vent.unsqueeze(1))
            loss = loss_mort + loss_los + loss_vent
            running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model_metrics(model, dataloader, device, threshold=0.5, print_eddi=False,
                           ethnicity_mapping=None, insurance_mapping=None):
    
    def compute_pairwise_gap(metric_list):
        valid_values = [v for v in metric_list if not np.isnan(v)]
        n = len(valid_values)
        if n < 2:
            return np.nan
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(abs(valid_values[i] - valid_values[j]))
        if len(diffs) == 0:
            return np.nan
        return np.mean(diffs)
    
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_vent_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_labels_vent = []
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
            
            mort_logits, los_logits, vent_logits, fused_embedding = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_vent_logits.append(vent_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_vent.append(labels_vent.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_los_logits  = torch.cat(all_los_logits, dim=0)
    all_vent_logits = torch.cat(all_vent_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_los  = torch.cat(all_labels_los, dim=0)
    all_labels_vent = torch.cat(all_labels_vent, dim=0)
    
    mort_probs = torch.sigmoid(all_mort_logits).numpy().squeeze()
    los_probs  = torch.sigmoid(all_los_logits).numpy().squeeze()
    vent_probs = torch.sigmoid(all_vent_logits).numpy().squeeze()
    labels_mort_np = all_labels_mort.numpy().squeeze()
    labels_los_np  = all_labels_los.numpy().squeeze()
    labels_vent_np = all_labels_vent.numpy().squeeze()
    
    tasks = ["mortality", "los", "mechanical_ventilation"]
    metrics = {}
    for task, probs, labels in zip(tasks, [mort_probs, los_probs, vent_probs],
                                    [labels_mort_np, labels_los_np, labels_vent_np]):
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
        
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall_val, "precision": precision_val}
    
    if print_eddi:
        age_order = ["15-29", "30-49", "50-69", "70-89"]
        ethnicity_order = ["white", "black", "hispanic", "asian", "others"]
        eddi_stats = {}
        for task, labels_np, probs in zip(tasks,
                                          [labels_mort_np, labels_los_np, labels_vent_np],
                                          [mort_probs, los_probs, vent_probs]):
            overall_age, age_eddi_sub = compute_eddi(labels_np.astype(int), probs, 
                                                     np.array([get_age_bucket(a) for a in torch.cat(all_age, dim=0).numpy().squeeze()]), 
                                                     threshold)
            overall_eth, eth_eddi_sub = compute_eddi(labels_np.astype(int), probs, 
                                                     np.array([map_ethnicity(e.item(), ethnicity_mapping) for e in torch.cat(all_ethnicity, dim=0)]), 
                                                     threshold)
            overall_ins, ins_eddi_sub = compute_eddi(labels_np.astype(int), probs, 
                                                     np.array([map_insurance(i.item(), insurance_mapping) for i in torch.cat(all_insurance, dim=0)]), 
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
        for task in tasks:
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
            for group in insurance_mapping.values():
                score = eddi["insurance_subgroup_eddi"].get(group, 0)
                print(f"    {group}: {score:.4f}")
            print("  Final Overall {} EDDI: {:.4f}".format(task.capitalize(), eddi["final_EDDI"]))
    
    age_order = ["15-29", "30-49", "50-69", "70-89"]
    ethnicity_order = ["white", "black", "hispanic", "asian", "others"]
    insurance_order = list(insurance_mapping.values()) if insurance_mapping is not None else \
                      ["government", "medicare", "medicaid", "private", "self pay", "other"]
    
    age_subgroups = np.array([get_age_bucket(a) for a in torch.cat(all_age, dim=0).numpy().squeeze()])
    ethnicity_subgroups = np.array([map_ethnicity(e.item(), ethnicity_mapping) for e in torch.cat(all_ethnicity, dim=0)])
    insurance_subgroups = np.array([map_insurance(i.item(), insurance_mapping) for i in torch.cat(all_insurance, dim=0)])
    
    sensitive_attributes = {
        "age": (age_subgroups, age_order),
        "ethnicity": (ethnicity_subgroups, ethnicity_order),
        "insurance": (insurance_subgroups, insurance_order)
    }
    
    fairness_metrics = {task: {"age": {}, "ethnicity": {}, "insurance": {}} for task in tasks}
    for task, probs, labels in zip(tasks,
                                   [mort_probs, los_probs, vent_probs],
                                   [labels_mort_np, labels_los_np, labels_vent_np]):
        preds = (probs > threshold).astype(int)
        for attr in sensitive_attributes:
            subgroup_array, order = sensitive_attributes[attr]
            for group in order:
                mask = (subgroup_array == group)
                if np.sum(mask) == 0:
                    fairness_metrics[task][attr][group] = {"TPR": np.nan, "FPR": np.nan}
                else:
                    group_preds = preds[mask]
                    group_labels = labels[mask]
                    TP = ((group_preds == 1) & (group_labels == 1)).sum()
                    FN = ((group_preds == 0) & (group_labels == 1)).sum()
                    FP = ((group_preds == 1) & (group_labels == 0)).sum()
                    TN = ((group_preds == 0) & (group_labels == 0)).sum()
                    TPR_grp = TP / (TP + FN) if (TP + FN) > 0 else np.nan
                    FPR_grp = FP / (FP + TN) if (FP + TN) > 0 else np.nan
                    fairness_metrics[task][attr][group] = {"TPR": TPR_grp, "FPR": FPR_grp}
    
    aggregated = {}
    for task in tasks:
        aggregated[task] = {}
        overall_attr_aggregates = []
        for attr in sensitive_attributes:
            subgroup_metrics = fairness_metrics[task][attr]
            tpr_values = [subgroup_metrics[g]["TPR"] for g in subgroup_metrics if not np.isnan(subgroup_metrics[g]["TPR"])]
            fpr_values = [subgroup_metrics[g]["FPR"] for g in subgroup_metrics if not np.isnan(subgroup_metrics[g]["FPR"])]
            tpr_gap = compute_pairwise_gap(tpr_values)
            fpr_gap = compute_pairwise_gap(fpr_values)
            attr_eo = np.nan
            if (not np.isnan(tpr_gap)) and (not np.isnan(fpr_gap)):
                attr_eo = (tpr_gap + fpr_gap) / 2.0
            aggregated[task][attr] = {"TPR_gap": tpr_gap, "FPR_gap": fpr_gap, "EO": attr_eo}
            if not np.isnan(attr_eo):
                overall_attr_aggregates.append(attr_eo)
        overall_eo = np.mean(overall_attr_aggregates) if overall_attr_aggregates else np.nan
        aggregated[task]["overall_EO"] = overall_eo
    
    metrics["fairness_metrics"] = {
        "subgroup_metrics": fairness_metrics,
        "aggregated_metrics": aggregated
    }
    
    print("\n--- Aggregated Fairness Metrics (Mean Pairwise Gaps) ---")
    for task in tasks:
        print(f"\nTask: {task.capitalize()}")
        for attr in sensitive_attributes:
            agg = aggregated[task][attr]
            print(f"  {attr.capitalize()} - TPR Gap: {agg['TPR_gap']:.4f}, FPR Gap: {agg['FPR_gap']:.4f}, EO: {agg['EO']:.4f}")
        print(f"  Overall EO for task {task.capitalize()}: {aggregated[task]['overall_EO']:.4f}")
    
    return metrics

def extract_fused_embeddings(model, dataloader, device, ethnicity_mapping, insurance_mapping):
    fused_embeddings_list = []
    outcome_labels_list = []
    sensitive_attributes_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
            
            mortality_logits, los_logits, vent_logits, fused_embedding = model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            fused_embeddings_list.append(fused_embedding.cpu())
            outcomes = torch.cat([labels_mortality.unsqueeze(1),
                                  labels_los.unsqueeze(1),
                                  labels_vent.unsqueeze(1)], dim=1)
            outcome_labels_list.append(outcomes.cpu())
            age_labels = [get_age_bucket(a.item()) for a in age_ids]
            ethnicity_labels = [map_ethnicity(e.item(), ethnicity_mapping) for e in ethnicity_ids]
            insurance_labels = [map_insurance(i.item(), insurance_mapping) for i in insurance_ids]
            combined_sensitive = list(zip(age_labels, ethnicity_labels, insurance_labels))
            sensitive_attributes_list.extend(combined_sensitive)
    
    fused_embeddings = torch.cat(fused_embeddings_list, dim=0)
    outcome_labels_all = torch.cat(outcome_labels_list, dim=0)
    return fused_embeddings, outcome_labels_all, sensitive_attributes_list


def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    df_unique = df_filtered.groupby("subject_id", as_index=False).first().copy()
    print("Number of unique patients:", len(df_unique))
    
    if "segment" not in df_unique.columns:
        df_unique["segment"] = 0

    sensitive_df = df_unique[['age', 'ethnicity', 'insurance']].copy()
    sensitive_df["ethnicity"] = sensitive_df["ethnicity"].astype(str)
    sensitive_df["insurance"] = sensitive_df["insurance"].astype(str)
    sensitive_df["ethnicity_code"] = sensitive_df["ethnicity"].astype("category").cat.codes
    sensitive_df["insurance_code"] = sensitive_df["insurance"].astype("category").cat.codes
    df_unique["ethnicity"] = sensitive_df["ethnicity_code"]
    df_unique["insurance"] = sensitive_df["insurance_code"]

    ethnicity_mapping = dict(enumerate(sensitive_df["ethnicity"].astype("category").cat.categories))
    insurance_mapping = dict(enumerate(sensitive_df["insurance"].astype("category").cat.categories))

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
    if df_unique["gender"].dtype == "object":
        df_unique["gender"] = df_unique["gender"].astype("category").cat.codes

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
    NUM_ETHNICITIES = sensitive_df["ethnicity_code"].nunique()
    NUM_INSURANCES = sensitive_df["insurance_code"].nunique()

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

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    best_model_path = "best_multimodal_model.pt"

    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                  criterion_mortality, criterion_los, criterion_mech)
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model_loss(multimodal_model, val_loader, device,
                                       criterion_mortality, criterion_los, criterion_mech)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        val_metrics = evaluate_model_metrics(multimodal_model, val_loader, device, threshold=0.5,
                                             ethnicity_mapping=ethnicity_mapping, insurance_mapping=insurance_mapping)
        print(f"--- Validation Metrics at Epoch {epoch+1} ---")
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            m = val_metrics[outcome]
            print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
                  f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")
    
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
    test_metrics = evaluate_model_metrics(multimodal_model, test_loader, device, threshold=0.5, print_eddi=True,
                                          ethnicity_mapping=ethnicity_mapping, insurance_mapping=insurance_mapping)
    print("\nFinal Evaluation Metrics on Test Set:")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = test_metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
              f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")
    
    fused_embeddings, outcome_labels, sensitive_attributes = extract_fused_embeddings(multimodal_model, test_loader, device,
                                                                                      ethnicity_mapping, insurance_mapping)
    print("\nExtracted Fused Embeddings (Pre-ReLU):")
    print("Shape of Fused Embeddings:", fused_embeddings.shape)
    print("Shape of Outcome Labels:", outcome_labels.shape)
    print("First 10 sensitive attributes entries:")
    for entry in sensitive_attributes[:10]:
        print(entry)
    
    fused_embeddings_np = fused_embeddings.numpy()
    outcome_labels_np = outcome_labels.numpy()
    
    np.savez("extracted_embeddings.npz",
             fused_embeddings=fused_embeddings_np,
             outcome_labels=outcome_labels_np,
             sensitive_attributes=sensitive_attributes)
    print("Extracted embeddings saved to 'extracted_embeddings.npz'")
    
    print("Training complete.")
    
if __name__ == "__main__":
    train_pipeline()
