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
from sklearn.metrics import confusion_matrix, precision_score
from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             recall_score, precision_score, confusion_matrix)
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
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

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

def calculate_tpr_and_fpr(y_true, y_pred, group_mask):
    if np.sum(group_mask) == 0:
        return 0, 0
    
    y_true_group = y_true[group_mask]
    y_pred_group = y_pred[group_mask]
    
    if len(y_true_group) == 0:
        return 0, 0
    
    print(f"Group size: {len(y_true_group)}, Positive samples: {np.sum(y_true_group)}")
    
    cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    
    tpr = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    
    return tpr, fpr

def calculate_equalized_odds_difference(y_true, y_pred, sensitive_attr):
    unique_classes = np.unique(sensitive_attr)
    if len(unique_classes) <= 1:
        return 0.0, 0.0
    
    tpr_diffs = []
    fpr_diffs = []
    
    print(f"Calculating EO differences for {len(unique_classes)} unique groups")
    
    tpr_values = {}
    fpr_values = {}
    
    for group in unique_classes:
        group_mask = sensitive_attr == group
        group_size = np.sum(group_mask)
        if group_size == 0:
            continue
            
        tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred, group_mask)
        tpr_values[group] = tpr
        fpr_values[group] = fpr
        print(f"Group {group}: TPR={tpr:.4f}, FPR={fpr:.4f}, size={group_size}")
    
    for i, group1 in enumerate(unique_classes):
        if group1 not in tpr_values:
            continue
        for group2 in unique_classes[i+1:]:
            if group2 not in tpr_values:
                continue
            tpr1, fpr1 = tpr_values[group1], fpr_values[group1]
            tpr2, fpr2 = tpr_values[group2], fpr_values[group2]
            
            tpr_diff = abs(tpr1 - tpr2)
            fpr_diff = abs(fpr1 - fpr2)
            
            tpr_diffs.append(tpr_diff)
            fpr_diffs.append(fpr_diff)
            
            print(f"Group {group1} vs {group2}: TPR diff={tpr_diff:.4f}, FPR diff={fpr_diff:.4f}")
    
    avg_tpr_diff = np.mean(tpr_diffs) if len(tpr_diffs) > 0 else 0.0
    avg_fpr_diff = np.mean(fpr_diffs) if len(fpr_diffs) > 0 else 0.0
    
    print(f"Average TPR diff: {avg_tpr_diff:.4f}, Average FPR diff: {avg_fpr_diff:.4f}")
    
    return avg_tpr_diff, avg_fpr_diff

def calculate_multiclass_fairness_metrics(y_true, y_pred, demographics):
    unique_classes = np.unique(demographics)
    metrics = {}

    for sensitive_class in unique_classes:
        sensitive_indices = demographics == sensitive_class
        non_sensitive_indices = ~sensitive_indices

        cm_sensitive = confusion_matrix(y_true[sensitive_indices], y_pred[sensitive_indices], labels=[0,1])
        cm_non_sensitive = confusion_matrix(y_true[non_sensitive_indices], y_pred[non_sensitive_indices], labels=[0,1])
        if cm_sensitive.size == 4:
            tn_s, fp_s, fn_s, tp_s = cm_sensitive.ravel()
        else:
            tn_s = fp_s = fn_s = tp_s = 0
        if cm_non_sensitive.size == 4:
            tn_ns, fp_ns, fn_ns, tp_ns = cm_non_sensitive.ravel()
        else:
            tn_ns = fp_ns = fn_ns = tp_ns = 0

        tpr_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) != 0 else 0
        fpr_s = fp_s / (fp_s + tn_s) if (fp_s + tn_s) != 0 else 0

        tpr_ns = tp_ns / (tp_ns + fn_ns) if (tp_ns + fn_ns) != 0 else 0
        eod = tpr_s - tpr_ns
        
        fpr_ns = fp_ns / (fp_ns + tn_ns) if (fp_ns + tn_ns) != 0 else 0
        eod_fpr = fpr_s - fpr_ns
        avg_eod = (abs(eod) + abs(eod_fpr)) / 2

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
        precision = precision_score(y_true[group_indices], y_pred[group_indices], zero_division=0)
        precision_scores[group] = precision
    return precision_scores

# BioClinicalBERT Fine-Tuning
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

# Multimodal Transformer with Dynamic EDDI Weighting
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device, beta=0.3):
        super(MultimodalTransformer, self).__init__()
        self.beta = beta
        self.device = device
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab

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

        # Outcome-specific classifiers.
        self.classifier_demo_mort = nn.Linear(256, 1)
        self.classifier_lab_mort  = nn.Linear(256, 1)
        self.classifier_text_mort = nn.Linear(256, 1)

        self.classifier_demo_los = nn.Linear(256, 1)
        self.classifier_lab_los  = nn.Linear(256, 1)
        self.classifier_text_los = nn.Linear(256, 1)

        self.classifier_demo_mv = nn.Linear(256, 1)
        self.classifier_lab_mv  = nn.Linear(256, 1)
        self.classifier_text_mv = nn.Linear(256, 1)

    def compute_weighted_logit(self, demo_proj, lab_proj, text_proj,
                               classifier_demo, classifier_lab, classifier_text, beta,
                               y_true, sensitive_labels, old_weights=None):
        raw_logit_demo = classifier_demo(demo_proj)
        raw_logit_lab  = classifier_lab(lab_proj)
        raw_logit_text = classifier_text(text_proj)

        eddi_logit_demo = raw_logit_demo.detach()
        eddi_logit_lab  = raw_logit_lab.detach()
        eddi_logit_text = raw_logit_text.detach()

        demo_prob = torch.sigmoid(eddi_logit_demo)
        lab_prob = torch.sigmoid(eddi_logit_lab)
        text_prob = torch.sigmoid(eddi_logit_text)

        if (y_true is not None) and (sensitive_labels is not None):
            if torch.is_tensor(y_true):
                y_true_np = y_true.cpu().numpy()
            else:
                y_true_np = y_true
            demo_prob_np = demo_prob.cpu().numpy().squeeze()
            lab_prob_np = lab_prob.cpu().numpy().squeeze()
            text_prob_np = text_prob.cpu().numpy().squeeze()
            eddi_demo_val, subgroup_demo = compute_eddi(y_true_np, demo_prob_np, sensitive_labels, threshold=0.5)
            eddi_lab_val, subgroup_lab = compute_eddi(y_true_np, lab_prob_np, sensitive_labels, threshold=0.5)
            eddi_text_val, subgroup_text = compute_eddi(y_true_np, text_prob_np, sensitive_labels, threshold=0.5)
            print(f"Computed EDDI - Demo: {eddi_demo_val:.4f}, Lab: {eddi_lab_val:.4f}, Text: {eddi_text_val:.4f}")
        else:
            eddi_demo_val = eddi_lab_val = eddi_text_val = 0.0
            subgroup_demo = subgroup_lab = subgroup_text = {}
            print("No y_true or sensitive_labels provided, setting EDDI values to 0.")

        eddi_max_val = max(eddi_demo_val, eddi_lab_val, eddi_text_val)

        if old_weights is not None:
            old_weight_demo, old_weight_lab, old_weight_text = old_weights
            weight_demo = old_weight_demo + beta * (eddi_max_val - eddi_demo_val)
            weight_lab = old_weight_lab + beta * (eddi_max_val - eddi_lab_val)
            weight_text = old_weight_text + beta * (eddi_max_val - eddi_text_val)
        else:
            weight_demo = 0.33 + beta * (eddi_max_val - eddi_demo_val)
            weight_lab = 0.33 + beta * (eddi_max_val - eddi_lab_val)
            weight_text = 0.33 + beta * (eddi_max_val - eddi_text_val)

        print(f"Modality weights - Demo: {weight_demo:.4f}, Lab: {weight_lab:.4f}, Text: {weight_text:.4f}")

        fused_logit = raw_logit_demo * weight_demo + raw_logit_lab * weight_lab + raw_logit_text * weight_text

        details = {
            "eddi": (eddi_demo_val, eddi_lab_val, eddi_text_val, eddi_max_val),
            "weights": (weight_demo, weight_lab, weight_text),
            "probs": (demo_prob, lab_prob, text_prob),
            "subgroups": (subgroup_demo, subgroup_lab, subgroup_text)
        }
        return fused_logit, details

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=None,
                y_true_dict=None, sensitive_labels_dict=None, old_eddi_weights=None):
        if beta is None:
            beta = self.beta

        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        demo_proj = self.demo_projector(demo_embedding)
        lab_proj = self.lab_projector(lab_embedding)
        text_proj = self.text_projector(text_embedding)

        mort_old_weights = old_eddi_weights.get("mortality") if old_eddi_weights is not None else None
        los_old_weights = old_eddi_weights.get("los") if old_eddi_weights is not None else None
        mv_old_weights = old_eddi_weights.get("mechanical_ventilation") if old_eddi_weights is not None else None

        mort_y_true = y_true_dict["mortality"] if (y_true_dict is not None and "mortality" in y_true_dict) else None
        mort_sensitive = sensitive_labels_dict["mortality"] if (sensitive_labels_dict is not None and "mortality" in sensitive_labels_dict) else None

        los_y_true = y_true_dict["los"] if (y_true_dict is not None and "los" in y_true_dict) else None
        los_sensitive = sensitive_labels_dict["los"] if (sensitive_labels_dict is not None and "los" in sensitive_labels_dict) else None

        mv_y_true = y_true_dict["mechanical_ventilation"] if (y_true_dict is not None and "mechanical_ventilation" in y_true_dict) else None
        mv_sensitive = sensitive_labels_dict["mechanical_ventilation"] if (sensitive_labels_dict is not None and "mechanical_ventilation" in sensitive_labels_dict) else None

        mort_logit, mort_details = self.compute_weighted_logit(
            demo_proj, lab_proj, text_proj,
            self.classifier_demo_mort, self.classifier_lab_mort, self.classifier_text_mort,
            beta, mort_y_true, mort_sensitive, old_weights=mort_old_weights
        )
        los_logit, los_details = self.compute_weighted_logit(
            demo_proj, lab_proj, text_proj,
            self.classifier_demo_los, self.classifier_lab_los, self.classifier_text_los,
            beta, los_y_true, los_sensitive, old_weights=los_old_weights
        )
        mv_logit, mv_details = self.compute_weighted_logit(
            demo_proj, lab_proj, text_proj,
            self.classifier_demo_mv, self.classifier_lab_mv, self.classifier_text_mv,
            beta, mv_y_true, mv_sensitive, old_weights=mv_old_weights
        )

        eddi_details = {"mortality": mort_details,
                        "los": los_details,
                        "mechanical_ventilation": mv_details}
        return mort_logit, los_logit, mv_logit, eddi_details

def train_step(model, dataloader, optimizer, device, beta=0.3, loss_gamma=1.0, target=1.0, old_eddi_weights=None):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features, aggregated_text_embedding,
         labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]

        optimizer.zero_grad()

        y_true_dict = {
            "mortality": labels_mortality.cpu().numpy(),
            "los": labels_los.cpu().numpy(),
            "mechanical_ventilation": labels_mechvent.cpu().numpy()
        }
        sensitive_labels_dict = {
            "mortality": gender_ids.cpu().numpy(),
            "los": gender_ids.cpu().numpy(),
            "mechanical_ventilation": gender_ids.cpu().numpy()
        }

        mort_logit, los_logit, mv_logit, _ = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding, beta=beta,
            y_true_dict=y_true_dict, sensitive_labels_dict=sensitive_labels_dict,
            old_eddi_weights=old_eddi_weights
        )
        loss_mort = criterion_mortality(mort_logit, labels_mortality.unsqueeze(1))
        loss_los = criterion_los(los_logit, labels_los.unsqueeze(1))
        loss_mv = criterion_mech(mv_logit, labels_mechvent.unsqueeze(1))
        eddi_loss = ((mort_logit - target) ** 2).mean()
        loss = loss_mort + loss_los + loss_mv + loss_gamma * eddi_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def validate_step(model, dataloader, device, beta=0.3, loss_gamma=1.0, target=1.0, old_eddi_weights=None):
    model.eval()
    running_loss = 0.0
    last_eddi_details = None
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding,
             labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]

            y_true_dict = {
                "mortality": labels_mortality.cpu().numpy(),
                "los": labels_los.cpu().numpy(),
                "mechanical_ventilation": labels_mechvent.cpu().numpy()
            }
            sensitive_labels_dict = {
                "mortality": gender_ids.cpu().numpy(),
                "los": gender_ids.cpu().numpy(),
                "mechanical_ventilation": gender_ids.cpu().numpy()
            }

            mort_logit, los_logit, mv_logit, eddi_details = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=beta,
                y_true_dict=y_true_dict, sensitive_labels_dict=sensitive_labels_dict,
                old_eddi_weights=old_eddi_weights
            )
            loss_mort = criterion_mortality(mort_logit, labels_mortality.unsqueeze(1))
            loss_los = criterion_los(los_logit, labels_los.unsqueeze(1))
            loss_mv = criterion_mech(mv_logit, labels_mechvent.unsqueeze(1))
            eddi_loss = ((mort_logit - target) ** 2).mean()
            loss = loss_mort + loss_los + loss_mv + loss_gamma * eddi_loss
            running_loss += loss.item()
            last_eddi_details = eddi_details
    return running_loss, last_eddi_details

def evaluate_model_with_confusion(model, dataloader, device, threshold=0.5, old_eddi_weights=None):
    model.eval()
    all_mort_logits, all_los_logits, all_mv_logits = [], [], []
    all_labels_mort, all_labels_los, all_labels_mv = [], [], []
    all_age, all_ethnicity, all_insurance = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding,
             labels_mortality, labels_los, labels_mechvent) = [x.to(device) for x in batch]
            
            y_true_dict = {
                "mortality": labels_mortality.cpu().numpy(),
                "los": labels_los.cpu().numpy(),
                "mechanical_ventilation": labels_mechvent.cpu().numpy()
            }
            sensitive_labels_dict = {
                "mortality": gender_ids.cpu().numpy(),
                "los": gender_ids.cpu().numpy(),
                "mechanical_ventilation": gender_ids.cpu().numpy()
            }
            
            mort_logits, los_logits, mv_logits, _ = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding, beta=0.3,
                y_true_dict=y_true_dict, sensitive_labels_dict=sensitive_labels_dict,
                old_eddi_weights=old_eddi_weights
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_mv_logits.append(mv_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_mv.append(labels_mechvent.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_los_logits  = torch.cat(all_los_logits, dim=0)
    all_mv_logits   = torch.cat(all_mv_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_los  = torch.cat(all_labels_los, dim=0)
    all_labels_mv   = torch.cat(all_labels_mv, dim=0)
    
    mort_probs = torch.sigmoid(all_mort_logits).numpy().squeeze()
    los_probs  = torch.sigmoid(all_los_logits).numpy().squeeze()
    mv_probs   = torch.sigmoid(all_mv_logits).numpy().squeeze()
    labels_mort_np = all_labels_mort.numpy().squeeze()
    labels_los_np  = all_labels_los.numpy().squeeze()
    labels_mv_np   = all_labels_mv.numpy().squeeze()
    
    metrics = {}
    for task, probs, labels in zip(["mortality", "los", "mechanical_ventilation"],
                                     [mort_probs, los_probs, mv_probs],
                                     [labels_mort_np, labels_los_np, labels_mv_np]):
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
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall_val, "precision": precision_val,
                         "tpr": tpr, "fpr": fpr}
    
    # Compute subgroup fairness metrics using demographic attributes.
    all_age = torch.cat(all_age, dim=0).numpy().squeeze()
    all_ethnicity = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    all_insurance = torch.cat(all_insurance, dim=0).numpy().squeeze()
    age_groups = np.array([get_age_bucket(a) for a in all_age])
    ethnicity_groups = np.array([map_ethnicity(e) for e in all_ethnicity])
    insurance_groups = np.array([map_insurance(i) for i in all_insurance])

    eddi_stats = {}
    for outcome, probs, labels in zip(["mortality", "los", "mechanical_ventilation"],
                                      [mort_probs, los_probs, mv_probs],
                                      [labels_mort_np, labels_los_np, labels_mv_np]):
        overall_age, age_eddi_sub = compute_eddi(labels.astype(int), probs, age_groups, threshold)
        overall_eth, eth_eddi_sub = compute_eddi(labels.astype(int), probs, ethnicity_groups, threshold)
        overall_ins, ins_eddi_sub = compute_eddi(labels.astype(int), probs, insurance_groups, threshold)
        total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
        eddi_stats[outcome] = {
            "age_eddi": overall_age,
            "age_subgroup_eddi": age_eddi_sub,
            "ethnicity_eddi": overall_eth,
            "ethnicity_subgroup_eddi": eth_eddi_sub,
            "insurance_eddi": overall_ins,
            "insurance_subgroup_eddi": ins_eddi_sub,
            "final_EDDI": total_eddi
        }
    metrics["eddi_stats"] = eddi_stats

    
    fairness_metrics = {}
    mort_preds = (mort_probs > threshold).astype(int)
    los_preds  = (los_probs > threshold).astype(int)
    mv_preds   = (mv_probs > threshold).astype(int)
    
    for outcome, labels, preds in zip(["mortality", "los", "mechanical_ventilation"],
                                      [labels_mort_np, labels_los_np, labels_mv_np],
                                      [mort_preds, los_preds, mv_preds]):
        outcome_fairness = {}
        for attr_name, attr_values in zip(["age", "ethnicity", "insurance"],
                                          [age_groups, ethnicity_groups, insurance_groups]):
            avg_tpr_diff, avg_fpr_diff = calculate_equalized_odds_difference(labels, preds, attr_values)
            combined_metric = (avg_tpr_diff + avg_fpr_diff) / 2
            outcome_fairness[attr_name] = {
                "Average TPR Difference": avg_tpr_diff,
                "Average FPR Difference": avg_fpr_diff,
                "Combined Equalized Odds Metric": combined_metric,
                "Group Fairness Metrics": calculate_multiclass_fairness_metrics(labels, preds, attr_values),
                "Predictive Parity": calculate_predictive_parity(labels, preds, attr_values)
            }
        fairness_metrics[outcome] = outcome_fairness
    
    overall_eo_metrics = {}
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        eo_vals = []
        for attr in ["age", "ethnicity", "insurance"]:
            eo = fairness_metrics[outcome][attr]["Combined Equalized Odds Metric"]
            eo_vals.append(eo)
        overall_eo = np.mean(eo_vals)
        overall_eo_metrics[outcome] = overall_eo
    metrics["overall_eo_metrics"] = overall_eo_metrics

    metrics["fairness_metrics"] = fairness_metrics
    return metrics

def evaluate_model(model, dataloader, device, threshold=0.5, old_eddi_weights=None):
    metrics = evaluate_model_with_confusion(model, dataloader, device, threshold, old_eddi_weights=old_eddi_weights)
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

    labels = df_filtered[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_val_idx, test_idx in msss.split(df_filtered, labels):
        train_val_df = df_filtered.iloc[train_val_idx]
        test_df = df_filtered.iloc[test_idx]
    
    labels_train_val = train_val_df[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    for train_idx, val_idx in msss_val.split(train_val_df, labels_train_val):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)

    print("\nProcessing training text embeddings...")
    agg_text_train = apply_bioclinicalbert_on_patient_notes(train_df, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")
    print("Processing validation text embeddings...")
    agg_text_val = apply_bioclinicalbert_on_patient_notes(val_df, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")
    print("Processing test text embeddings...")
    agg_text_test = apply_bioclinicalbert_on_patient_notes(test_df, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")

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
            lab_features_t, aggregated_text_embedding,
            labels_mortality, labels_los, labels_mechvent
        )
        return dataset

    train_dataset = create_dataset(train_df, agg_text_train, lab_train)
    val_dataset = create_dataset(val_df, agg_text_val, lab_val)
    test_dataset = create_dataset(test_df, agg_text_test, lab_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

    beta_value = 0.3

    multimodal_model = MultimodalTransformer(
        text_embed_size=768,
        behrt_demo=behrt_demo,
        behrt_lab=behrt_lab,
        device=device,
        beta=beta_value
    ).to(device)

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)

    global criterion_mortality, criterion_los, criterion_mech
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')

    max_epochs = 20
    patience_limit = 5  
    loss_gamma = 1.0

    best_val_loss = float('inf')
    epochs_no_improve = 0

    old_eddi_weights = {}

    for epoch in range(max_epochs):
        train_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                beta=beta_value, loss_gamma=loss_gamma, target=1.0,
                                old_eddi_weights=old_eddi_weights)
        train_loss_epoch = train_loss / len(train_loader)
        
        val_loss, last_eddi_details = validate_step(multimodal_model, val_loader, device,
                                                    beta=beta_value, loss_gamma=loss_gamma, target=1.0,
                                                    old_eddi_weights=old_eddi_weights)
        val_loss_epoch = val_loss / len(val_loader)
        
        val_metrics = evaluate_model_with_confusion(multimodal_model, val_loader, device,
                                                    threshold=0.5, old_eddi_weights=old_eddi_weights)
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            m = val_metrics[outcome]
            print(f"  {outcome.capitalize()}: AUROC={m['aucroc']:.4f}, AUPRC={m['auprc']:.4f}, F1={m['f1']:.4f}, "
                  f"Recall={m['recall']:.4f}, Precision={m['precision']:.4f}, TPR={m['tpr']:.4f}, FPR={m['fpr']:.4f}")
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            eddi = val_metrics["eddi_stats"][outcome]["final_EDDI"]
            print(f"  {outcome.capitalize()} EDDI: {eddi:.4f}")
        print("\nFairness Metrics (Equalized Odds) on Validation Set:")
        for outcome, fm in val_metrics["fairness_metrics"].items():
            print(f"\nOutcome: {outcome.capitalize()}")
            for attr, vals in fm.items():
                print(f"  {attr.capitalize()}:")
                print(f"    Avg TPR Diff: {vals['Average TPR Difference']:.4f}")
                print(f"    Avg FPR Diff: {vals['Average FPR Difference']:.4f}")
                print(f"    Combined Equalized Odds Metric: {vals['Combined Equalized Odds Metric']:.4f}")
        print("\nOverall EO Metrics (averaged over age, ethnicity, insurance):")
        for outcome, overall_eo in val_metrics["overall_eo_metrics"].items():
            print(f"  {outcome.capitalize()} Overall EO: {overall_eo:.4f}")
        
        scheduler.step(val_loss_epoch)
        
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

        if last_eddi_details is not None:
            old_eddi_weights = {
                "mortality": last_eddi_details["mortality"]["weights"],
                "los": last_eddi_details["los"]["weights"],
                "mechanical_ventilation": last_eddi_details["mechanical_ventilation"]["weights"]
            }
            print("Updated old EDDI weights for next epoch:", old_eddi_weights)
    
    print("Training complete.\n")
    
    multimodal_model.load_state_dict(torch.load("best_model.pth"))
    final_metrics = evaluate_model(multimodal_model, test_loader, device, threshold=0.5, old_eddi_weights=old_eddi_weights)
    
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
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        eddi_stats = final_metrics["eddi_stats"][outcome]
        print(f"\n{outcome.capitalize()}:")
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
        print("  Final Overall EDDI: {:.4f}".format(eddi_stats["final_EDDI"]))

    print("\n--- Additional Fairness Metrics on Test Set ---")
    fairness_final = final_metrics["fairness_metrics"]
    for outcome, fm in fairness_final.items():
        print(f"\nOutcome: {outcome.capitalize()}")
        for attr, vals in fm.items():
            print(f"  {attr.capitalize()}:")
            print(f"    Avg TPR Diff: {vals['Average TPR Difference']:.4f}")
            print(f"    Avg FPR Diff: {vals['Average FPR Difference']:.4f}")
            print(f"    Combined Equalized Odds Metric: {vals['Combined Equalized Odds Metric']:.4f}")
    
    print("\n--- Overall EO Metrics on Test Set ---")
    for outcome, overall_eo in final_metrics["overall_eo_metrics"].items():
        print(f"  {outcome.capitalize()} Overall EO: {overall_eo:.4f}")
    
    print("\nTesting complete.")

if __name__ == "__main__":
    train_pipeline()
