import os
import sys
# Remove current directory from sys.path to avoid conflicts with local modules.
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

import os
import sys
import time
import random
import argparse
import datetime  # <-- For timestamping
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

def compute_eddi(y_true, y_pred, sensitive_labels, threshold=0.5, complete_groups=None):
    y_pred_bin = (y_pred > threshold).astype(int)
    
    # Determine groups to check: either from complete_groups or the unique groups in the sensitive labels.
    groups = np.array(complete_groups) if complete_groups is not None else np.unique(sensitive_labels)
    
    # Compute overall error rate of the entire dataset.
    overall_error = np.mean(y_pred_bin != y_true)
    # Protect against degenerate cases: if overall_error is 0 or 1, we use 1.0 as the denominator.
    denom = overall_error if overall_error not in [0, 1] else 1.0
    if overall_error < 0.5:
        denom = 1 - overall_error
    else:
        denom = overall_error
    
    subgroup_eddi = {}
    valid_groups_count = 0  # Count of groups with at least one sample.
    
    # Compute the EDDI for each subgroup that has samples.
    for group in groups:
        mask = (sensitive_labels == group)
        if np.sum(mask) == 0:
            continue
        valid_groups_count += 1
        er_group = np.mean(y_pred_bin[mask] != y_true[mask])
        subgroup_eddi[group] = (er_group - overall_error) / denom

    # Compute RMS over subgroups with samples.
    if valid_groups_count > 0:
        overall_eddi = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / valid_groups_count
    else:
        overall_eddi = 0.0

    return overall_eddi, subgroup_eddi

# Additional Fairness Metric Functions
def calculate_tpr_and_fpr(y_true, y_pred, group_mask):
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for a given group.
    """
    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask], labels=[1, 0])
    if np.prod(cm.shape) != 4:
        TP = np.sum((y_true[group_mask] == 1) & (y_pred[group_mask] == 1))
        FN = np.sum((y_true[group_mask] == 1) & (y_pred[group_mask] == 0))
        FP = np.sum((y_true[group_mask] == 0) & (y_pred[group_mask] == 1))
        TN = np.sum((y_true[group_mask] == 0) & (y_pred[group_mask] == 0))
    else:
        tp, fn, fp, tn = cm.ravel()
        TP, FN, FP, TN = tp, fn, fp, tn

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    return tpr, fpr

def print_fairness_metrics(y_true, y_pred, demographics, sensitive_attr_name):

    unique_groups = np.unique(demographics)
    tpr_list = []
    fpr_list = []
    print(f"Fairness metrics for sensitive attribute: {sensitive_attr_name}")
    for group in unique_groups:
        group_mask = demographics == group
        tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred, group_mask)
        print(f"  Group {group}: TPR = {tpr:.3f}, FPR = {fpr:.3f}")
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    # Calculate absolute pairwise differences.
    tpr_diffs = []
    fpr_diffs = []
    for i, tpr_i in enumerate(tpr_list):
        for j in range(i+1, len(tpr_list)):
            tpr_diffs.append(abs(tpr_i - tpr_list[j]))
            fpr_diffs.append(abs(fpr_list[i] - fpr_list[j]))
    avg_tpr_diff = np.mean(tpr_diffs) if tpr_diffs else 0.0
    avg_fpr_diff = np.mean(fpr_diffs) if fpr_diffs else 0.0
    print(f"  Average TPR difference across groups: {avg_tpr_diff:.3f}")
    print(f"  Average FPR difference across groups: {avg_fpr_diff:.3f}\n")
    return avg_tpr_diff, avg_fpr_diff

def calculate_predictive_parity(y_true, y_pred, sensitive_attrs):
    unique_groups = np.unique(sensitive_attrs)
    precision_scores = {}
    for group in unique_groups:
        group_indices = sensitive_attrs == group
        precision = precision_score(y_true[group_indices], y_pred[group_indices], zero_division=0, average='weighted')
        precision_scores[group] = precision
    return precision_scores


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
                encoded = tokenizer.encode_plus(text=note,
                                                add_special_tokens=True,
                                                max_length=512,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                attn_mask = encoded['attention_mask'].to(device)
                with torch.no_grad():
                    emb = model(input_ids, attn_mask)
                embeddings.append(emb.cpu().numpy())
            embeddings = np.vstack(embeddings)
            agg_emb = np.mean(embeddings, axis=0) if aggregation == "mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    return np.vstack(aggregated_embeddings)

class BEHRTModel_Demo(nn.Module):
    def __init__(self, num_ages, num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel_Demo, self).__init__()
        vocab_size = num_ages + num_genders + num_ethnicities + num_insurances + 2
        config = BertConfig(vocab_size=vocab_size,
                            hidden_size=hidden_size,
                            num_hidden_layers=12,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1)
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
        return cls_token + extra

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
        return x.mean(dim=1)

class MultimodalTransformer_EDDI_Sigmoid(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device, fusion_hidden=512, beta=1.0):
        """
        Fusion Process:
          1. Each modality (demo, lab, text) is projected into a 256-D space.
          2. The projections are weighted (dynamically updated) and concatenated into a 768-D vector.
          3. A shared 768-D sigmoid gating vector is applied elementwise.
          4. The gated representation is fed through an MLP to produce 3 logits.
        """
        super(MultimodalTransformer_EDDI_Sigmoid, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device
        self.beta = beta

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

        self.classifier_demo = nn.Linear(256, 3)
        self.classifier_lab = nn.Linear(256, 3)
        self.classifier_text = nn.Linear(256, 3)

        self.sig_weights = nn.Parameter(torch.randn(768))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(768, fusion_hidden),  # Pre-ReLU extraction here.
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden, 3)
        )

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding,
                beta=None, old_eddi_weights=None,
                return_modality_logits=False,
                return_gated_vector=False,
                return_intermediate=False):
        if beta is None:
            beta = self.beta

        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        demo_proj = self.demo_projector(demo_embedding)
        lab_proj = self.lab_projector(lab_embedding)
        text_proj = self.text_projector(text_embedding)

        if old_eddi_weights is None:
            weight_demo = weight_lab = weight_text = 0.33
        else:
            weight_demo = old_eddi_weights.get("mortality", {"demo": 0.33})["demo"]
            weight_lab = old_eddi_weights.get("mortality", {"lab": 0.33})["lab"]
            weight_text = old_eddi_weights.get("mortality", {"text": 0.33})["text"]

        weighted_demo = weight_demo * demo_proj
        weighted_lab = weight_lab * lab_proj
        weighted_text = weight_text * text_proj

        fused_vector = torch.cat([weighted_demo, weighted_lab, weighted_text], dim=1)
        sig_weights = torch.sigmoid(self.sig_weights)
        gated_vector = fused_vector * sig_weights

        fusion_pre_relu = self.fusion_mlp[0](gated_vector)
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
        if return_gated_vector:
            outputs["gated_vector"] = gated_vector
        if return_intermediate:
            outputs["fusion_pre_relu"] = fusion_pre_relu
        return outputs

def update_dynamic_weights_all_tasks(model, dataloader, device, old_eddi_weights, beta, threshold=0.5):
    """
    Compute outcome-specific RMS-based EDDI (using the demo modality predictions)
    and update dynamic weights for each outcome.
    
    For each modality (demo, lab, text), we compute the overall EDDI using the RMS of the
    per-sensitive-attribute EDDI values (age, ethnicity, insurance). Then, the maximum overall
    EDDI (across the modalities) is used to update each modality weight.
    
    The update rule per modality is:
      new_weight = old_weight + beta * (max_eddi - modality_overall_eddi)
    
    The weight updates are clipped to a maximum magnitude (update_limit) and finally normalized
    so that the weights sum to 1.
    
    Expected subgroup codes are passed via complete_groups.
    """
    outcome_names = ["mortality", "los", "mechanical_ventilation"]
    predictions = {outcome: {"demo": [], "lab": [], "text": []} for outcome in outcome_names}
    labels_all = {outcome: [] for outcome in outcome_names}
    sensitive_attrs = {"age": [], "ethnicity": [], "insurance": []}

    # Collect predictions, labels, and sensitive attributes from each batch.
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
            for outcome_idx, outcome in enumerate(outcome_names):
                demo_pred = (torch.sigmoid(modality_logits["demo"])[:, outcome_idx] > threshold).float().cpu().numpy()
                lab_pred = (torch.sigmoid(modality_logits["lab"])[:, outcome_idx] > threshold).float().cpu().numpy()
                text_pred = (torch.sigmoid(modality_logits["text"])[:, outcome_idx] > threshold).float().cpu().numpy()
                predictions[outcome]["demo"].append(demo_pred)
                predictions[outcome]["lab"].append(lab_pred)
                predictions[outcome]["text"].append(text_pred)
                labels_all[outcome].append(labels[:, outcome_idx].cpu().numpy())
            sensitive_attrs["age"].append(age_ids.cpu().numpy())
            sensitive_attrs["ethnicity"].append(ethnicity_ids.cpu().numpy())
            sensitive_attrs["insurance"].append(insurance_ids.cpu().numpy())
    
    for outcome in outcome_names:
        for modality in ["demo", "lab", "text"]:
            predictions[outcome][modality] = np.concatenate(predictions[outcome][modality])
        labels_all[outcome] = np.concatenate(labels_all[outcome])
    for key in sensitive_attrs:
        sensitive_attrs[key] = np.concatenate(sensitive_attrs[key])
    
    expected_age_codes = np.array([0, 1, 2, 3])
    expected_ethnicity_codes = np.array([0, 1, 2, 3, 4])
    expected_insurance_codes = np.array([0, 1, 2, 3, 4, 5])
    
    new_weights = {}
    for outcome in outcome_names:
        def modality_overall_eddi(mod_preds):
            eddi_age, _ = compute_eddi(labels_all[outcome], mod_preds, sensitive_attrs["age"], threshold=threshold, complete_groups=expected_age_codes)
            eddi_ethnicity, _ = compute_eddi(labels_all[outcome], mod_preds, sensitive_attrs["ethnicity"], threshold=threshold, complete_groups=expected_ethnicity_codes)
            eddi_insurance, _ = compute_eddi(labels_all[outcome], mod_preds, sensitive_attrs["insurance"], threshold=threshold, complete_groups=expected_insurance_codes)
            overall_val = np.sqrt(eddi_age**2 + eddi_ethnicity**2 + eddi_insurance**2) / 3.0
            return overall_val
        
        eddi_demo = modality_overall_eddi(predictions[outcome]["demo"])
        eddi_lab  = modality_overall_eddi(predictions[outcome]["lab"])
        eddi_text = modality_overall_eddi(predictions[outcome]["text"])
        eddi_max = max(eddi_demo, eddi_lab, eddi_text)
        
        print(f"[{outcome} Weight Update] EDDI:")
        print("  Demo modality - Overall EDDI: {:.4f}".format(eddi_demo))
        print("  Lab modality  - Overall EDDI: {:.4f}".format(eddi_lab))
        print("  Text modality - Overall EDDI: {:.4f}".format(eddi_text))
        print("  Maximum EDDI among modalities: {:.4f}".format(eddi_max))
        
        prev = old_eddi_weights.get(outcome, {"demo": 0.33, "lab": 0.33, "text": 0.33})
        raw_update_demo = beta * (eddi_max - eddi_demo)
        raw_update_lab  = beta * (eddi_max - eddi_lab)
        raw_update_text = beta * (eddi_max - eddi_text)
        
        update_limit = 0.05
        update_demo = np.clip(raw_update_demo, -update_limit, update_limit)
        update_lab  = np.clip(raw_update_lab, -update_limit, update_limit)
        update_text = np.clip(raw_update_text, -update_limit, update_limit)
        
        new_demo = max(prev["demo"] + update_demo, 0.1)
        new_lab  = max(prev["lab"] + update_lab, 0.1)
        new_text = max(prev["text"] + update_text, 0.1)
        total = new_demo + new_lab + new_text
        
        new_weights[outcome] = {
            "demo": new_demo / total,
            "lab": new_lab / total,
            "text": new_text / total
        }
        print(f"[{outcome} Weight Update] New dynamic weights: {new_weights[outcome]}\n")
    
    return new_weights

def train_step(model, dataloader, optimizer, device, criterion, beta=1.0,
               lambda_edd=1.0, lambda_l1=0.01, target=1.0, threshold=0.5,
               old_eddi_weights=None):
    model.train()
    running_loss = 0.0
    running_bce_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features, aggregated_text_embedding, labels) = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(demo_dummy_ids, demo_attn_mask,
                        age_ids, gender_ids, ethnicity_ids, insurance_ids,
                        lab_features, aggregated_text_embedding,
                        beta=beta, old_eddi_weights=old_eddi_weights,
                        return_modality_logits=True)
        fused_logits = outputs["fused_logits"]
        modality_logits = outputs["modality_logits"]

        bce_loss = criterion(fused_logits, labels)
        running_bce_loss += bce_loss.item()

        l1_reg = lambda_l1 * torch.sum(torch.abs(model.sig_weights))
        fused_probs = torch.sigmoid(fused_logits)
        leddi_losses = []
        num_outcomes = fused_probs.shape[1]
        for i in range(num_outcomes):
            p_i = fused_probs[:, i]
            y_i = labels[:, i]
            overall_err = torch.mean(torch.abs(p_i - y_i))
            for sens_tensor in [age_ids, ethnicity_ids, insurance_ids]:
                unique_groups = torch.unique(sens_tensor)
                group_diffs = []
                for group in unique_groups:
                    mask = (sens_tensor == group)
                    if mask.sum() > 0:
                        subgroup_err = torch.mean(torch.abs(p_i[mask] - y_i[mask]))
                        group_diffs.append((subgroup_err - overall_err) ** 2)
                if group_diffs:
                    rmse = torch.sqrt(torch.mean(torch.stack(group_diffs)) + 1e-8)
                    leddi_losses.append(rmse)
        leddi_loss = torch.mean(torch.stack(leddi_losses)) if leddi_losses else 0.0

        total_loss = bce_loss + lambda_edd * (10 * leddi_loss) + l1_reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += total_loss.item()
    return running_loss, running_bce_loss

def calibrate_thresholds(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding, labels) = [x.to(device) for x in batch]
            outputs = model(demo_dummy_ids, demo_attn_mask,
                            age_ids, gender_ids, ethnicity_ids, insurance_ids,
                            lab_features, aggregated_text_embedding)
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
    """
    Evaluate the model on multiple outcomes and compute fairness metrics.
    For each outcome, subgroup fairness metrics (TPR, FPR) are computed per sensitive attribute
    (age, ethnicity, insurance), then the average differences are recorded.
    """
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
             lab_features, aggregated_text_embedding, labels) = [x.to(device) for x in batch]
            outputs = model(demo_dummy_ids, demo_attn_mask,
                            age_ids, gender_ids, ethnicity_ids, insurance_ids,
                            lab_features, aggregated_text_embedding)
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
    fairness_details = {}
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
            tpr = fpr = 0
        metrics[outcome] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                            "recall (TPR)": recall_val, "TPR": tpr,
                            "precision": precision_val, "fpr": fpr,
                            "optimal_threshold": thresh}
        fairness_details[outcome] = {}
        print(f"\nOutcome: {outcome} (Threshold: {thresh:.2f})")
        for attr_name, dem_values in zip(["age", "ethnicity", "insurance"],
                                         [all_age, all_ethnicity, all_insurance]):
            print_fairness_metrics(labels_np, preds, dem_values, sensitive_attr_name=attr_name)
            avg_tpr_diff, avg_fpr_diff = print_fairness_metrics(labels_np, preds, dem_values, sensitive_attr_name=attr_name)
            fairness_details[outcome][attr_name] = {"avg_tpr_diff": avg_tpr_diff, "avg_fpr_diff": avg_fpr_diff}
    return metrics, all_logits.numpy(), all_labels.numpy(), all_age, all_ethnicity, all_insurance

def evaluate_model(model, dataloader, device, threshold=0.5, old_eddi_weights=None):
    metrics, logits_all, labels_all, age_all, ethnicity_all, insurance_all = evaluate_model_multi(
        model, dataloader, device, thresholds=threshold, print_eddi=True)
    return metrics, logits_all, labels_all, age_all, ethnicity_all, insurance_all


def extract_and_save_vectors(model, dataloader, device, save_path="extracted_vectors1.npz"):
    """
    Extracts the gated vector and the fusion pre-ReLU vector for all batches,
    along with ground truth labels and sensitive attribute arrays, and saves them to a file.
    """
    model.eval()
    all_gated = []
    all_fusion_pre_relu = []
    all_labels = []
    all_age = []
    all_ethnicity = []
    all_insurance = []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features, aggregated_text_embedding, labels) = [x.to(device) for x in batch]
            outputs = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding,
                return_gated_vector=True,
                return_intermediate=True
            )
            gated_vec = outputs["gated_vector"]
            fusion_pre_relu = outputs["fusion_pre_relu"]
            
            all_gated.append(gated_vec.cpu().numpy())
            all_fusion_pre_relu.append(fusion_pre_relu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_age.append(age_ids.cpu().numpy())
            all_ethnicity.append(ethnicity_ids.cpu().numpy())
            all_insurance.append(insurance_ids.cpu().numpy())
    
    all_gated = np.concatenate(all_gated, axis=0)
    all_fusion_pre_relu = np.concatenate(all_fusion_pre_relu, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_age = np.concatenate(all_age, axis=0)
    all_ethnicity = np.concatenate(all_ethnicity, axis=0)
    all_insurance = np.concatenate(all_insurance, axis=0)
    
    np.savez(save_path,
             gated_vectors=all_gated,
             fusion_pre_relu_vectors=all_fusion_pre_relu,
             labels=all_labels,
             age=all_age,
             ethnicity=all_ethnicity,
             insurance=all_insurance)
    print("Saved extracted vectors and metadata to", save_path)
    return all_gated, all_fusion_pre_relu, all_labels, all_age, all_ethnicity, all_insurance

def run_experiment(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    structured_data = pd.read_csv("final_structured_common.csv", low_memory=False)
    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)
    
    unstructured_data.drop(columns=["short_term_mortality", "los_binary", "mechanical_ventilation",
                                      "age", "GENDER", "ETHNICITY", "INSURANCE"],
                             errors='ignore', inplace=True)
    merged_df = pd.merge(structured_data, unstructured_data,
                         on=["subject_id", "hadm_id"],
                         how="inner", suffixes=("_struct", "_unstruct"))
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

    def get_age_bucket(age):
        try:
            age = float(age)
        except:
            return "Other"
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
    df_filtered['age'] = df_filtered['age'].apply(get_age_bucket)
    df_filtered['age'] = df_filtered['age'].astype("category").cat.codes

    def map_ethnicity(e):
        try:
            e = int(e)
            mapping = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian"}
            return mapping.get(e, "Other")
        except:
            e = str(e).strip().title()
            mapping = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic"}
            return mapping.get(e, "Other")
    if "ETHNICITY" in df_filtered.columns:
        df_filtered["ETHNICITY"] = df_filtered["ETHNICITY"].apply(map_ethnicity)
        df_filtered["ETHNICITY"] = df_filtered["ETHNICITY"].astype("category").cat.codes
    else:
        df_filtered["ETHNICITY"] = 0

    def map_insurance(i):
        try:
            i = int(i)
            mapping = {0: "Government", 1: "Medicare", 2: "Medicaid", 3: "Private", 4: "Self Pay"}
            return mapping.get(i, "Other")
        except:
            i = str(i).strip().title()
            mapping = {"Government": "Government", "Medicare": "Medicare", "Medicaid": "Medicaid",
                       "Private": "Private", "Self Pay": "Self Pay"}
            return mapping.get(i, "Other")
    if "INSURANCE" in df_filtered.columns:
        df_filtered["INSURANCE"] = df_filtered["INSURANCE"].apply(map_insurance)
        df_filtered["INSURANCE"] = df_filtered["INSURANCE"].astype("category").cat.codes
    else:
        df_filtered["INSURANCE"] = 0

    if "GENDER" in df_filtered.columns and df_filtered["GENDER"].dtype == object:
        df_filtered["GENDER"] = df_filtered["GENDER"].astype("category").cat.codes
    else:
        gender_col = "GENDERS" if "GENDERS" in df_filtered.columns else "GENDER"
        df_filtered[gender_col] = df_filtered[gender_col].astype("category").cat.codes

    # Identify lab feature columns.
    exclude_cols = set(["subject_id", "ROW_ID", "hadm_id", "ICUSTAY_ID",
                        "short_term_mortality", "los_binary", "mechanical_ventilation",
                        "age", "GENDER", "GENDERS", "ETHNICITY", "INSURANCE"])
    lab_feature_columns = [col for col in df_filtered.columns if col not in exclude_cols
                           and not col.startswith("note_")
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

    print("Computing aggregated text embeddings for each patient...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean")
    print("Aggregated text embeddings shape:", aggregated_text_embeddings_np.shape)
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    labels_multilabel = df_filtered[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values
    for train_val_idx, test_idx in msss.split(df_filtered, labels_multilabel):
        train_val_df = df_filtered.iloc[train_val_idx]
        test_df = df_filtered.iloc[test_idx]
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    for train_idx, val_idx in msss_val.split(train_val_df, train_val_df[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    def create_dataset(indices):
        return TensorDataset(demo_dummy_ids[indices], demo_attn_mask[indices],
                             age_ids[indices], gender_ids[indices], ethnicity_ids[indices], insurance_ids[indices],
                             lab_features_t[indices], aggregated_text_embeddings_t[indices], labels[indices])
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

    behrt_demo = BEHRTModel_Demo(num_ages=NUM_AGES, num_genders=NUM_GENDERS,
                                 num_ethnicities=NUM_ETHNICITIES, num_insurances=NUM_INSURANCES,
                                 hidden_size=768).to(device)
    behrt_lab = BEHRTModel_Lab(lab_token_count=NUM_LAB_FEATURES, hidden_size=768,
                               nhead=8, num_layers=2).to(device)
    beta_value = hparams['beta']
    multimodal_model = MultimodalTransformer_EDDI_Sigmoid(text_embed_size=768,
                                                          behrt_demo=behrt_demo,
                                                          behrt_lab=behrt_lab,
                                                          device=device,
                                                          fusion_hidden=512,
                                                          beta=beta_value).to(device)
    optimizer = AdamW(multimodal_model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    old_eddi_weights = {
        "mortality": {"demo": 0.33, "lab": 0.33, "text": 0.33},
        "los": {"demo": 0.33, "lab": 0.33, "text": 0.33},
        "mechanical_ventilation": {"demo": 0.33, "lab": 0.33, "text": 0.33}
    }
    tracked_dynamic_weights = {outcome: [] for outcome in ["mortality", "los", "mechanical_ventilation"]}
    tracked_sigmoid_weights = []
    
    dynamic_weights_csv = "dynamic_weights_per_epoch1.csv"
    with open(dynamic_weights_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Outcome", "demo_weight", "lab_weight", "text_weight"])
    
    max_epochs = hparams['num_epochs']
    for epoch in range(max_epochs):
        train_loss, bce_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                          criterion, beta=beta_value, lambda_edd=hparams['lambda_edd'],
                                          lambda_l1=hparams['lambda_l1'], target=1.0, threshold=hparams['threshold'],
                                          old_eddi_weights=old_eddi_weights)
        avg_train_loss = train_loss / len(train_loader)
        avg_bce_loss = bce_loss / len(train_loader)
        multimodal_model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (demo_dummy_ids, demo_attn_mask,
                 age_ids, gender_ids, ethnicity_ids, insurance_ids,
                 lab_features, aggregated_text_embedding, labels) = [x.to(device) for x in batch]
                outputs = multimodal_model(demo_dummy_ids, demo_attn_mask,
                                            age_ids, gender_ids, ethnicity_ids, insurance_ids,
                                            lab_features, aggregated_text_embedding,
                                            beta=beta_value, old_eddi_weights=old_eddi_weights)
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

        new_weights = update_dynamic_weights_all_tasks(multimodal_model, train_loader, device, old_eddi_weights,
                                                        beta=beta_value, threshold=hparams['threshold'])
        old_eddi_weights = new_weights
        for outcome in new_weights:
            print(f"Epoch {epoch+1} - {outcome} dynamic weights: {new_weights[outcome]}")
            tracked_dynamic_weights[outcome].append([new_weights[outcome]["demo"],
                                                      new_weights[outcome]["lab"],
                                                      new_weights[outcome]["text"]])
            with open(dynamic_weights_csv, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, outcome,
                                 new_weights[outcome]["demo"],
                                 new_weights[outcome]["lab"],
                                 new_weights[outcome]["text"]])
        tracked_sigmoid_weights.append(torch.sigmoid(multimodal_model.sig_weights).detach().cpu().numpy())
    
    print("Training complete.\n")
    if best_model_state is not None:
        multimodal_model.load_state_dict(best_model_state)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_filename = f"best_model_{timestamp}.pt"
        torch.save(best_model_state, best_model_filename)
        print("Saved best model to", best_model_filename)
        extracted_save_path = f"extracted_vectors_{timestamp}.npz"
        extract_and_save_vectors(multimodal_model, test_loader, device, save_path=extracted_save_path)
        
    val_thresholds = calibrate_thresholds(multimodal_model, val_loader, device)
    print("\nOptimal thresholds from validation:")
    for outcome, thresh in val_thresholds.items():
        print(f"{outcome}: {thresh:.2f}")
    final_metrics, logits_all, labels_all, age_all, ethnicity_all, insurance_all = evaluate_model(
        multimodal_model, test_loader, device, threshold=val_thresholds)
    print("\n--- Final Evaluation Metrics on Test Set ---")
    for outcome, m in final_metrics.items():
        print(f"\nOutcome: {outcome}")
        print("  AUROC     : {:.4f}".format(m["aucroc"]))
        print("  AUPRC     : {:.4f}".format(m["auprc"]))
        print("  F1 Score  : {:.4f}".format(m["f1"]))
        print("  Recall    : {:.4f}".format(m["recall (TPR)"]))
        print("  Precision : {:.4f}".format(m["precision"]))
        print("  TPR       : {:.4f}".format(m["TPR"]))
        print("  FPR       : {:.4f}".format(m["fpr"]))
        print("  Optimal Thresh: {:.2f}".format(m["optimal_threshold"]))
    
    expected_age_codes = [0, 1, 2, 3]
    expected_ethnicity_codes = [0, 1, 2, 3, 4]
    expected_insurance_codes = [0, 1, 2, 3, 4, 5]
    outcome_names = ["mortality", "los", "mechanical_ventilation"]
    print("\n--- Sensitive Subgroup EDDI Statistics ---")
    combined_eddi = {}
    for i, outcome in enumerate(outcome_names):
        probs = torch.sigmoid(torch.tensor(logits_all))[:, i].numpy().squeeze()
        true_vals = labels_all[:, i]
        thresh = val_thresholds[outcome]
        print(f"\nOutcome: {outcome} (Threshold: {thresh:.2f})")
        eddi_age, subgroup_age = compute_eddi(true_vals, probs, np.array(age_all), threshold=thresh, complete_groups=expected_age_codes)
        eddi_ethnicity, subgroup_ethnicity = compute_eddi(true_vals, probs, np.array(ethnicity_all), threshold=thresh, complete_groups=expected_ethnicity_codes)
        eddi_insurance, subgroup_insurance = compute_eddi(true_vals, probs, np.array(insurance_all), threshold=thresh, complete_groups=expected_insurance_codes)
        overall_combined = np.sqrt(eddi_age**2 + eddi_ethnicity**2 + eddi_insurance**2) / 3.0
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
        print(" Combined EDDI:", overall_combined)
    overall_combined_eddi = np.mean(list(combined_eddi.values()))
    print("\n--- Overall Combined EDDI across outcomes ---")
    print("Overall Combined EDDI:", overall_combined_eddi)
    
    np.save("tracked_dynamic_weights_1.0.npy", tracked_dynamic_weights)
    np.save("tracked_sigmoid_weights_1.0.npy", np.array(tracked_sigmoid_weights))
    
if __name__ == "__main__":
    hyperparameter_grid = [
        {'lr': 1e-5, 'num_epochs': 50, 'lambda_edd': 1.0, 'lambda_l1': 0.01,
         'batch_size': 16, 'threshold': 0.50, 'weight_decay': 0.01, 'beta': 1.0},
    ]
    results = {}
    for idx, hparams in enumerate(hyperparameter_grid):
        print("\n==============================")
        print(f"Running experiment {idx+1} with hyperparameters: {hparams}")
        metrics = run_experiment(hparams)
        results[f"experiment_{idx+1}"] = metrics
        if metrics is not None:
            print(f"Results for lambda_edd={hparams['lambda_edd']}:")
            for outcome, m in metrics.items():
                print(f" Outcome {outcome}: AUROC={m['aucroc']:.4f}, AUPRC={m['auprc']:.4f}, F1={m['f1']:.4f}")
        else:
            print("No metrics returned for experiment", idx+1)
    print("\n=== Hyperparameter Search Complete ===")
