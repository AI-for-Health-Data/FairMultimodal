import os
import re
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, confusion_matrix
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from transformers import BertModel, BertConfig


def compute_eddi(sensitive_attr, true_labels, pred_labels, threshold=0.5):
    y_pred_bin = (pred_labels > threshold).astype(int)
    unique_groups = np.unique(sensitive_attr)
    subgroup_eddi = {}
    overall_error = np.mean(y_pred_bin != true_labels)
    denom = max(overall_error, 1 - overall_error) if overall_error not in [0, 1] else 1.0
    for group in unique_groups:
        mask = (sensitive_attr == group)
        if np.sum(mask) == 0:
            subgroup_eddi[group] = np.nan
        else:
            er_group = np.mean(y_pred_bin[mask] != true_labels[mask])
            subgroup_eddi[group] = (er_group - overall_error) / denom
    overall_eddi = np.sqrt(np.nansum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return overall_eddi, subgroup_eddi

def compute_attribute_eddi(age_eddi, ethnicity_eddi, insurance_eddi):
    return np.sqrt(age_eddi**2 + ethnicity_eddi**2 + insurance_eddi**2) / 3.0

def print_subgroup_eddi(true_labels, pred_labels, sensitive_name, sensitive_values):
    overall_eddi, subgroup_eddi = compute_eddi(sensitive_values, true_labels, pred_labels)
    print(f"\nSensitive Attribute: {sensitive_name}")
    print(f"Overall Error Rate-based EDDI: {overall_eddi:.4f}")
    for group, disparity in subgroup_eddi.items():
        print(f"  Subgroup {group}: EDDI = {disparity:.4f}")
    return subgroup_eddi, overall_eddi


class BEHRTModel_Lab(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768, nhead=8, num_layers=2):
        super(BEHRTModel_Lab, self).__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Linear(1, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(lab_token_count, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, lab_features):
        # lab_features: (batch, lab_token_count)
        x = lab_features.unsqueeze(-1)  # (batch, lab_token_count, 1)
        x = self.token_embedding(x)       # (batch, lab_token_count, hidden_size)
        x = x + self.pos_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)            # (lab_token_count, batch, hidden_size)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)            # (batch, lab_token_count, hidden_size)
        lab_embedding = x.mean(dim=1)     # (batch, hidden_size)
        return lab_embedding


class BEHRTModel_Combined(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768):
        super(BEHRTModel_Combined, self).__init__()
        self.lab_model = BEHRTModel_Lab(lab_token_count, hidden_size, nhead=8, num_layers=2)
        self.fusion_fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mort = nn.Linear(hidden_size, 1)
        self.classifier_los = nn.Linear(hidden_size, 1)
        self.classifier_mech = nn.Linear(hidden_size, 1)
    
    def forward(self, lab_features):
        lab_embed = self.lab_model(lab_features)
        fused = self.fusion_fc(lab_embed)
        fused = self.dropout(fused)
        logits_mort = self.classifier_mort(fused)
        logits_los = self.classifier_los(fused)
        logits_mech = self.classifier_mech(fused)
        return logits_mort, logits_los, logits_mech


class FinalStructuredDataset(Dataset):
    """
    Dataset for structured data from final_structured_common.csv.
    Expected columns:
      - Lab features: columns starting with 'lab_t' (e.g., lab_t0, lab_t2, …)
      - Demographic features: 'age', 'gender', 'ethnicity_category', 'insurance_category'
      - Outcome labels: 'short_term_mortality', 'los_binary', 'mechanical_ventilation'
    """
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.fillna(0, inplace=True)
        # Identify lab columns
        self.lab_cols = [col for col in self.df.columns if col.startswith('lab_t')]
        self.lab_cols.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        # Normalize lab features
        for col in self.lab_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std
            else:
                self.df[col] = 0.0
        for col in ['gender', 'ethnicity_category', 'insurance_category']:
            self.df[col] = self.df[col].astype(str)
        self.df['gender_code'] = self.df['gender'].astype('category').cat.codes
        self.df['ethnicity_code'] = self.df['ethnicity_category'].astype('category').cat.codes
        self.df['insurance_code'] = self.df['insurance_category'].astype('category').cat.codes
        self.df['age_int'] = self.df['age'].astype(int)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lab_features = torch.tensor(row[self.lab_cols].values.astype(np.float32))
        # The demographic features are no longer used by the model but kept here in case needed for analysis.
        age = torch.tensor(row['age_int'], dtype=torch.long)
        gender = torch.tensor(row['gender_code'], dtype=torch.long)
        ethnicity = torch.tensor(row['ethnicity_code'], dtype=torch.long)
        insurance = torch.tensor(row['insurance_code'], dtype=torch.long)
        labels = torch.tensor([row['short_term_mortality'],
                               row['los_binary'],
                               row['mechanical_ventilation']], dtype=torch.float32)
        return lab_features, age, gender, ethnicity, insurance, labels


def compute_class_weights(loader):
    all_labels = []
    for batch in loader:
        *_, labels = batch
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    weights = []
    for i in range(3):
        pos_count = (all_labels[:, i] == 1).sum().item()
        neg_count = (all_labels[:, i] == 0).sum().item()
        weight = neg_count / pos_count if pos_count > 0 else 1.0
        weights.append(weight)
    return weights

def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=5, lr=1e-5, weight_decay=0.01):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    train_class_weights = compute_class_weights(train_loader)
    loss_fn_mort = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_class_weights[0], device=device))
    loss_fn_los = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_class_weights[1], device=device))
    loss_fn_mech = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_class_weights[2], device=device))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            lab_features, _, _, _, _, labels = batch
            lab_features = lab_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits_mort, logits_los, logits_mech = model(lab_features)
            loss_mort = loss_fn_mort(logits_mort.squeeze(), labels[:, 0])
            loss_los = loss_fn_los(logits_los.squeeze(), labels[:, 1])
            loss_mech = loss_fn_mech(logits_mech.squeeze(), labels[:, 2])
            loss = loss_mort + loss_los + loss_mech
            if torch.isnan(loss):
                print("NaN loss encountered; skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        
        # Validation pass: accumulate logits, predictions, and true labels.
        model.eval()
        val_losses = []
        val_logits = {'mortality': [], 'los': [], 'mech': []}
        val_preds = {'mortality': [], 'los': [], 'mech': []}
        val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                lab_features, _, _, _, _, labels = batch
                lab_features = lab_features.to(device)
                labels = labels.to(device)
                logits_mort, logits_los, logits_mech = model(lab_features)
                loss_mort = loss_fn_mort(logits_mort.squeeze(), labels[:, 0])
                loss_los = loss_fn_los(logits_los.squeeze(), labels[:, 1])
                loss_mech = loss_fn_mech(logits_mech.squeeze(), labels[:, 2])
                loss = loss_mort + loss_los + loss_mech
                val_losses.append(loss.item())
                val_logits['mortality'].append(logits_mort.cpu().numpy())
                val_logits['los'].append(logits_los.cpu().numpy())
                val_logits['mech'].append(logits_mech.cpu().numpy())
                preds_mort = (torch.sigmoid(logits_mort) > 0.5).int().cpu().numpy()
                preds_los = (torch.sigmoid(logits_los) > 0.5).int().cpu().numpy()
                preds_mech = (torch.sigmoid(logits_mech) > 0.5).int().cpu().numpy()
                val_preds['mortality'].append(preds_mort)
                val_preds['los'].append(preds_los)
                val_preds['mech'].append(preds_mech)
                val_trues.append(labels.cpu().numpy())
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        scheduler.step(avg_val_loss)
        
        # Concatenate validation outputs.
        for key in val_logits:
            val_logits[key] = np.concatenate(val_logits[key], axis=0)
        for key in val_preds:
            val_preds[key] = np.concatenate(val_preds[key], axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        
        # Compute metrics per outcome on the validation set.
        outcome_names = ["Mortality", "LOS", "Mechanical Ventilation"]
        for i, outcome in enumerate(outcome_names):
            try:
                auroc = roc_auc_score(val_trues[:, i], val_logits[list(val_logits.keys())[i]])
            except Exception:
                auroc = np.nan
            precision_vals, recall_vals, _ = precision_recall_curve(val_trues[:, i], val_logits[list(val_logits.keys())[i]])
            auprc = auc(recall_vals, precision_vals)
            prec = precision_score(val_trues[:, i], val_preds[list(val_preds.keys())[i]], zero_division=0)
            rec = recall_score(val_trues[:, i], val_preds[list(val_preds.keys())[i]])
            f1 = f1_score(val_trues[:, i], val_preds[list(val_preds.keys())[i]])
            cm = confusion_matrix(val_trues[:, i], val_preds[list(val_preds.keys())[i]])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
            FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
            print(f"Epoch {epoch+1} - {outcome} Metrics:")
            print(f"  AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
            print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            print(f"  TPR: {TPR:.4f}, FPR: {FPR:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_behrt_model.pt")
            print("Best model saved.\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.\n")
                break

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_logits = {'mortality': [], 'los': [], 'mech': []}
    all_predictions = {'mortality': [], 'los': [], 'mech': []}
    with torch.no_grad():
        for batch in dataloader:
            lab_features, _, _, _, _, labels = batch
            lab_features = lab_features.to(device)
            labels = labels.to(device)
            logits_mort, logits_los, logits_mech = model(lab_features)
            all_logits['mortality'].append(logits_mort.cpu().numpy())
            all_logits['los'].append(logits_los.cpu().numpy())
            all_logits['mech'].append(logits_mech.cpu().numpy())
            preds_mort = (torch.sigmoid(logits_mort) > threshold).cpu().numpy().astype(int)
            preds_los = (torch.sigmoid(logits_los) > threshold).cpu().numpy().astype(int)
            preds_mech = (torch.sigmoid(logits_mech) > threshold).cpu().numpy().astype(int)
            all_predictions['mortality'].append(preds_mort)
            all_predictions['los'].append(preds_los)
            all_predictions['mech'].append(preds_mech)
            all_labels.append(labels.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    for key in all_logits:
        all_logits[key] = np.concatenate(all_logits[key], axis=0)
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
    metrics = {}
    for i, task in enumerate(['mortality', 'los', 'mech']):
        auroc = roc_auc_score(all_labels[:, i], all_logits[task])
        prec = precision_score(all_labels[:, i], all_predictions[task], zero_division=0)
        rec = recall_score(all_labels[:, i], all_predictions[task])
        f1 = f1_score(all_labels[:, i], all_predictions[task])
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels[:, i], all_logits[task])
        auprc = auc(recall_vals, precision_vals)
        metrics[task] = {'auroc': auroc, 'auprc': auprc, 'precision': prec, 'recall': rec, 'f1': f1}
    return metrics, all_logits, all_labels

def get_model_predictions(model, dataloader, device, threshold=0.5):
    model.eval()
    all_predictions = {'mortality': [], 'los': [], 'mech': []}
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            lab_features, _, _, _, _, labels = batch
            lab_features = lab_features.to(device)
            labels = labels.to(device)
            logits_mort, logits_los, logits_mech = model(lab_features)
            preds_mort = (torch.sigmoid(logits_mort) > threshold).cpu().numpy().astype(int)
            preds_los = (torch.sigmoid(logits_los) > threshold).cpu().numpy().astype(int)
            preds_mech = (torch.sigmoid(logits_mech) > threshold).cpu().numpy().astype(int)
            all_predictions['mortality'].append(preds_mort)
            all_predictions['los'].append(preds_los)
            all_predictions['mech'].append(preds_mech)
            all_labels.append(labels.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
    return all_predictions['mortality'], all_predictions['los'], all_predictions['mech'], all_labels

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Subset

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    csv_file = "final_structured_common.csv"
    dataset = FinalStructuredDataset(csv_file)
    
    total_size = len(dataset)
    test_size = int(0.2 * total_size)       
    
    # Extract labels for stratification.
    df = pd.read_csv(csv_file)
    labels = df[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_val_idx, test_idx in msss.split(np.zeros(len(labels)), labels):
        pass

    val_fraction = 0.05 / (1 - (test_size / total_size))
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    labels_train_val = labels[train_val_idx]
    for train_idx_rel, val_idx_rel in msss_val.split(np.zeros(len(train_val_idx)), labels_train_val):
        train_idx = np.array(train_val_idx)[train_idx_rel]
        val_idx = np.array(train_val_idx)[val_idx_rel]

    # Create dataset subsets.
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Read CSV to extract sensitive attributes.
    df = pd.read_csv(csv_file)
    if 'age_group' not in df.columns:
        age_bins = [15, 30, 50, 70, 90]
        age_labels = ['15-29', '30-49', '50-69', '70-89']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)
    if 'categorized_ethnicity' in df.columns:
        ethnicity_groups = df['categorized_ethnicity'].values
    else:
        ethnicity_groups = df['ethnicity_category'].values
    if 'insurance_category' in df.columns:
        insurance_groups = df['insurance_category'].values
    else:
        insurance_groups = df['INSURANCE'].values
    
    # In this version, only lab features are used, so demo vocab sizes are not applicable.
    print("Demo vocab sizes removed since BEHRT demo is no longer used.")
    
    model = BEHRTModel_Combined(len(dataset.lab_cols), hidden_size=768)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    train_model(model, train_loader, val_loader, device, num_epochs=10, patience=5, lr=1e-5, weight_decay=0.01)
    model.load_state_dict(torch.load("best_behrt_model.pt", map_location=device))
    eval_metrics, all_logits, all_labels = evaluate_model(model, test_loader, device, threshold=0.5)
    print("Test Set Evaluation Metrics:")
    for task, m in eval_metrics.items():
        print(f"--- {task.upper()} ---")
        for subtask, value in m.items():
            print(f"{subtask}: {value:.4f}")
    
    preds_mort, preds_los, preds_mech, labels_arr = get_model_predictions(model, test_loader, device, threshold=0.5)
    
    # For fairness evaluation, extract sensitive attributes for test samples.
    test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else list(range(len(dataset)))[-len(test_loader.dataset):]
    df_sensitive = pd.read_csv(csv_file)
    if 'age_group' not in df_sensitive.columns:
        age_bins = [15, 30, 50, 70, 90]
        age_labels = ['15-29', '30-49', '50-69', '70-89']
        df_sensitive['age_group'] = pd.cut(df_sensitive['age'], bins=age_bins, labels=age_labels, right=False).astype(str)
    if 'categorized_ethnicity' in df_sensitive.columns:
        test_sensitive_ethnicity = df_sensitive.iloc[test_indices]['categorized_ethnicity'].values
    else:
        test_sensitive_ethnicity = df_sensitive.iloc[test_indices]['ethnicity_category'].values
    if 'insurance_category' in df_sensitive.columns:
        test_sensitive_insurance = df_sensitive.iloc[test_indices]['insurance_category'].values
    else:
        test_sensitive_insurance = df_sensitive.iloc[test_indices]['INSURANCE'].values

    test_sensitive_age = df_sensitive.iloc[test_indices]['age_group'].values

    print("\n--- EDDI Evaluation for Mortality Outcome ---")
    print("Age Groups:")
    _, age_overall_mort = print_subgroup_eddi(labels_arr[:, 0], preds_mort, "Age Groups", test_sensitive_age)
    print("Ethnicity Groups:")
    _, eth_overall_mort = print_subgroup_eddi(labels_arr[:, 0], preds_mort, "Ethnicity Groups", test_sensitive_ethnicity)
    print("Insurance Groups:")
    _, ins_overall_mort = print_subgroup_eddi(labels_arr[:, 0], preds_mort, "Insurance Groups", test_sensitive_insurance)
    overall_eddi_mort = compute_attribute_eddi(age_overall_mort, eth_overall_mort, ins_overall_mort)
    print(f"Combined Overall EDDI for Mortality: {overall_eddi_mort:.4f}\n")
    
    print("\n--- EDDI Evaluation for LOS Outcome ---")
    print("Age Groups:")
    _, age_overall_los = print_subgroup_eddi(labels_arr[:, 1], preds_los, "Age Groups", test_sensitive_age)
    print("Ethnicity Groups:")
    _, eth_overall_los = print_subgroup_eddi(labels_arr[:, 1], preds_los, "Ethnicity Groups", test_sensitive_ethnicity)
    print("Insurance Groups:")
    _, ins_overall_los = print_subgroup_eddi(labels_arr[:, 1], preds_los, "Insurance Groups", test_sensitive_insurance)
    overall_eddi_los = compute_attribute_eddi(age_overall_los, eth_overall_los, ins_overall_los)
    print(f"Combined Overall EDDI for LOS: {overall_eddi_los:.4f}\n")
    
    print("\n--- EDDI Evaluation for Mechanical Ventilation Outcome ---")
    print("Age Groups:")
    _, age_overall_mech = print_subgroup_eddi(labels_arr[:, 2], preds_mech, "Age Groups", test_sensitive_age)
    print("Ethnicity Groups:")
    _, eth_overall_mech = print_subgroup_eddi(labels_arr[:, 2], preds_mech, "Ethnicity Groups", test_sensitive_ethnicity)
    print("Insurance Groups:")
    _, ins_overall_mech = print_subgroup_eddi(labels_arr[:, 2], preds_mech, "Insurance Groups", test_sensitive_insurance)
    overall_eddi_mech = compute_attribute_eddi(age_overall_mech, eth_overall_mech, ins_overall_mech)
    print(f"Combined Overall EDDI for Mechanical Ventilation: {overall_eddi_mech:.4f}\n")

if __name__ == "__main__":
    main()
