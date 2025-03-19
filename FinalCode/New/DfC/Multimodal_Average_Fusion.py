import os
import sys
# Remove current directory from sys.path to avoid conflicts with local modules named like standard ones.
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

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


def compute_eddi(true_labels, predicted_labels, sensitive_labels, threshold=0.5):
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

# Structured Branch: Dummy BEHRT (No Demographics)
class BEHRT_NoDemo(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        """
        A simplified BEHRT-like model that uses a dummy input.
        Since we exclude demographic/clinical features, only a dummy input is used.
        """
        super(BEHRT_NoDemo, self).__init__()
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
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

# Multimodal Fusion Model (Excluding Demographics)
class MultimodalTransformer_NoDemo(nn.Module):
    def __init__(self, text_embed_size, BEHRT, hidden_size=512):
        """
        Fuses the dummy structured branch and the unstructured text branch.
        """
        super(MultimodalTransformer_NoDemo, self).__init__()
        self.BEHRT = BEHRT  # Structured branch (dummy input)
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
            nn.Linear(hidden_size, 3)  # Three outcomes: mortality, LOS, mechanical ventilation
        )

    def forward(self, input_ids_struct, attn_mask_struct, aggregated_text_embedding):
        structured_emb = self.BEHRT(input_ids_struct, attn_mask_struct)
        ts_proj = self.ts_projector(structured_emb)
        text_proj = self.text_projector(aggregated_text_embedding)
        combined = torch.cat((ts_proj, text_proj), dim=1)
        logits = self.classifier(combined)
        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits = logits[:, 1].unsqueeze(1)
        vent_logits = logits[:, 2].unsqueeze(1)
        return mortality_logits, los_logits, vent_logits

# Training and Evaluation Functions
def train_step(model, dataloader, optimizer, device, crit_mort, crit_los, crit_vent):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        # In this DfC setting, the batch contains:
        # [dummy_input_ids, dummy_attn_mask, aggregated_text_embedding, labels_mortality, labels_los, labels_vent, sensitive attributes...]
        (dummy_input_ids, dummy_attn_mask,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_vent, *_sensitive) = [x.to(device) for x in batch]
        optimizer.zero_grad()
        mortality_logits, los_logits, vent_logits = model(dummy_input_ids, dummy_attn_mask, aggregated_text_embedding)
        loss_mort = crit_mort(mortality_logits, labels_mortality.unsqueeze(1))
        loss_los = crit_los(los_logits, labels_los.unsqueeze(1))
        loss_vent = crit_vent(vent_logits, labels_vent.unsqueeze(1))
        loss = loss_mort + loss_los + loss_vent
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate the model and compute performance metrics.
    Additionally, use the appended sensitive features (age, ethnicity, insurance)
    for post-hoc fairness analysis (EDDI calculation).
    """
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_vent_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_labels_vent = []
    sensitive_dict = {"age": [], "ethnicity": [], "insurance": []}
    with torch.no_grad():
        for batch in dataloader:
            # In our dataset, the batch contains:
            # [dummy_input_ids, dummy_attn_mask, aggregated_text_embedding,
            #  labels_mortality, labels_los, labels_vent,
            #  age, gender, ethnicity, insurance]
            (dummy_input_ids, dummy_attn_mask,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent,
             age, gender, ethnicity, insurance) = [x.to(device) for x in batch]
            
            mort_logits, los_logits, vent_logits = model(dummy_input_ids, dummy_attn_mask, aggregated_text_embedding)
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_vent_logits.append(vent_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_vent.append(labels_vent.cpu())
            sensitive_dict["age"].append(age.cpu())
            sensitive_dict["ethnicity"].append(ethnicity.cpu())
            sensitive_dict["insurance"].append(insurance.cpu())
    
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
    
    # Compute performance metrics for each task
    metrics = {}
    for task, probs, labels in zip(["mortality", "los", "mechanical_ventilation"],
                                    [mort_probs, los_probs, vent_probs],
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
        rec = recall_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": rec, "precision": prec}
    
    # Combine sensitive attributes from all batches
    age_all = torch.cat(sensitive_dict["age"], dim=0).numpy().squeeze()
    eth_all = torch.cat(sensitive_dict["ethnicity"], dim=0).numpy().squeeze()
    ins_all = torch.cat(sensitive_dict["insurance"], dim=0).numpy().squeeze()
    
    # Create subgroup arrays (using the provided mapping functions)
    age_groups = np.array([get_age_bucket(a) for a in age_all])
    eth_groups = np.array([map_ethnicity(e) for e in eth_all])
    ins_groups = np.array([map_insurance(i) for i in ins_all])
    
    # Calculate EDDI for each outcome
    eddi_stats = {}
    for task, labels_np, probs in zip(["mortality", "los", "mechanical_ventilation"],
                                      [labels_mort_np, labels_los_np, labels_vent_np],
                                      [mort_probs, los_probs, vent_probs]):
        overall_age, age_eddi_sub = compute_eddi(labels_np.astype(int), probs, age_groups, threshold)
        overall_eth, eth_eddi_sub = compute_eddi(labels_np.astype(int), probs, eth_groups, threshold)
        overall_ins, ins_eddi_sub = compute_eddi(labels_np.astype(int), probs, ins_groups, threshold)
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
    return metrics

# DfC Training Pipeline (Excluding Demographics)
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    keep_cols = {"subject_id", "hadm_id", "short_term_mortality", "los_binary", "mechanical_ventilation"}
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

    # Convert outcomes to int
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
    
    df_unique = df_filtered.groupby("subject_id", as_index=False).first()
    print("Number of unique patients:", len(df_unique))
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    from transformers import BertModel
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_unique, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)
    
    num_samples = len(df_unique)
    # Dummy structured branch inputs
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    
    labels_mortality = torch.tensor(df_unique["short_term_mortality"].values, dtype=torch.float32)
    labels_los = torch.tensor(df_unique["los_binary"].values, dtype=torch.float32)
    labels_vent = torch.tensor(df_unique["mechanical_ventilation"].values, dtype=torch.float32)
    
    # Retain sensitive features for fairness evaluation (even though they are not used by the model)
    age_ids = torch.tensor(df_unique["age"].values, dtype=torch.long)
    # For ethnicity and insurance, assume columns exist (if not, default values are created)
    if "ethnicity" not in df_unique.columns:
        df_unique["ethnicity"] = 0
    if "insurance" not in df_unique.columns:
        df_unique["insurance"] = 0
    ethnicity_ids = torch.tensor(df_unique["ethnicity"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_unique["insurance"].values, dtype=torch.long)
    
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')
    
    # Create dataset with sensitive attributes appended (for post-hoc EDDI calculation)
    dataset = TensorDataset(
        dummy_input_ids, dummy_attn_mask,
        aggregated_text_embeddings_t,
        labels_mortality, labels_los, labels_vent,
        age_ids, ethnicity_ids, insurance_ids
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # For the structured branch we use a dummy BEHRT with vocab_size=1.
    vocab_size = 1
    behrt_model_dummy = BEHRT_NoDemo(vocab_size=vocab_size, hidden_size=768).to(device)
    
    # Multimodal fusion model that fuses dummy structured branch with unstructured branch.
    dfc_multimodal_model = MultimodalTransformer_NoDemo(text_embed_size=768, BEHRT=behrt_model_dummy, hidden_size=512).to(device)
    
    optimizer = torch.optim.Adam(dfc_multimodal_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        dfc_multimodal_model.train()
        running_loss = train_step(dfc_multimodal_model, dataloader, optimizer, device,
                                  criterion_mortality, criterion_los, criterion_mech)
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)
    
    metrics = evaluate_model(dfc_multimodal_model, dataloader, device, threshold=0.5)
    print("\nFinal Evaluation Metrics (DfC):")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
              f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")
    
    print("\nDetailed EDDI Statistics (DfC):")
    eddi_stats = metrics["eddi_stats"]
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        print(f"\n{outcome.capitalize()} EDDI Stats:")
        stats = eddi_stats[outcome]
        print("  Aggregated Age EDDI    : {:.4f}".format(stats["age_eddi"]))
        print("  Age Subgroup EDDI:")
        for bucket, score in stats["age_subgroup_eddi"].items():
            print(f"    {bucket}: {score:.4f}")
        print("  Aggregated Ethnicity EDDI: {:.4f}".format(stats["ethnicity_eddi"]))
        print("  Ethnicity Subgroup EDDI:")
        for group, score in stats["ethnicity_subgroup_eddi"].items():
            print(f"    {group}: {score:.4f}")
        print("  Aggregated Insurance EDDI: {:.4f}".format(stats["insurance_eddi"]))
        print("  Insurance Subgroup EDDI:")
        for group, score in stats["insurance_subgroup_eddi"].items():
            print(f"    {group}: {score:.4f}")
        print("  Final Overall EDDI     : {:.4f}".format(stats["final_EDDI"]))
    
    print("Training complete (DfC).")
    
if __name__ == "__main__":
    train_pipeline()
