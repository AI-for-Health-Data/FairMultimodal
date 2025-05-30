import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix

DEBUG = True


def calculate_tpr_and_fpr(y_true, y_pred, group_mask):
    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask], labels=[1, 0])
    if cm.size != 4:
        return 0, 0
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

def calculate_equalized_opportunity(y_true, y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    tpr_diffs = []
    fpr_diffs = []
    for i, group1 in enumerate(unique_groups):
        mask1 = sensitive_attr == group1
        tpr1, fpr1 = calculate_tpr_and_fpr(y_true, y_pred, mask1)
        for group2 in unique_groups[i+1:]:
            mask2 = sensitive_attr == group2
            tpr2, fpr2 = calculate_tpr_and_fpr(y_true, y_pred, mask2)
            tpr_diffs.append(abs(tpr1 - tpr2))
            fpr_diffs.append(abs(fpr1 - fpr2))
    avg_tpr_diff = np.mean(tpr_diffs) if tpr_diffs else 0
    avg_fpr_diff = np.mean(fpr_diffs) if fpr_diffs else 0
    combined_eo = (avg_tpr_diff + avg_fpr_diff) / 2
    details = {"avg_tpr_diff": avg_tpr_diff, "avg_fpr_diff": avg_fpr_diff}
    return combined_eo, details

def calculate_predictive_parity(y_true, y_pred, sensitive_attrs):
    unique_groups = np.unique(sensitive_attrs)
    precision_scores = {}
    for group in unique_groups:
        group_indices = sensitive_attrs == group
        precision = precision_score(y_true[group_indices], y_pred[group_indices], zero_division=0, average='weighted')
        precision_scores[group] = precision
    return precision_scores


def get_age_bucket(age):
    if 15 <= age <= 29:
        return "15-29"
    elif 30 <= age <= 49:
        return "30-49"
    elif 50 <= age <= 69:
        return "50-69"
    elif 70 <= age <= 90:
        return "70-90"
    else:
        return "Other"

def map_ethnicity(e):
    mapping = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian"}
    return mapping.get(e, "Other")

def map_insurance(i):
    mapping = {0: "Government", 1: "Medicare", 2: "Medicaid", 3: "Private", 4: "Self Pay"}
    return mapping.get(i, "Other")

def compute_eddi(y_true, y_pred, sensitive_labels):
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
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values()))**2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

# BioClinicalBERT Fine-Tuning Model for Text
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token embedding.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

# BEHRT for Demographics and Lab Features
class BEHRTModel_Demo(nn.Module):
    def __init__(self, num_ages, num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel_Demo, self).__init__()
        # The vocabulary size is set to cover all demographic categorical values plus two extra tokens.
        vocab_size = num_ages + num_genders + num_ethnicities + num_insurances + 2
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
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, feature]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        lab_embedding = x.mean(dim=1)
        return lab_embedding

class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, BEHRT, behrt_lab, device, hidden_size=512):
        super(MultimodalTransformer, self).__init__()
        # Note: We use the name BEHRT to align with keys in the checkpoint.
        self.BEHRT = BEHRT  
        self.behrt_lab = behrt_lab
        self.device = device

        self.demo_projector = nn.Sequential(
            nn.Linear(self.BEHRT.bert.config.hidden_size, 256),
            nn.ReLU()
        )
        self.lab_projector = nn.Sequential(
            nn.Linear(self.behrt_lab.hidden_size, 256),
            nn.ReLU()
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_size, 256),
            nn.ReLU()
        )

        self.sig_weights_demo = nn.Parameter(torch.randn(256))
        self.sig_weights_lab  = nn.Parameter(torch.randn(256))
        self.sig_weights_text = nn.Parameter(torch.randn(256))

        self.aggregate_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

    def forward(self, demo_dummy_ids, demo_attn_mask, 
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding):
        demo_embedding = self.BEHRT(demo_dummy_ids, demo_attn_mask,
                                    age_ids, gender_ids, ethnicity_ids, insurance_ids)
        lab_embedding = self.behrt_lab(lab_features)
        text_embedding = aggregated_text_embedding

        demo_proj = self.demo_projector(demo_embedding)  
        lab_proj  = self.lab_projector(lab_embedding)      
        text_proj = self.text_projector(text_embedding)    

        weighted_demo = demo_proj * torch.sigmoid(self.sig_weights_demo)
        weighted_lab  = lab_proj  * torch.sigmoid(self.sig_weights_lab)
        weighted_text = text_proj * torch.sigmoid(self.sig_weights_text)

        concatenated = torch.cat((weighted_demo, weighted_lab, weighted_text), dim=1)
        aggregated = self.aggregate_projector(concatenated)  
        logits = self.classifier(aggregated)  

        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits       = logits[:, 1].unsqueeze(1)
        mech_logits      = logits[:, 2].unsqueeze(1)
        return mortality_logits, los_logits, mech_logits, aggregated


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
            agg_emb = np.mean(embeddings, axis=0) if aggregation == "mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings

def evaluate_model(model, dataloader, device, threshold=0.5, print_eddi=False):
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_mech_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_labels_mech = []
    all_final_embeddings = []
    all_age = []
    all_ethnicity = []
    all_insurance = []
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_mech) = [x.to(device) for x in batch]
            mort_logits, los_logits, mech_logits, final_embedding = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_mech_logits.append(mech_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_labels_mech.append(labels_mech.cpu())
            all_final_embeddings.append(final_embedding.cpu())
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
    outcomes = {
        "mortality": (mort_probs, labels_mort_np),
        "los": (los_probs, labels_los_np),
        "mechanical_ventilation": (mech_probs, labels_mech_np)
    }
    
    eddi_stats = {}
    fairness_stats = {}
    for outcome, (probs, labels) in outcomes.items():
        try:
            aucroc = roc_auc_score(labels, probs)
        except Exception:
            aucroc = float('nan')
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        precision_val = precision_score(labels, preds, zero_division=0)
        
        cm = confusion_matrix(labels, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tpr = 0
            fpr = 0

        metrics[outcome] = {
            "aucroc": aucroc, 
            "f1": f1, 
            "tpr": tpr,
            "precision": precision_val, 
            "fpr": fpr
        }
        
        eddi_age, age_eddi_sub = compute_eddi(labels, preds, 
                            np.array([get_age_bucket(a) for a in torch.cat(all_age, dim=0).numpy().squeeze()]))
        eddi_eth, eth_eddi_sub = compute_eddi(labels, preds, 
                            np.array([map_ethnicity(e) for e in torch.cat(all_ethnicity, dim=0).numpy().squeeze()]))
        eddi_ins, ins_eddi_sub = compute_eddi(labels, preds, 
                            np.array([map_insurance(i) for i in torch.cat(all_insurance, dim=0).numpy().squeeze()]))
        age_order = ["15-29", "30-49", "50-69", "70-90", "Other"]
        ethnicity_order = ["White", "Black", "Hispanic", "Asian", "Other"]
        insurance_order = ["Government", "Medicare", "Medicaid", "Private", "Self Pay", "Other"]
        age_scores = [age_eddi_sub.get(bucket, 0) for bucket in age_order]
        eth_scores = [eth_eddi_sub.get(group, 0) for group in ethnicity_order]
        ins_scores = [ins_eddi_sub.get(group, 0) for group in insurance_order]
        overall_age = np.sqrt(np.sum(np.square(age_scores))) / len(age_order)
        overall_eth = np.sqrt(np.sum(np.square(eth_scores))) / len(ethnicity_order)
        overall_ins = np.sqrt(np.sum(np.square(ins_scores))) / len(insurance_order)
        total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
        
        eddi_stats[outcome] = {
            "age_subgroup_eddi": age_eddi_sub,
            "age_eddi": overall_age,
            "ethnicity_subgroup_eddi": eth_eddi_sub,
            "ethnicity_eddi": overall_eth,
            "insurance_subgroup_eddi": ins_eddi_sub,
            "insurance_eddi": overall_ins,
            "final_EDDI": total_eddi
        }
        
        age_groups = np.array([get_age_bucket(a) for a in torch.cat(all_age, dim=0).numpy().squeeze()])
        eth_groups = np.array([map_ethnicity(e) for e in torch.cat(all_ethnicity, dim=0).numpy().squeeze()])
        ins_groups = np.array([map_insurance(i) for i in torch.cat(all_insurance, dim=0).numpy().squeeze()])
        
        eo_age, details_age = calculate_equalized_opportunity(labels, preds, age_groups)
        eo_eth, details_eth = calculate_equalized_opportunity(labels, preds, eth_groups)
        eo_ins, details_ins = calculate_equalized_opportunity(labels, preds, ins_groups)
        
        fairness_stats[outcome] = {
            "Equalized Opportunity (Age)": eo_age,
            "EO Details (Age)": details_age,
            "Equalized Opportunity (Ethnicity)": eo_eth,
            "EO Details (Ethnicity)": details_eth,
            "Equalized Opportunity (Insurance)": eo_ins,
            "EO Details (Insurance)": details_ins,
            "Overall EO": (eo_age + eo_eth + eo_ins) / 3
        }
    
    metrics["eddi_stats"] = eddi_stats
    metrics["fairness_stats"] = fairness_stats
    metrics["eddi_stats"]["overall"] = {
        "mortality": eddi_stats["mortality"].get("final_EDDI", np.nan),
        "los": eddi_stats["los"].get("final_EDDI", np.nan),
        "mechanical_ventilation": eddi_stats["mechanical_ventilation"].get("final_EDDI", np.nan)
    }
    
    if print_eddi:
        for outcome in outcomes.keys():
            print(f"\n--- EDDI Calculation for {outcome.capitalize()} Outcome ---")
            print("Age subgroup EDDI:")
            for bucket in age_order:
                print(f"  {bucket}: {eddi_stats[outcome]['age_subgroup_eddi'].get(bucket, np.nan):.4f}")
            print("Overall Age EDDI:", eddi_stats[outcome]["age_eddi"])
            print("Ethnicity subgroup EDDI:")
            for group in ethnicity_order:
                print(f"  {group}: {eddi_stats[outcome]['ethnicity_subgroup_eddi'].get(group, np.nan):.4f}")
            print("Overall Ethnicity EDDI:", eddi_stats[outcome]["ethnicity_eddi"])
            print("Insurance subgroup EDDI:")
            for group in insurance_order:
                print(f"  {group}: {eddi_stats[outcome]['insurance_subgroup_eddi'].get(group, np.nan):.4f}")
            print("Overall Insurance EDDI:", eddi_stats[outcome]["insurance_eddi"])
            print("Final Overall {} EDDI: {:.4f}".format(outcome.capitalize(), eddi_stats[outcome]["final_EDDI"]))
            
            fs = fairness_stats[outcome]
            print(f"\n--- Fairness (Equalized Opportunity) for {outcome.capitalize()} Outcome ---")
            print("Equalized Opportunity (Age): {:.4f} with details {}".format(fs["Equalized Opportunity (Age)"], fs["EO Details (Age)"]))
            print("Equalized Opportunity (Ethnicity): {:.4f} with details {}".format(fs["Equalized Opportunity (Ethnicity)"], fs["EO Details (Ethnicity)"]))
            print("Equalized Opportunity (Insurance): {:.4f} with details {}".format(fs["Equalized Opportunity (Insurance)"], fs["EO Details (Insurance)"]))
            print("Combined Overall Equalized Opportunity: {:.4f}".format(fs["Overall EO"]))
    
    return metrics

def train_step(model, dataloader, optimizer, device, criterion_mortality, criterion_los, criterion_mech):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features,
         aggregated_text_embedding,
         labels_mortality, labels_los, labels_mech) = [x.to(device) for x in batch]

        optimizer.zero_grad()
        mortality_logits, los_logits, mech_logits, _ = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding
        )
        loss_mort = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
        loss_los = criterion_los(los_logits, labels_los.unsqueeze(1))
        loss_mech = criterion_mech(mech_logits, labels_mech.unsqueeze(1))
        loss = loss_mort + loss_los + loss_mech
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_step(model, dataloader, device, criterion_mortality, criterion_los, criterion_mech):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_mech) = [x.to(device) for x in batch]
            mortality_logits, los_logits, mech_logits, _ = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            loss_mort = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
            loss_los = criterion_los(los_logits, labels_los.unsqueeze(1))
            loss_mech = criterion_mech(mech_logits, labels_mech.unsqueeze(1))
            loss = loss_mort + loss_los + loss_mech
            running_loss += loss.item()
    return running_loss / len(dataloader)

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    structured_data = pd.read_csv('final_structured_common.csv', low_memory=False)
    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)
    
    unstructured_data.drop(
        columns=["short_term_mortality", "los_binary", "mechanical_ventilation", "age",
                 "GENDER", "ETHNICITY", "INSURANCE"],
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

    if "age" not in df_filtered.columns:
        if "Age" in df_filtered.columns:
            df_filtered.rename(columns={"Age": "age"}, inplace=True)
            print("Renamed column 'Age' to 'age'.")
        else:
            print("Column 'age' not found; creating default age=0 for all samples.")
            df_filtered["age"] = 0

    print("Computing aggregated text embeddings for each patient...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    print("Aggregated text embeddings shape:", aggregated_text_embeddings_np.shape)
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    demographics_cols = ["age", "GENDER", "ETHNICITY", "INSURANCE"]
    for col in demographics_cols:
        if col not in df_filtered.columns:
            print(f"Column {col} not found; creating default values.")
            df_filtered[col] = 0
        elif df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].astype("category").cat.codes

    exclude_cols = set(["subject_id", "ROW_ID", "hadm_id", "ICUSTAY_ID", "DBSOURCE", "FIRST_CAREUNIT",
                        "LAST_CAREUNIT", "FIRST_WARDID", "LAST_WARDID", "INTIME", "OUTTIME", "LOS",
                        "ADMITTIME", "DISCHTIME", "DEATHTIME", "GENDER", "ETHNICITY", "INSURANCE",
                        "DOB", "short_term_mortality", "los_binary", "mechanical_ventilation", "age"])
    lab_feature_columns = [col for col in df_filtered.columns 
                           if col not in exclude_cols and not col.startswith("note_") 
                           and pd.api.types.is_numeric_dtype(df_filtered[col])]
    print("Number of lab feature columns:", len(lab_feature_columns))
    df_filtered[lab_feature_columns] = df_filtered[lab_feature_columns].fillna(0)

    lab_features_np = df_filtered[lab_feature_columns].values.astype(np.float32)
    lab_mean = np.mean(lab_features_np, axis=0)
    lab_std = np.std(lab_features_np, axis=0)
    lab_features_np = (lab_features_np - lab_mean) / (lab_std + 1e-6)
    
    train_val_df, test_df = train_test_split(df_filtered, test_size=0.20, random_state=42, stratify=df_filtered["short_term_mortality"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.05, random_state=42, stratify=train_val_df["short_term_mortality"])
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    def build_dataset(df, aggregated_embeddings):
        num_samples = len(df)
        demo_dummy_ids = torch.zeros((num_samples, 1), dtype=torch.long)
        demo_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
        age_ids = torch.tensor(df["age"].values, dtype=torch.long)
        gender_ids = torch.tensor(df["GENDER"].values, dtype=torch.long)
        ethnicity_ids = torch.tensor(df["ETHNICITY"].values, dtype=torch.long)
        insurance_ids = torch.tensor(df["INSURANCE"].values, dtype=torch.long)
        lab_features_t = torch.tensor(lab_features_np[df.index], dtype=torch.float32)
        aggregated_text_embedding = torch.tensor(aggregated_embeddings[df["subject_id"].unique().argsort()], dtype=torch.float32)
        labels_mortality = torch.tensor(df["short_term_mortality"].values, dtype=torch.float32)
        labels_los = torch.tensor(df["los_binary"].values, dtype=torch.float32)
        labels_mech = torch.tensor(df["mechanical_ventilation"].values, dtype=torch.float32)
        dataset = TensorDataset(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features_t,
            aggregated_text_embedding,
            labels_mortality, labels_los, labels_mech
        )
        return dataset

    train_dataset = build_dataset(train_df, aggregated_text_embeddings_np)
    val_dataset = build_dataset(val_df, aggregated_text_embeddings_np)
    test_dataset = build_dataset(test_df, aggregated_text_embeddings_np)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

    # Instantiate the BEHRT branch for demographics using the new name.
    BEHRT = BEHRTModel_Demo(
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
        BEHRT=BEHRT,
        behrt_lab=behrt_lab,
        device=device,
        hidden_size=512
    ).to(device)

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_binary"], device)
    mech_pos_weight = get_pos_weight(df_filtered["mechanical_ventilation"], device)
    
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')
    criterion_mech = FocalLoss(gamma=1, pos_weight=mech_pos_weight, reduction='mean')

    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_step(multimodal_model, train_loader, optimizer, device,
                                criterion_mortality, criterion_los, criterion_mech)
        val_loss = validate_step(multimodal_model, val_loader, device,
                                 criterion_mortality, criterion_los, criterion_mech)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(multimodal_model.state_dict(), "best_multimodal_model.pt")
            print("Best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered. No improvement for 5 consecutive epochs.")
                break

    multimodal_model.load_state_dict(torch.load("best_multimodal_model.pt"))
    test_metrics = evaluate_model(multimodal_model, test_loader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation on Test Set:")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = test_metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, F1: {m['f1']:.4f}, "
              f"TPR: {m['tpr']:.4f}, Precision: {m['precision']:.4f}, FPR: {m['fpr']:.4f}")
    
    print("\nFinal Detailed EDDI Statistics on Test Set:")
    eddi_stats = test_metrics["eddi_stats"]
    print("\nMortality EDDI Stats:")
    print("  Age subgroup EDDI      :", eddi_stats["mortality"]["age_subgroup_eddi"])
    print("  Aggregated Age EDDI      : {:.4f}".format(eddi_stats["mortality"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI  :", eddi_stats["mortality"]["ethnicity_subgroup_eddi"])
    print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi_stats["mortality"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI  :", eddi_stats["mortality"]["insurance_subgroup_eddi"])
    print("  Aggregated Insurance EDDI: {:.4f}".format(eddi_stats["mortality"]["insurance_eddi"]))
    print("  Final Overall Mortality EDDI: {:.4f}".format(eddi_stats["mortality"]["final_EDDI"]))

    print("\nLOS EDDI Stats:")
    print("  Age subgroup EDDI      :", eddi_stats["los"]["age_subgroup_eddi"])
    print("  Aggregated Age EDDI      : {:.4f}".format(eddi_stats["los"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI  :", eddi_stats["los"]["ethnicity_subgroup_eddi"])
    print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi_stats["los"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI  :", eddi_stats["los"]["insurance_subgroup_eddi"])
    print("  Aggregated Insurance EDDI: {:.4f}".format(eddi_stats["los"]["insurance_eddi"]))
    print("  Final Overall LOS EDDI: {:.4f}".format(eddi_stats["los"]["final_EDDI"]))

    print("\nMechanical Ventilation EDDI Stats:")
    print("  Age subgroup EDDI      :", eddi_stats["mechanical_ventilation"]["age_subgroup_eddi"])
    print("  Aggregated Age EDDI      : {:.4f}".format(eddi_stats["mechanical_ventilation"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI  :", eddi_stats["mechanical_ventilation"]["ethnicity_subgroup_eddi"])
    print("  Aggregated Ethnicity EDDI: {:.4f}".format(eddi_stats["mechanical_ventilation"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI  :", eddi_stats["mechanical_ventilation"]["insurance_subgroup_eddi"])
    print("  Aggregated Insurance EDDI: {:.4f}".format(eddi_stats["mechanical_ventilation"]["insurance_eddi"]))
    print("  Final Overall Mechanical Ventilation EDDI: {:.4f}".format(eddi_stats["mechanical_ventilation"]["final_EDDI"]))

    print("\nOverall EDDI Summary:")
    print(test_metrics["eddi_stats"]["overall"])
    print("\nOverall Equalized Opportunity (Fairness) Summary:")
    fairness_stats = test_metrics["fairness_stats"]
    for outcome in fairness_stats.keys():
        fs = fairness_stats[outcome]
        print(f"\nFor outcome {outcome.capitalize()}:")
        print("  Equalized Opportunity (Age): {:.4f}".format(fs["Equalized Opportunity (Age)"]))
        print("  Equalized Opportunity (Ethnicity): {:.4f}".format(fs["Equalized Opportunity (Ethnicity)"]))
        print("  Equalized Opportunity (Insurance): {:.4f}".format(fs["Equalized Opportunity (Insurance)"]))
        print("  Combined Overall EO: {:.4f}".format(fs["Overall EO"]))
    
    print("Training and testing complete.")

if __name__ == "__main__":
    train_pipeline()
