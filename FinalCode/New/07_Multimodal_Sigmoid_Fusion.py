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
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertModel, BertConfig, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

DEBUG = True

# 1. Loss Functions and Class Weight Computation
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

# 2. BioClinicalBERT Fine-Tuning and Note Aggregation
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
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
            agg_emb = np.mean(embeddings, axis=0) if aggregation == "mean" else np.max(embeddings, axis=0)
            aggregated_embeddings.append(agg_emb)
    aggregated_embeddings = np.vstack(aggregated_embeddings)
    return aggregated_embeddings

# 3. BEHRT Models for Structured Data
# Demographics Branch
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

# Lab Features Branch
class BEHRTModel_Lab(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768, nhead=8, num_layers=2):
        super(BEHRTModel_Lab, self).__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Linear(1, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(lab_token_count, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, lab_features):
        x = lab_features.unsqueeze(-1)  # [batch, tokens, 1]
        x = self.token_embedding(x)     # [batch, tokens, hidden_size]
        x = x + self.pos_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)  # [tokens, batch, hidden_size]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, tokens, hidden_size]
        lab_embedding = x.mean(dim=1)
        return lab_embedding

# 4. Multimodal Transformer with Single Fusion Weight and Concatenation
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device, hidden_size=512):
        """
        Combines three branches:
          - Demographics (via BEHRTModel_Demo)
          - Lab features (via BEHRTModel_Lab)
          - Text (aggregated BioClinicalBERT embedding)
        Each branch is projected to 256 dimensions. Their outputs are concatenated (resulting in 768 features)
        and then reduced via a fusion layer to 256 dimensions. Next, a single learnable weight vector (self.sig_weights)
        is applied elementwise after a sigmoid activation, and the weighted tensor is reduced to a scalar per sample.
        Finally, the scalar is passed through a classifier to predict mortality and length of stay.
        """
        super(MultimodalTransformer, self).__init__()
        self.behrt_demo = behrt_demo
        self.behrt_lab = behrt_lab
        self.device = device

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

        # Fusion layer to reduce concatenated projections (256*3 = 768) to 256 dimensions.
        self.fusion_layer = nn.Linear(768, 256)

        # Single learnable weight vector.
        self.sig_weights = nn.Parameter(torch.randn(256))

        # Classifier takes the fused scalar and outputs 2 logits (mortality and length of stay).
        self.classifier = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding):
        # Demographics branch
        demo_embedding = self.behrt_demo(demo_dummy_ids, demo_attn_mask,
                                         age_ids, gender_ids, ethnicity_ids, insurance_ids)
        # Lab branch
        lab_embedding = self.behrt_lab(lab_features)
        # Text branch
        text_embedding = aggregated_text_embedding

        # Project each branch.
        demo_proj = self.demo_projector(demo_embedding)  # [batch, 256]
        lab_proj = self.lab_projector(lab_embedding)       # [batch, 256]
        text_proj = self.text_projector(text_embedding)    # [batch, 256]

        # Concatenate the projections -> [batch, 768]
        data_concat = torch.cat((demo_proj, lab_proj, text_proj), dim=1)

        # Reduce to 256 dimensions.
        data_proj = self.fusion_layer(data_concat)  # [batch, 256]

        # Apply the single learnable weight vector.
        sig_weights = torch.sigmoid(self.sig_weights)  # [256]
        data_sigmoid = data_proj * sig_weights          # elementwise multiplication

        # Reduce to a single scalar per sample.
        fused_scalar = torch.sum(data_sigmoid, dim=1, keepdim=True)  # [batch, 1]

        # Classifier produces two logits: one for mortality and one for length of stay.
        logits = self.classifier(fused_scalar)
        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits = logits[:, 1].unsqueeze(1)
        return mortality_logits, los_logits, fused_scalar

# 5. Helper Functions for Fairness (EDDI) Calculation and Demographic Mapping
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
        return "other"

def map_ethnicity(code):
    mapping = {0: "white", 1: "black", 2: "asian", 3: "hispanic"}
    return mapping.get(code, "others")

def map_insurance(code):
    mapping = {
        0: "government",
        1: "medicare",
        2: "medicaid",
        3: "private",
        4: "self pay"
    }
    return mapping.get(code, "others")

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
    eddi_attr = np.sqrt(np.sum(np.array(list(subgroup_eddi.values())) ** 2)) / len(unique_groups)
    return eddi_attr, subgroup_eddi

# 6. Training and Evaluation Functions
def train_step(model, dataloader, optimizer, device, criterion_mortality, criterion_los):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features,
         aggregated_text_embedding,
         labels_mortality, labels_los) = [x.to(device) for x in batch]
        optimizer.zero_grad()
        mortality_logits, los_logits, _ = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding
        )
        loss_mortality = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
        loss_los = criterion_los(los_logits, labels_los.unsqueeze(1))
        loss = loss_mortality + loss_los
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate_model(model, dataloader, device, threshold=0.5, print_eddi=False):
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_final_embeddings = []  # fused scalar
    all_age = []
    all_ethnicity = []
    all_insurance = []
    
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels_mortality, labels_los) = [x.to(device) for x in batch]
            mort_logits, los_logits, final_embedding = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_los_logits.append(los_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_los.append(labels_los.cpu())
            all_final_embeddings.append(final_embedding.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
    
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_los_logits = torch.cat(all_los_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_los = torch.cat(all_labels_los, dim=0)
    final_embeddings = torch.cat(all_final_embeddings, dim=0).numpy() 
    ages = torch.cat(all_age, dim=0).numpy()
    ethnicities = torch.cat(all_ethnicity, dim=0).numpy()
    insurances = torch.cat(all_insurance, dim=0).numpy()
    
    mort_probs = torch.sigmoid(all_mort_logits).numpy().squeeze()
    los_probs = torch.sigmoid(all_los_logits).numpy().squeeze()
    labels_mort_np = all_labels_mort.numpy().squeeze()
    labels_los_np = all_labels_los.numpy().squeeze()
    
    metrics = {}
    for task, probs, labels in zip(["mortality", "los"],
                                   [mort_probs, los_probs],
                                   [labels_mort_np, labels_los_np]):
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
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall, "precision": precision}
    
    # EDDI Calculation for Each Outcome
    y_pred_mort = (mort_probs > threshold).astype(int)
    y_true_mort = labels_mort_np.astype(int)
    y_pred_los = (los_probs > threshold).astype(int)
    y_true_los = labels_los_np.astype(int)
    
    age_groups = np.array([get_age_bucket(a) for a in ages])
    ethnicity_groups = np.array([map_ethnicity(e) for e in ethnicities])
    insurance_groups = np.array([map_insurance(i) for i in insurances])
    
    eddi_age_mort, age_eddi_sub_mort = compute_eddi(y_true_mort, y_pred_mort, age_groups)
    eddi_eth_mort, eth_eddi_sub_mort = compute_eddi(y_true_mort, y_pred_mort, ethnicity_groups)
    eddi_ins_mort, ins_eddi_sub_mort = compute_eddi(y_true_mort, y_pred_mort, insurance_groups)
    total_eddi_mort = np.sqrt((eddi_age_mort**2 + eddi_eth_mort**2 + eddi_ins_mort**2)) / 3
    
    eddi_age_los, age_eddi_sub_los = compute_eddi(y_true_los, y_pred_los, age_groups)
    eddi_eth_los, eth_eddi_sub_los = compute_eddi(y_true_los, y_pred_los, ethnicity_groups)
    eddi_ins_los, ins_eddi_sub_los = compute_eddi(y_true_los, y_pred_los, insurance_groups)
    total_eddi_los = np.sqrt((eddi_age_los**2 + eddi_eth_los**2 + eddi_ins_los**2)) / 3
    
    if print_eddi:
        print("\n--- EDDI Calculation for Mortality Outcome ---")
        print("Age Buckets EDDI (mortality):")
        for bucket, score in age_eddi_sub_mort.items():
            print(f"  {bucket}: {score:.4f}")
        print("Overall Age EDDI (mortality):", eddi_age_mort)
        
        print("\nEthnicity Groups EDDI (mortality):")
        for group, score in eth_eddi_sub_mort.items():
            print(f"  {group}: {score:.4f}")
        print("Overall Ethnicity EDDI (mortality):", eddi_eth_mort)
        
        print("\nInsurance Groups EDDI (mortality):")
        for group, score in ins_eddi_sub_mort.items():
            print(f"  {group}: {score:.4f}")
        print("Overall Insurance EDDI (mortality):", eddi_ins_mort)
        
        print("\nTotal EDDI for Mortality:", total_eddi_mort)
        
        print("\n--- EDDI Calculation for LOS Outcome ---")
        print("Age Buckets EDDI (LOS):")
        for bucket, score in age_eddi_sub_los.items():
            print(f"  {bucket}: {score:.4f}")
        print("Overall Age EDDI (LOS):", eddi_age_los)
        
        print("\nEthnicity Groups EDDI (LOS):")
        for group, score in eth_eddi_sub_los.items():
            print(f"  {group}: {score:.4f}")
        print("Overall Ethnicity EDDI (LOS):", eddi_eth_los)
        
        print("\nInsurance Groups EDDI (LOS):")
        for group, score in ins_eddi_sub_los.items():
            print(f"  {group}: {score:.4f}")
        print("Overall Insurance EDDI (LOS):", eddi_ins_los)
        
        print("\nTotal EDDI for LOS:", total_eddi_los)
    
    eddi_stats = {
        "mortality": {
            "age_subgroup_eddi": age_eddi_sub_mort,
            "age_eddi": eddi_age_mort,
            "ethnicity_subgroup_eddi": eth_eddi_sub_mort,
            "ethnicity_eddi": eddi_eth_mort,
            "insurance_subgroup_eddi": ins_eddi_sub_mort,
            "insurance_eddi": eddi_ins_mort,
            "final_EDDI": total_eddi_mort
        },
        "los": {
            "age_subgroup_eddi": age_eddi_sub_los,
            "age_eddi": eddi_age_los,
            "ethnicity_subgroup_eddi": eth_eddi_sub_los,
            "ethnicity_eddi": eddi_eth_los,
            "insurance_subgroup_eddi": ins_eddi_sub_los,
            "insurance_eddi": eddi_ins_los,
            "final_EDDI": total_eddi_los
        }
    }
    
    metrics["eddi_stats"] = eddi_stats
    return metrics

# 7. Main Training and Evaluation Pipeline
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Merge Structured and Unstructured Data 
    structured_data = pd.read_csv('final_structured_common.csv')
    unstructured_data = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("\n--- Debug Info: Before Merge ---")
    print("Structured data shape:", structured_data.shape)
    print("Unstructured data shape:", unstructured_data.shape)
    
    unstructured_data.drop(
        columns=["short_term_mortality", "readmission_within_30_days", "age",
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
    merged_df["los_gt_3"] = merged_df["los_gt_3"].astype(int)  # Assuming LOS >3 indicator is available
    
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    # Compute Aggregated Text Embeddings (Text Branch)
    print("Computing aggregated text embeddings for each patient...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    print("Aggregated text embeddings shape:", aggregated_text_embeddings_np.shape)
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    # Process Structured Data: Demographics and Lab Features 
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
                        "DOB", "short_term_mortality", "los_gt_3", "current_admission_dischtime", "next_admission_icu_intime",
                        "age"])
    lab_feature_columns = [col for col in df_filtered.columns 
                           if col not in exclude_cols and not col.startswith("note_") 
                           and pd.api.types.is_numeric_dtype(df_filtered[col])]
    print("Number of lab feature columns:", len(lab_feature_columns))
    df_filtered[lab_feature_columns] = df_filtered[lab_feature_columns].fillna(0)

    # Normalize Lab Features 
    lab_features_np = df_filtered[lab_feature_columns].values.astype(np.float32)
    lab_mean = np.mean(lab_features_np, axis=0)
    lab_std = np.std(lab_features_np, axis=0)
    lab_features_np = (lab_features_np - lab_mean) / (lab_std + 1e-6)
    
    # Create Inputs for Each Branch
    num_samples = len(df_filtered)
    demo_dummy_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    demo_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    age_ids = torch.tensor(df_filtered["age"].values, dtype=torch.long)
    gender_ids = torch.tensor(df_filtered["GENDER"].values, dtype=torch.long)
    ethnicity_ids = torch.tensor(df_filtered["ETHNICITY"].values, dtype=torch.long)
    insurance_ids = torch.tensor(df_filtered["INSURANCE"].values, dtype=torch.long)
    lab_features_t = torch.tensor(lab_features_np, dtype=torch.float32)
    labels_mortality = torch.tensor(df_filtered["short_term_mortality"].values, dtype=torch.float32)
    labels_los = torch.tensor(df_filtered["los_gt_3"].values, dtype=torch.float32)
    
    dataset = TensorDataset(
        demo_dummy_ids, demo_attn_mask,
        age_ids, gender_ids, ethnicity_ids, insurance_ids,
        lab_features_t,
        aggregated_text_embeddings_t,
        labels_mortality, labels_los
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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

    multimodal_model = MultimodalTransformer(
        text_embed_size=768,
        behrt_demo=behrt_demo,
        behrt_lab=behrt_lab,
        device=device,
        hidden_size=512
    ).to(device)

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    mortality_pos_weight = get_pos_weight(df_filtered["short_term_mortality"], device)
    los_pos_weight = get_pos_weight(df_filtered["los_gt_3"], device)
    criterion_mortality = FocalLoss(gamma=1, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_los = FocalLoss(gamma=1, pos_weight=los_pos_weight, reduction='mean')

    num_epochs = 20
    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = train_step(multimodal_model, dataloader, optimizer, device,
                                  criterion_mortality, criterion_los)
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

    metrics = evaluate_model(multimodal_model, dataloader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation Metrics (including subgroup-level EDDI):")
    for outcome in ["mortality", "los"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
              f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")
    
    print("\nFinal Detailed EDDI Statistics:")
    eddi_stats = metrics["eddi_stats"]
    
    print("\nMortality EDDI Stats:")
    print("  Age subgroup EDDI      :", eddi_stats["mortality"]["age_subgroup_eddi"])
    print("  Aggregated age_eddi      : {:.4f}".format(eddi_stats["mortality"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI  :", eddi_stats["mortality"]["ethnicity_subgroup_eddi"])
    print("  Aggregated ethnicity_eddi: {:.4f}".format(eddi_stats["mortality"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI  :", eddi_stats["mortality"]["insurance_subgroup_eddi"])
    print("  Aggregated insurance_eddi: {:.4f}".format(eddi_stats["mortality"]["insurance_eddi"]))
    print("  Final Overall Mortality EDDI: {:.4f}".format(eddi_stats["mortality"]["final_EDDI"]))

    print("\nLOS EDDI Stats:")
    print("  Age subgroup EDDI      :", eddi_stats["los"]["age_subgroup_eddi"])
    print("  Aggregated age_eddi      : {:.4f}".format(eddi_stats["los"]["age_eddi"]))
    print("  Ethnicity subgroup EDDI  :", eddi_stats["los"]["ethnicity_subgroup_eddi"])
    print("  Aggregated ethnicity_eddi: {:.4f}".format(eddi_stats["los"]["ethnicity_eddi"]))
    print("  Insurance subgroup EDDI  :", eddi_stats["los"]["insurance_subgroup_eddi"])
    print("  Aggregated insurance_eddi: {:.4f}".format(eddi_stats["los"]["insurance_eddi"]))
    print("  Final Overall LOS EDDI: {:.4f}".format(eddi_stats["los"]["final_EDDI"]))

    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
