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
    """
    A fine-tuning wrapper for BioClinicalBERT.
    Returns the CLS token embedding.
    """
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        #if DEBUG:
        #    print("BioClinicalBERT CLS embedding shape:", cls_embedding.shape)
        return cls_embedding

def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean"):
    """
    For each unique patient (by subject_id), extracts all non-null notes from the given note columns,
    tokenizes them, computes the CLS embedding via BioClinicalBERT_FT,
    and aggregates them (using mean or max).
    Returns an array of shape (num_patients, hidden_size).
    """
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
# 3a. Demographics Branch
class BEHRTModel_Demo(nn.Module):
    def __init__(self, num_ages, num_genders, num_ethnicities, num_insurances, hidden_size=768):
        """
        A BEHRT-like model for demographic features.
        """
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

# 3b. Lab Features Branch
class BEHRTModel_Lab(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768, nhead=8, num_layers=2):
        """
        A BEHRT-like model for lab features.
        Each lab feature (a scalar) is projected and then passed through a Transformer encoder.
        """
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


# 4. EDDI Enhancement Layer
class EDDIEnhancementLayer(nn.Module):
    """
    Applies a sigmoid to the input and multiplies element‑wise by a learnable weight vector.
    Initialized to ones so the initial enhancement is near identity.
    """
    def __init__(self, input_dim):
        super(EDDIEnhancementLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.eddi_weight = nn.Parameter(torch.ones(input_dim))  

    def forward(self, x):
        x_sigmoid = self.sigmoid(x)
        enhanced = x_sigmoid * self.eddi_weight
        return enhanced


# 5. Multimodal Transformer: Fusion of Branches

# 5. Multimodal Transformer: Fusion of Branches
class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed_size, behrt_demo, behrt_lab, device, hidden_size=512):
        """
        Combines three branches:
          - Demographics (via BEHRTModel_Demo)
          - Lab features (via BEHRTModel_Lab)
          - Text (aggregated BioClinicalBERT embedding)
        Each branch is projected to 256 dimensions, passed through an EDDI enhancement layer.
        Then, we compute a weight for each modality as:
           weight = 0.333 + (maxEDDI - modality_EDDI)
        where modality_EDDI is computed as the dot product between the projection and its EDDI enhancement.
        The modality embedding is re‐weighted by multiplying by this computed weight, a new dot product is computed,
        and the three scalars (one per branch) are concatenated and fed to a classifier.
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

        self.eddi_demo = EDDIEnhancementLayer(256)
        self.eddi_lab = EDDIEnhancementLayer(256)
        self.eddi_text = EDDIEnhancementLayer(256)

        self.classifier = nn.Sequential(
            nn.Linear(3, hidden_size),
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

        # Projection
        demo_proj = self.demo_projector(demo_embedding)
        lab_proj = self.lab_projector(lab_embedding)
        text_proj = self.text_projector(text_embedding)

        # EDDI enhancement (original)
        demo_enhanced = self.eddi_demo(demo_proj)
        lab_enhanced = self.eddi_lab(lab_proj)
        text_enhanced = self.eddi_text(text_proj)

        # Compute original dot products (EDDI scores) per modality
        demo_dot = torch.sum(demo_proj * demo_enhanced, dim=1, keepdim=True)
        lab_dot = torch.sum(lab_proj * lab_enhanced, dim=1, keepdim=True)
        text_dot = torch.sum(text_proj * text_enhanced, dim=1, keepdim=True)

        # Compute maxEDDI per sample over the three modalities
        stacked = torch.cat((demo_dot, lab_dot, text_dot), dim=1)
        maxEDDI, _ = torch.max(stacked, dim=1, keepdim=True)

        # Compute new weights for each modality: weight = 0.333 + (maxEDDI - modality_EDDI)
        weight_demo = 0.333 + (maxEDDI - demo_dot)
        weight_lab = 0.333 + (maxEDDI - lab_dot)
        weight_text = 0.333 + (maxEDDI - text_dot)

        # Re-weight the projected embeddings with the computed weights
        weighted_demo = demo_proj * weight_demo
        weighted_lab = lab_proj * weight_lab
        weighted_text = text_proj * weight_text

        # Recompute dot products from the weighted embeddings
        new_demo_dot = torch.sum(weighted_demo, dim=1, keepdim=True)
        new_lab_dot = torch.sum(weighted_lab, dim=1, keepdim=True)
        new_text_dot = torch.sum(weighted_text, dim=1, keepdim=True)
        
        # Concatenate the three new dot products and classify
        concatenated = torch.cat((new_demo_dot, new_lab_dot, new_text_dot), dim=1)
        logits = self.classifier(concatenated)
        mortality_logits = logits[:, 0].unsqueeze(1)
        readmission_logits = logits[:, 1].unsqueeze(1)
        
        # Return extra values for later analysis (e.g. demo_dot and demographic info)
        return (mortality_logits, readmission_logits,
                weight_demo, weight_lab, weight_text,
                demo_dot, lab_dot, text_dot)


# 6. Training and Evaluation Functions
def train_step(model, dataloader, optimizer, device, criterion_mortality, criterion_readmission, beta=0.3):
    """
    Training step with loss = L_ce + beta * EDDI.
    L_ce is the sum of the focal losses for mortality and readmission.
    EDDI is computed as the average of the original dot products (one per modality).
    
    beta: Hyperparameter to control the contribution of the EDDI term.
    """
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        (demo_dummy_ids, demo_attn_mask,
         age_ids, gender_ids, ethnicity_ids, insurance_ids,
         lab_features,
         aggregated_text_embedding,
         labels_mortality, labels_readmission) = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        
        # Forward pass: we expect the model to return extra outputs including the original dot products.
        mortality_logits, readmission_logits, weight_demo, weight_lab, weight_text, demo_dot, lab_dot, text_dot = model(
            demo_dummy_ids, demo_attn_mask,
            age_ids, gender_ids, ethnicity_ids, insurance_ids,
            lab_features, aggregated_text_embedding
        )
        
        # Compute the standard classification loss (L_ce)
        loss_mortality = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
        loss_readmission = criterion_readmission(readmission_logits, labels_readmission.unsqueeze(1))
        ce_loss = loss_mortality + loss_readmission
        
        # Compute EDDI as the average of the original dot products
        eddi_value = (demo_dot.mean() + lab_dot.mean() + text_dot.mean()) / 3.0
        
        # Total loss = classification loss + beta * EDDI
        loss = ce_loss + beta * eddi_value
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate_model(model, dataloader, device, threshold=0.5, df_filtered=None):
    model.eval()
    all_mort_logits = []
    all_readm_logits = []
    all_labels_mort = []
    all_labels_readm = []
    
    # For EDDI analysis from demo branch:
    all_demo_eddi = []
    all_age = []
    all_ethnicity = []
    all_insurance = []
    
    with torch.no_grad():
        for batch in dataloader:
            (demo_dummy_ids, demo_attn_mask,
             age_ids, gender_ids, ethnicity_ids, insurance_ids,
             lab_features,
             aggregated_text_embedding,
             labels_mortality, labels_readmission) = [x.to(device) for x in batch]
            # Unpack extra outputs including demo_dot
            (mort_logits, readm_logits,
             weight_demo, weight_lab, weight_text,
             demo_dot, lab_dot, text_dot) = model(
                demo_dummy_ids, demo_attn_mask,
                age_ids, gender_ids, ethnicity_ids, insurance_ids,
                lab_features, aggregated_text_embedding
            )
            all_mort_logits.append(mort_logits.cpu())
            all_readm_logits.append(readm_logits.cpu())
            all_labels_mort.append(labels_mortality.cpu())
            all_labels_readm.append(labels_readmission.cpu())
            
            # Collect demo EDDI (demo_dot) and corresponding demographic info
            all_demo_eddi.append(demo_dot.cpu())
            all_age.append(age_ids.cpu())
            all_ethnicity.append(ethnicity_ids.cpu())
            all_insurance.append(insurance_ids.cpu())
            
    # Concatenate outputs across batches
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_readm_logits = torch.cat(all_readm_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_readm = torch.cat(all_labels_readm, dim=0)
    
    mort_probs = torch.sigmoid(all_mort_logits).numpy().squeeze()
    readm_probs = torch.sigmoid(all_readm_logits).numpy().squeeze()
    labels_mort_np = all_labels_mort.numpy().squeeze()
    labels_readm_np = all_labels_readm.numpy().squeeze()
    
    metrics = {}
    for task, probs, labels in zip(["mortality", "readmission"],
                                   [mort_probs, readm_probs],
                                   [labels_mort_np, labels_readm_np]):
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
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1, "recall": recall, "precision": precision}
    
    if df_filtered is not None:
        baseline_auprc_mortality = df_filtered["short_term_mortality"].mean()
        baseline_auprc_readmission = df_filtered["readmission_within_30_days"].mean()
        metrics["baseline_auprc"] = {"mortality": baseline_auprc_mortality, "readmission": baseline_auprc_readmission}
    
    # Compute and print EDDI stats for demo branch
    # Concatenate demographic info and demo EDDI scores
    demo_eddi_all = torch.cat(all_demo_eddi, dim=0).squeeze().numpy()  # shape (num_samples,)
    age_all = torch.cat(all_age, dim=0).numpy().squeeze()
    ethnicity_all = torch.cat(all_ethnicity, dim=0).numpy().squeeze()
    insurance_all = torch.cat(all_insurance, dim=0).numpy().squeeze()
    
    demo_stats = {}
    for feature_name, feature_vals in zip(["age", "ethnicity", "insurance"],
                                            [age_all, ethnicity_all, insurance_all]):
        df_demo = pd.DataFrame({feature_name: feature_vals, "eddi_demo": demo_eddi_all})
        group_means = df_demo.groupby(feature_name)["eddi_demo"].mean().to_dict()
        overall_mean = df_demo["eddi_demo"].mean()
        demo_stats[feature_name] = {"group_means": group_means, "overall_mean": overall_mean}
        print(f"EDDI stats for {feature_name}:")
        print("Group means:", group_means)
        print("Overall mean:", overall_mean)
    
    # Add EDDI stats to the returned metrics if desired
    metrics["eddi_demo_stats"] = demo_stats

    return metrics, all_mort_logits, all_readm_logits

# 7. Main Training Pipeline

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Merge Structured and Unstructured Data 
    structured_data = pd.read_csv('filtered_structured_output.csv')
    unstructured_data = pd.read_csv("filtered_unstructured.csv", low_memory=False)
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
    merged_df["readmission_within_30_days"] = merged_df["readmission_within_30_days"].astype(int)
    
    note_columns = [col for col in merged_df.columns if col.startswith("note_")]
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = merged_df[merged_df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))

    # Compute Aggregated Text Embeddings (Text Branch) 
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    aggregated_text_embeddings_np = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean"
    )
    aggregated_text_embeddings_t = torch.tensor(aggregated_text_embeddings_np, dtype=torch.float32)

    # Process Structured Data: Split into Demographics and Lab Features 
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
                        "DOB", "short_term_mortality", "current_admission_dischtime", "next_admission_icu_intime",
                        "readmission_within_30_days", "age"])
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
    labels_readmission = torch.tensor(df_filtered["readmission_within_30_days"].values, dtype=torch.float32)

    dataset = TensorDataset(
        demo_dummy_ids, demo_attn_mask,
        age_ids, gender_ids, ethnicity_ids, insurance_ids,
        lab_features_t,
        aggregated_text_embeddings_t,
        labels_mortality, labels_readmission
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
    readmission_pos_weight = get_pos_weight(df_filtered["readmission_within_30_days"], device)
    criterion_mortality = FocalLoss(gamma=2, pos_weight=mortality_pos_weight, reduction='mean')
    criterion_readmission = FocalLoss(gamma=2, pos_weight=readmission_pos_weight, reduction='mean')

    max_epochs = 50
    patience_limit = 5  # Stop if no improvement for 5 epochs

    beta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for beta_value in beta_values:
        print(f"\nTraining with beta = {beta_value}")
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            running_loss = train_step(multimodal_model, dataloader, optimizer, device,
                                      criterion_mortality, criterion_readmission, beta=beta_value)
            epoch_loss = running_loss / len(dataloader)
            print(f"[Beta {beta_value} | Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")

            scheduler.step(epoch_loss)
            metrics, _, _ = evaluate_model(multimodal_model, dataloader, device, threshold=0.5)
            print(f"Metrics at threshold=0.5 after epoch {epoch+1}: {metrics}")

            # Check for improvement; if no improvement for 'patience_limit' epochs, stop early.
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience_limit:
                print(f"No improvement for {patience_limit} consecutive epochs. Early stopping for beta = {beta_value}.")
                break

    print("Training complete.")

if __name__ == "__main__":
    train_pipeline()
