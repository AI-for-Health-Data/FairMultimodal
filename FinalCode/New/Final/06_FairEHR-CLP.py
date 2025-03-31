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
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModel, RobertaModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
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

# Synthetic Data Generation Functions

def generate_synthetic_notes(note):
    # Dummy synthetic clinical note generation.
    if isinstance(note, str) and note.strip():
        return note + " [SYN]"
    else:
        return ""

def generate_synthetic_demographics(demo_tensor):
    noise = torch.randn_like(demo_tensor) * 0.05
    return demo_tensor + noise

def generate_synthetic_longitudinal(long_tensor):
    noise = torch.randn_like(long_tensor) * 0.01
    return long_tensor + noise


# BioClinicalBERT Fine-Tuning Wrapper
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = base_model
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

# Function to Aggregate Clinical Note Embeddings
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

# Modality-Specific Encoders and Fusion Modules
class DemographicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DemographicEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class LongitudinalEncoder(nn.Module):
    def __init__(self, num_features, embed_dim, conv_out_channels, transformer_hidden, nhead, num_layers):
        super(LongitudinalEncoder, self).__init__()
        self.num_features = num_features
        self.feature_embedding = nn.Linear(1, embed_dim)
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=conv_out_channels, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out_channels, nhead=nhead, dim_feedforward=transformer_hidden)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(conv_out_channels, embed_dim)
    
    def forward(self, x):
        x = x.unsqueeze(2)  # (batch, num_features, 1)
        x = self.feature_embedding(x)  # (batch, num_features, embed_dim)
        conv_out = self.conv1d(x)  # (batch, num_features, conv_out_channels)
        transformer_in = conv_out.transpose(0, 1)  # (num_features, batch, conv_out_channels)
        transformer_out = self.transformer(transformer_in)
        out = transformer_out.mean(dim=0)  # (batch, conv_out_channels)
        out = self.proj(out)  # (batch, embed_dim)
        return out

class NotesEncoder(nn.Module):
    def __init__(self, model_name="roberta-large", output_dim=256):
        super(NotesEncoder, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.ReLU()
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls_emb)

class FusionModule(nn.Module):
    def __init__(self, input_dim, fusion_dim):
        super(FusionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class DynamicRelevance(nn.Module):
    def __init__(self, dim):
        super(DynamicRelevance, self).__init__()
        self.weights = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        gate = torch.sigmoid(self.weights)
        return gate * x

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.mlp(x)

# BEHRT Model for Structured Data
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

# Multimodal Transformer Model
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
        # Classifier for 3 outcomes: mortality, LOS, mechanical ventilation
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
        ts_proj = self.ts_projector(structured_emb)
        text_proj = self.text_projector(aggregated_text_embedding)
        combined = torch.cat((ts_proj, text_proj), dim=1)
        logits = self.classifier(combined)
        mortality_logits = logits[:, 0].unsqueeze(1)
        los_logits = logits[:, 1].unsqueeze(1)
        vent_logits = logits[:, 2].unsqueeze(1)
        return mortality_logits, los_logits, vent_logits

# FairEHR-CLP Model Definition
class FairEHR_CLP(nn.Module):
    def __init__(self, demo_input_dim=4, demo_hidden=128,
                 num_long_features=20, long_embed_dim=256, conv_out=256,
                 transformer_hidden=512, nhead=8, num_layers=2,
                 notes_model_name="roberta-large", notes_out=256,
                 fusion_dim=256, num_classes=2):
        super(FairEHR_CLP, self).__init__()
        self.demo_encoder = DemographicEncoder(demo_input_dim, demo_hidden)
        self.long_encoder = LongitudinalEncoder(num_long_features, embed_dim=long_embed_dim,
                                                conv_out_channels=conv_out,
                                                transformer_hidden=transformer_hidden,
                                                nhead=nhead, num_layers=num_layers)
        self.notes_encoder = NotesEncoder(model_name=notes_model_name, output_dim=notes_out)
        fusion_input_dim = demo_hidden + long_embed_dim + notes_out
        self.fusion = FusionModule(fusion_input_dim, fusion_dim)
        self.dr = DynamicRelevance(fusion_dim)
        self.classifier = Classifier(fusion_dim, num_classes)
    def forward(self,
                demo_real, long_real, notes_real_input_ids, notes_real_attention_mask,
                demo_syn, long_syn, notes_syn_input_ids, notes_syn_attention_mask):
        ed_real = self.demo_encoder(demo_real)
        ed_syn = self.demo_encoder(demo_syn)
        el_real = self.long_encoder(long_real)
        el_syn = self.long_encoder(long_syn)
        en_real = self.notes_encoder(notes_real_input_ids, notes_real_attention_mask)
        en_syn = self.notes_encoder(notes_syn_input_ids, notes_syn_attention_mask)
        fused_real = self.fusion(torch.cat([ed_real, el_real, en_real], dim=1))
        fused_syn  = self.fusion(torch.cat([ed_syn, el_syn, en_syn], dim=1))
        e_adj = self.dr(fused_real)
        e_adj_syn = self.dr(fused_syn)
        logits = self.classifier(e_adj)
        return logits, e_adj, e_adj_syn

# Contrastive Loss for Fairness-Aware Learning
def contrastive_loss(e_real, e_syn, tau=0.5, gamma=0.1):
    batch_size = e_real.size(0)
    e_real_norm = F.normalize(e_real, p=2, dim=1)
    e_syn_norm  = F.normalize(e_syn, p=2, dim=1)
    sim_matrix = torch.mm(e_real_norm, e_syn_norm.t())  # (batch, batch)
    sim_matrix = sim_matrix / tau
    positives = sim_matrix.diag()
    loss = 0.0
    for i in range(batch_size):
        numerator = torch.exp(positives[i])
        denominator = torch.exp(sim_matrix[i, :]).sum()
        loss += -torch.log(numerator / denominator)
    loss = loss / batch_size
    mean_syn = e_syn.mean(dim=0, keepdim=True)
    reg = torch.mean((e_syn - mean_syn).pow(2))
    return loss + gamma * reg


# Evaluation Metrics Function for TensorDataset
def evaluate_model_metrics(model, dataloader, device, threshold=0.5, print_eddi=False):
    model.eval()
    all_mort_logits = []
    all_los_logits = []
    all_vent_logits = []
    all_labels_mort = []
    all_labels_los = []
    all_labels_vent = []
    all_sensitive = []  # using age as sensitive attribute
    with torch.no_grad():
        for batch in dataloader:
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
            mort_logits, los_logits, vent_logits = model(
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
            all_sensitive.append(age_ids.cpu())
    all_mort_logits = torch.cat(all_mort_logits, dim=0)
    all_los_logits = torch.cat(all_los_logits, dim=0)
    all_vent_logits = torch.cat(all_vent_logits, dim=0)
    all_labels_mort = torch.cat(all_labels_mort, dim=0)
    all_labels_los = torch.cat(all_labels_los, dim=0)
    all_labels_vent = torch.cat(all_labels_vent, dim=0)
    all_sensitive = torch.cat(all_sensitive, dim=0).numpy()

    mort_probs = torch.sigmoid(all_mort_logits).numpy().squeeze()
    los_probs  = torch.sigmoid(all_los_logits).numpy().squeeze()
    vent_probs = torch.sigmoid(all_vent_logits).numpy().squeeze()
    labels_mort_np = all_labels_mort.numpy().squeeze()
    labels_los_np  = all_labels_los.numpy().squeeze()
    labels_vent_np = all_labels_vent.numpy().squeeze()
    
    tasks = ["mortality", "los", "mechanical_ventilation"]
    probs_list = [mort_probs, los_probs, vent_probs]
    labels_list = [labels_mort_np, labels_los_np, labels_vent_np]
    metrics = {}
    eddi_stats = {}
    for task, probs, labels in zip(tasks, probs_list, labels_list):
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
        TP = ((preds == 1) & (labels == 1)).sum()
        FP = ((preds == 1) & (labels == 0)).sum()
        TN = ((preds == 0) & (labels == 0)).sum()
        FN = ((preds == 0) & (labels == 1)).sum()
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        metrics[task] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                         "recall": recall, "precision": precision, "tpr": TPR, "fpr": FPR}
        # Compute EDDI for each task
        sensitive_buckets = np.array([get_age_bucket(a) for a in all_sensitive])
        overall_eddi, subgroup_eddi = compute_eddi(labels.astype(int), probs, sensitive_buckets, threshold)
        eddi_stats[task] = {"age_eddi": overall_eddi, "age_subgroup_eddi": subgroup_eddi}
    if print_eddi:
        metrics["eddi_stats"] = eddi_stats
        print("\n--- EDDI Calculation ---")
        for task in tasks:
            stats = eddi_stats[task]
            print(f"\nTask: {task.capitalize()}")
            print("  Aggregated Age EDDI    : {:.4f}".format(stats["age_eddi"]))
            print("  Age Subgroup EDDI:")
            for bucket, score in stats["age_subgroup_eddi"].items():
                print(f"    {bucket}: {score:.4f}")
    return metrics

# Evaluation Loss Function for TensorDataset
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
            mort_logits, los_logits, vent_logits = model(
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

# Training Pipeline
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    structured_data = pd.read_csv('final_structured_common.csv')
    keep_cols = {"subject_id", "hadm_id", "short_term_mortality", "los_binary", 
                 "mechanical_ventilation", "age", "FIRST_WARDID", "LAST_WARDID", "ETHNICITY", "INSURANCE", "GENDER"}
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
        print("Column 'age' not found; creating default 'age' column with zeros.")
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
            print(f"Column {col} not found; creating default values.")
            df_filtered[col] = 0
    
    df_unique = df_filtered.groupby("subject_id", as_index=False).first()
    print("Number of unique patients:", len(df_unique))
    if "segment" not in df_unique.columns:
        df_unique["segment"] = 0

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
    
    num_samples = len(df_unique)
    dummy_input_ids = torch.zeros((num_samples, 1), dtype=torch.long)
    dummy_attn_mask = torch.ones((num_samples, 1), dtype=torch.long)
    
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
    
    # Create TensorDataset with 13 items per sample.
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
    
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    train_idx_rel, val_idx_rel = next(msss_val.split(np.zeros(len(train_val_idx)), labels_array[train_val_idx]))
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
    
    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    best_model_path = "best_multimodal_model.pt"
    
    for epoch in range(num_epochs):
        multimodal_model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            (dummy_input_ids, dummy_attn_mask,
             age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids,
             aggregated_text_embedding,
             labels_mortality, labels_los, labels_vent) = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            mortality_logits, los_logits, vent_logits = multimodal_model(
                dummy_input_ids, dummy_attn_mask,
                age_ids, segment_ids, adm_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids,
                aggregated_text_embedding
            )
            loss_mort = criterion_mortality(mortality_logits, labels_mortality.unsqueeze(1))
            loss_los = criterion_los(los_logits, labels_los.unsqueeze(1))
            loss_vent = criterion_mech(vent_logits, labels_vent.unsqueeze(1))
            loss = loss_mort + loss_los + loss_vent
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        val_loss = evaluate_model_loss(multimodal_model, val_loader, device,
                                       criterion_mortality, criterion_los, criterion_mech)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        
        val_metrics = evaluate_model_metrics(multimodal_model, val_loader, device, threshold=0.5)
        print(f"--- Validation Metrics at Epoch {epoch+1} ---")
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            m = val_metrics[outcome]
            print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
                  f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}, "
                  f"TPR: {m['tpr']:.4f}, FPR: {m['fpr']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(multimodal_model.state_dict(), best_model_path)
            print("  Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
    
    multimodal_model.load_state_dict(torch.load(best_model_path))
    print("\nEvaluating on test set...")
    metrics = evaluate_model_metrics(multimodal_model, test_loader, device, threshold=0.5, print_eddi=True)
    print("\nFinal Evaluation Metrics on Test Set:")
    for outcome in ["mortality", "los", "mechanical_ventilation"]:
        m = metrics[outcome]
        print(f"{outcome.capitalize()} - AUC-ROC: {m['aucroc']:.4f}, AUPRC: {m['auprc']:.4f}, "
              f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}, "
              f"TPR: {m['tpr']:.4f}, FPR: {m['fpr']:.4f}")
    
    if "eddi_stats" in metrics:
        print("\nDetailed EDDI Statistics on Test Set:")
        eddi_stats = metrics["eddi_stats"]
        for outcome in ["mortality", "los", "mechanical_ventilation"]:
            print(f"\n{outcome.capitalize()} EDDI Stats:")
            stats = eddi_stats[outcome]
            print("  Age subgroup EDDI      :", stats["age_subgroup_eddi"])
            print("  Aggregated Age EDDI    : {:.4f}".format(stats["age_eddi"]))
    print("Training complete.")

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
            mort_logits, los_logits, vent_logits = model(
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

if __name__ == "__main__":
    train_pipeline()
