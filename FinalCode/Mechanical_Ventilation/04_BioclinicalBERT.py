import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from scipy.special import expit  # logistic sigmoid

#########################
# Section 1: Mechanical Ventilation Flags Extraction and Dataset Merging
#########################

# --- 1.1 Extracting Ventilation Flags from CHARTEVENTS ---
print("Loading and processing CHARTEVENTS...")

chartevents = pd.read_csv('CHARTEVENTS.csv.gz', compression='gzip', low_memory=False,
                           usecols=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'ERROR'])
chartevents.columns = chartevents.columns.str.lower()

# Keep rows with non-null values and error not equal to 1 (or error is null)
chartevents = chartevents[chartevents['value'].notnull()]
chartevents = chartevents[(chartevents['error'] != 1) | (chartevents['error'].isnull())]

# Define itemids for ventilation settings (for both mechanical ventilation and oxygen devices)
vent_itemids = [
    720, 223848, 223849, 467,
    445, 448, 449, 450, 1340, 1486, 1600, 224687,
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
    218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747,
    221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
    543, 5865, 5866, 224707, 224709, 224705, 224706,
    60, 437, 505, 506, 686, 220339, 224700,
    3459,
    501, 502, 503, 224702,
    223, 667, 668, 669, 670, 671, 672,
    224701,
    # Oxygen device related itemids
    468, 469, 470, 471, 227287, 226732, 223834
]
chartevents = chartevents[chartevents['itemid'].isin(vent_itemids)]

def determine_flags(row):
    """Determine ventilation-related flags for a given row."""
    mechvent = 0
    oxygen = 0
    extubated = 0
    self_extubated = 0

    iv = row['itemid']
    val = row['value']

    # Mechanical Ventilation conditions:
    if iv == 720 and val != 'Other/Remarks':
        mechvent = 1
    if iv == 223848 and val != 'Other':
        mechvent = 1
    if iv == 223849:
        mechvent = 1
    if iv == 467 and val == 'Ventilator':
        mechvent = 1
    if iv in [445, 448, 449, 450, 1340, 1486, 1600, 224687,
              639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
              218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747,
              221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
              543, 5865, 5866, 224707, 224709, 224705, 224706,
              60, 437, 505, 506, 686, 220339, 224700,
              3459, 501, 502, 503, 224702,
              223, 667, 668, 669, 670, 671, 672, 224701]:
        mechvent = 1

    # Oxygen Therapy conditions:
    if iv == 226732 and val in ['Nasal cannula', 'Face tent', 'Aerosol-cool', 'Trach mask ', 
                                'High flow neb', 'Non-rebreather', 'Venti mask ', 'Medium conc mask ',
                                'T-piece', 'High flow nasal cannula', 'Ultrasonic neb', 'Vapomist']:
        oxygen = 1
    if iv == 467 and val in ['Cannula', 'Nasal Cannula', 'Face Tent', 'Aerosol-Cool', 'Trach Mask',
                              'Hi Flow Neb', 'Non-Rebreather', 'Venti Mask', 'Medium Conc Mask',
                              'Vapotherm', 'T-Piece', 'Hood', 'Hut', 'TranstrachealCat',
                              'Heated Neb', 'Ultrasonic Neb']:
        oxygen = 1

    # Extubation conditions:
    if iv == 640 and val in ['Extubated', 'Self Extubation']:
        extubated = 1
    if iv == 640 and val == 'Self Extubation':
        self_extubated = 1

    return pd.Series({
        'mechvent': mechvent,
        'oxygentherapy': oxygen,
        'extubated': extubated,
        'selfextubated': self_extubated
    })

vent_flags = chartevents.apply(determine_flags, axis=1)
chartevents = pd.concat([chartevents, vent_flags], axis=1)

# Aggregate flags per ICU stay and charttime
vent_chartevents = chartevents.groupby(['icustay_id', 'charttime'], as_index=False).agg({
    'mechvent': 'max',
    'oxygentherapy': 'max',
    'extubated': 'max',
    'selfextubated': 'max'
})

# --- 1.2 Process PROCEDUREEVENTS_MV for extubation events ---
print("Loading and processing PROCEDUREEVENTS_MV...")

proc_events = pd.read_csv('PROCEDUREEVENTS_MV.csv.gz', compression='gzip', low_memory=False,
                          usecols=['ICUSTAY_ID', 'STARTTIME', 'ITEMID'])
proc_events.columns = proc_events.columns.str.lower()
proc_events = proc_events[proc_events['itemid'].isin([227194, 225468, 225477])]
proc_events.rename(columns={'starttime': 'charttime'}, inplace=True)
proc_events['mechvent'] = 0
proc_events['oxygentherapy'] = 0
proc_events['extubated'] = 1
proc_events['selfextubated'] = proc_events['itemid'].apply(lambda x: 1 if x == 225468 else 0)
vent_proc = proc_events[['icustay_id', 'charttime', 'mechvent', 'oxygentherapy', 'extubated', 'selfextubated']].drop_duplicates()

# --- 1.3 Combine both sources ---
print("Combining ventilation flags from chart and procedure events...")

ventilation_flags = pd.concat([vent_chartevents, vent_proc], ignore_index=True).drop_duplicates(subset=['icustay_id', 'charttime'])

# Map ventilation flags to subject and admission IDs using ICUSTAYS
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
icu_stays.columns = icu_stays.columns.str.lower()
ventilation_flags = ventilation_flags.merge(
    icu_stays[['icustay_id', 'subject_id', 'hadm_id']],
    on='icustay_id',
    how='left'
)

# Aggregate ventilation flags per subject and admission (using max over all charttimes)
ventilation_flags_agg = ventilation_flags.groupby(['subject_id', 'hadm_id'], as_index=False).agg({
    'mechvent': 'max',
    'oxygentherapy': 'max',
    'extubated': 'max',
    'selfextubated': 'max'
})

# Create a single mechanical ventilation column:
ventilation_flags_agg['mechanical_ventilation'] = ventilation_flags_agg[['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']].max(axis=1)

# --- 1.4 Merge into Structured Dataset ---
print("Merging ventilation flags into structured dataset...")

structured_df = pd.read_csv('final_structured_common.csv')
structured_with_vent = structured_df.merge(ventilation_flags_agg[['subject_id', 'hadm_id', 'mechanical_ventilation']],
                                             on=['subject_id', 'hadm_id'], how='left')
structured_with_vent['mechanical_ventilation'] = structured_with_vent['mechanical_ventilation'].fillna(0).astype(int)

# Drop extra columns if they exist.
cols_to_drop = ['short_term_mortality', 'readmission_within_30_days']
for col in cols_to_drop:
    if col in structured_with_vent.columns:
        structured_with_vent.drop(columns=[col], inplace=True)

for col in ['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']:
    if col in structured_with_vent.columns:
        structured_with_vent.drop(columns=[col], inplace=True)

structured_outfile = 'final_structured_with_mechanical_ventilation.csv'
structured_with_vent.to_csv(structured_outfile, index=False)
print("Updated structured dataset saved as:", structured_outfile)

# --- 1.5 Merge into Unstructured Dataset ---
print("Merging ventilation flags into unstructured dataset...")

unstructured_df = pd.read_csv('final_unstructured_common.csv', engine='python', on_bad_lines='skip')
unstructured_with_vent = unstructured_df.merge(ventilation_flags_agg[['subject_id', 'hadm_id', 'mechanical_ventilation']],
                                               on=['subject_id', 'hadm_id'], how='left')
unstructured_with_vent['mechanical_ventilation'] = unstructured_with_vent['mechanical_ventilation'].fillna(0).astype(int)

for col in cols_to_drop:
    if col in unstructured_with_vent.columns:
        unstructured_with_vent.drop(columns=[col], inplace=True)

for col in ['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']:
    if col in unstructured_with_vent.columns:
        unstructured_with_vent.drop(columns=[col], inplace=True)

unstructured_outfile = 'final_unstructured_with_mechanical_ventilation.csv'
unstructured_with_vent.to_csv(unstructured_outfile, index=False)
print("Updated unstructured dataset saved as:", unstructured_outfile)

#########################
# Section 2: BioClinicalBERT Training Pipeline and Fairness Evaluation
#########################

# Loss and Helper Functions
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
    print("Positive weight:", weight.item())
    return weight

# BioClinicalBERT Fine-Tuning Wrapper
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, base_model, config, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.bert = base_model  
        self.device = device

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean", max_length=128):
    patient_ids = df["subject_id"].unique()
    aggregated_embeddings = []
    for pid in tqdm(patient_ids, desc="Aggregating text embeddings"):
        patient_data = df[df["subject_id"] == pid]
        notes = []
        for col in note_columns:
            vals = patient_data[col].dropna().tolist()
            notes.extend([v for v in vals if isinstance(v, str) and v.strip() != ""])
        if len(notes) == 0:
            aggregated_embeddings.append(np.zeros(model.bert.config.hidden_size))
        else:
            embeddings = []
            for note in notes:
                encoded = tokenizer.encode_plus(
                    text=note,
                    add_special_tokens=True,
                    max_length=max_length,
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
    return aggregated_embeddings, patient_ids

# Unstructured Dataset and Classifier
class UnstructuredDataset(Dataset):
    def __init__(self, embeddings, vent_labels):
        """
        embeddings: NumPy array of shape (num_patients, hidden_size)
        vent_labels: binary labels (0/1) for mechanical ventilation
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.vent_labels = torch.tensor(vent_labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.vent_labels[idx]

class UnstructuredClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super(UnstructuredClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)  # Output for mechanical ventilation
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits

def train_model(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        embeddings, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = [x.to(device) for x in batch]
            logits = model(embeddings)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = expit(all_logits)
    preds = (probs >= threshold).astype(int)
    metrics = {}
    try:
        aucroc = roc_auc_score(all_labels, probs)
    except Exception:
        aucroc = float('nan')
    try:
        auprc = average_precision_score(all_labels, probs)
    except Exception:
        auprc = float('nan')
    f1 = f1_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    precision = precision_score(all_labels, preds, zero_division=0)
    metrics["mechanical_ventilation"] = {"aucroc": aucroc, "auprc": auprc, "f1": f1,
                                          "recall": recall, "precision": precision}
    return metrics

def get_patient_probabilities(model, dataloader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, _ = [x.to(device) for x in batch]
            logits = model(embeddings)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    probs = expit(all_logits)
    return probs

# Fairness Helper Functions
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

def map_ethnicity(val):
    # Map numeric values or standardize strings
    if isinstance(val, (int, float)):
        mapping = {0: "white", 1: "black", 2: "asian", 3: "hispanic"}
        return mapping.get(val, "others")
    else:
        return val.strip().lower()

def map_insurance(val):
    # Map numeric values or standardize strings
    if isinstance(val, (int, float)):
        mapping = {0: "government", 1: "medicare", 2: "medicaid", 3: "private", 4: "self pay"}
        return mapping.get(val, "others")
    else:
        return val.strip().lower()

def compute_subgroup_eddi(df, sensitive_attr, true_label_col, pred_label_col):
    """
    Compute subgroup-level EDDI for each class in a sensitive attribute.
    Disparity = |group_error - overall_error| / max(overall_error, 1 - overall_error)
    """
    groups = df.groupby(sensitive_attr)
    overall_error = np.mean(df[true_label_col] != df[pred_label_col])
    subgroup_eddi = {}
    for group_name, group_df in groups:
        group_error = np.mean(group_df[true_label_col] != group_df[pred_label_col])
        norm_factor = max(overall_error, 1 - overall_error)
        disparity = abs(group_error - overall_error) / norm_factor
        subgroup_eddi[group_name] = disparity
    return subgroup_eddi

def compute_aggregated_eddi(df, sensitive_attr, true_label_col, pred_label_col):
    subgroup_eddi = compute_subgroup_eddi(df, sensitive_attr, true_label_col, pred_label_col)
    disparities = np.array(list(subgroup_eddi.values()))
    if len(disparities) == 0:
        return 0.0
    aggregated_eddi = np.sqrt(np.sum(disparities ** 2)) / len(disparities)
    return aggregated_eddi

# Main Training Pipeline
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load unstructured dataset containing patient notes and mechanical ventilation flag.
    df = pd.read_csv("final_unstructured_with_mechanical_ventilation.csv", low_memory=False)
    print("Unstructured data shape:", df.shape)
    
    # Identify note columns (those starting with "note_")
    note_columns = [col for col in df.columns if col.startswith("note_")]
    print("Note columns found:", note_columns)
    
    # Filter rows that have at least one valid note.
    def has_valid_note(row):
        for col in note_columns:
            if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].strip():
                return True
        return False
    df_filtered = df[df.apply(has_valid_note, axis=1)].copy()
    print("After filtering, number of rows:", len(df_filtered))
    
    # Prepare tokenizer and BioClinicalBERT model.
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_base = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bioclinical_bert_ft = BioClinicalBERT_FT(bioclinical_bert_base, bioclinical_bert_base.config, device).to(device)
    
    # Compute aggregated text embeddings for each patient.
    print("Computing aggregated text embeddings for each patient...")
    aggregated_embeddings_np, patient_ids = apply_bioclinicalbert_on_patient_notes(
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean", max_length=128
    )
    print("Aggregated text embeddings shape:", aggregated_embeddings_np.shape)
    
    # Use unique patients for labels.
    df_unique = df_filtered.drop_duplicates(subset="subject_id")
    vent_labels = df_unique["mechanical_ventilation"].values

    # Build dataset and dataloader.
    dataset = UnstructuredDataset(aggregated_embeddings_np, vent_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate the unstructured classifier.
    classifier = UnstructuredClassifier(input_size=768, hidden_size=256).to(device)
    
    # Compute class weights for mechanical ventilation.
    class_weights = compute_class_weights(df_unique, "mechanical_ventilation")
    pos_weight = torch.tensor(class_weights.iloc[1], dtype=torch.float, device=device)
    
    criterion = FocalLoss(gamma=2, pos_weight=pos_weight, reduction='mean')
    optimizer = AdamW(classifier.parameters(), lr=2e-5)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(classifier, dataloader, optimizer, device, criterion)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}")
        for thresh in [0.4, 0.5]:
            metrics = evaluate_model(classifier, dataloader, device, threshold=thresh)
            print(f"[Epoch {epoch+1}/{num_epochs}] Threshold: {thresh} Metrics: {metrics}")
    print("Training complete.")
    
    # Inference and Fairness Evaluation.
    probs = get_patient_probabilities(classifier, dataloader, device)
    
    # Load structured demographics dataset.
    structured_demo = pd.read_csv("final_structured_with_mechanical_ventilation.csv", low_memory=False)
    demo_cols = ['subject_id', 'age', 'ETHNICITY', 'INSURANCE']
    df_demo = structured_demo[demo_cols].copy()
    
    # Build predictions dataframe.
    df_probs = pd.DataFrame({
        'subject_id': patient_ids,
        'ventilation_prob': probs[:, 0]
    })
    
    # Merge predictions with demographics.
    df_results = pd.merge(df_demo, df_probs, on='subject_id', how='inner')
    threshold = 0.5
    df_results['ventilation_pred'] = (df_results['ventilation_prob'] >= threshold).astype(int)
    
    # Merge the true mechanical ventilation labels.
    df_results = pd.merge(df_results,
                          df_unique[['subject_id', 'mechanical_ventilation']],
                          on='subject_id', how='inner')
    df_results.rename(columns={'mechanical_ventilation': 'ventilation_true'}, inplace=True)
    
    # Create age buckets.
    df_results['age_bucket'] = df_results['age'].apply(get_age_bucket)
    
    # Map ethnicity and insurance values.
    df_results['ETHNICITY'] = df_results['ETHNICITY'].apply(map_ethnicity)
    df_results['INSURANCE'] = df_results['INSURANCE'].apply(map_insurance)
    
    # Evaluate fairness using sensitive attributes.
    for sensitive_attr in ['age_bucket', 'ETHNICITY', 'INSURANCE']:
        subgroup_eddi = compute_subgroup_eddi(df_results, sensitive_attr, 'ventilation_true', 'ventilation_pred')
        agg_eddi = compute_aggregated_eddi(df_results, sensitive_attr, 'ventilation_true', 'ventilation_pred')
        print(f"\nSubgroup EDDI for {sensitive_attr}:")
        for group, disparity in subgroup_eddi.items():
            print(f"  {group}: {disparity:.4f}")
        print(f"Aggregated EDDI for {sensitive_attr}: {agg_eddi:.4f}")
    
    # Overall attribute-level fairness EDDI (average of the three).
    eddi_age = compute_aggregated_eddi(df_results, 'age_bucket', 'ventilation_true', 'ventilation_pred')
    eddi_ethnicity = compute_aggregated_eddi(df_results, 'ETHNICITY', 'ventilation_true', 'ventilation_pred')
    eddi_insurance = compute_aggregated_eddi(df_results, 'INSURANCE', 'ventilation_true', 'ventilation_pred')
    overall_eddi = np.sqrt(eddi_age**2 + eddi_ethnicity**2 + eddi_insurance**2) / 3
    print("\nOverall Attribute-Level EDDI for Mechanical Ventilation:", overall_eddi)
    
    # Detailed subgroup EDDI for fixed lists.
    fixed_age = ["15-29", "30-49", "50-69", "70-89", "other"]
    fixed_ethnicity = ["white", "black", "asian", "hispanic", "others"]
    fixed_insurance = ["government", "medicare", "medicaid", "private", "self pay", "others"]

    print("\n--- Detailed Age EDDI ---")
    age_eddi_dict = compute_subgroup_eddi(df_results, "age_bucket", "ventilation_true", "ventilation_pred")
    for subgroup in fixed_age:
        eddi = age_eddi_dict.get(subgroup, 0.0)
        print(f"    {subgroup}: {eddi:.4f}")
    
    print("\n--- Detailed Ethnicity EDDI ---")
    eth_eddi_dict = compute_subgroup_eddi(df_results, "ETHNICITY", "ventilation_true", "ventilation_pred")
    for subgroup in fixed_ethnicity:
        eddi = eth_eddi_dict.get(subgroup, 0.0)
        print(f"    {subgroup}: {eddi:.4f}")
    
    print("\n--- Detailed Insurance EDDI ---")
    ins_eddi_dict = compute_subgroup_eddi(df_results, "INSURANCE", "ventilation_true", "ventilation_pred")
    for subgroup in fixed_insurance:
        eddi = ins_eddi_dict.get(subgroup, 0.0)
        print(f"    {subgroup}: {eddi:.4f}")

if __name__ == "__main__":
    train_pipeline()
