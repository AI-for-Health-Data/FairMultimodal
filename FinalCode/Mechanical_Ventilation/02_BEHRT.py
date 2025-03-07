import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertModel, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import chi2_contingency, ttest_ind

# Define helper functions.
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

# Function to compute class weights (for imbalanced classes)
def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

# Read the dataset.
df = pd.read_csv('final_structured_with_mechanical_ventilation.csv')
print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())

# Map disease IDs.
unique_diseases = df['hadm_id'].unique()
disease_mapping = {d: i for i, d in enumerate(unique_diseases)}
df['mapped_disease_id'] = df['hadm_id'].map(disease_mapping)

# Map age to codes.
unique_ages = sorted(df['age'].unique())
age_mapping = {age: i for i, age in enumerate(unique_ages)}
df['age_code'] = df['age'].map(age_mapping)

# Process ethnicity.
if 'categorized_ethnicity' not in df.columns:
    if 'ETHNICITY' in df.columns:
        df['categorized_ethnicity'] = df['ETHNICITY'].fillna('Other').str.upper().str.strip()
    elif 'ethnicity' in df.columns:
        df['categorized_ethnicity'] = df['ethnicity'].fillna('Other').str.upper().str.strip()
    else:
        df['categorized_ethnicity'] = 'OTHER'
if 'categorized_ethnicity_code' not in df.columns:
    df['categorized_ethnicity_code'] = df['categorized_ethnicity'].astype('category').cat.codes

# Create a new column for ethnicity group using our desired mapping.
df['ethnicity_group'] = df['categorized_ethnicity_code'].apply(map_ethnicity)

# Encode gender.
if 'GENDER' in df.columns:
    df['GENDER'] = df['GENDER'].astype('category').cat.codes
elif 'gender' in df.columns:
    df['gender'] = df['gender'].astype('category').cat.codes

# Encode insurance.
if 'INSURANCE' in df.columns:
    df['INSURANCE'] = df['INSURANCE'].astype('category').cat.codes
elif 'insurance' in df.columns:
    df['INSURANCE'] = df['insurance'].astype('category').cat.codes

# Create a new column for insurance group using our desired mapping.
df['insurance_group'] = df['INSURANCE'].apply(map_insurance)

# Map ward IDs.
if 'FIRST_WARDID' in df.columns:
    unique_first_wards = df['FIRST_WARDID'].unique()
    first_ward_mapping = {ward: i for i, ward in enumerate(unique_first_wards)}
    df['first_ward_code'] = df['FIRST_WARDID'].map(first_ward_mapping)
else:
    df['first_ward_code'] = 0

if 'LAST_WARDID' in df.columns:
    unique_last_wards = df['LAST_WARDID'].unique()
    last_ward_mapping = {ward: i for i, ward in enumerate(unique_last_wards)}
    df['last_ward_code'] = df['LAST_WARDID'].map(last_ward_mapping)
else:
    df['last_ward_code'] = 0

# (Optional) Compute class weights.
class_weights = compute_class_weights(df, 'mechanical_ventilation')
print("Class Weights (mechanical_ventilation):")
print(class_weights)

# 2. Prepare Sequences for Model Input.
def prepare_sequences(df):
    patients = df['subject_id'].unique()
    sequences = []
    labels = []
    patient_ids = []
    
    for patient in patients:
        patient_data = df[df['subject_id'] == patient].sort_values(by='ADMITTIME')
        
        # Use the mapped age code rather than raw age.
        age_sequence = patient_data['age_code'].tolist()
        disease_sequence = patient_data['mapped_disease_id'].tolist()
        # Use mapped ward codes.
        admission_loc_sequence = patient_data['first_ward_code'].tolist()
        discharge_loc_sequence = patient_data['last_ward_code'].tolist()
        segment_sequence = [0 if i % 2 == 0 else 1 for i in range(len(age_sequence))]
        
        # Mechanical ventilation label (binary) is the max flag.
        mech_vent_label = patient_data['mechanical_ventilation'].max()
        
        sequences.append({
            'age': age_sequence,
            'diseases': disease_sequence,
            'admission_loc': admission_loc_sequence,
            'discharge_loc': discharge_loc_sequence,
            'segment': segment_sequence,
            'gender': patient_data['GENDER'].tolist(),
            'ethnicity': patient_data['categorized_ethnicity_code'].tolist(),
            'insurance': patient_data['INSURANCE'].tolist()
        })
        labels.append(mech_vent_label)
        patient_ids.append(patient)
    
    return sequences, labels, patient_ids

sequences, labels, patient_ids = prepare_sequences(df)

# Tokenize sequences and pad them.
input_ids = [seq['diseases'] for seq in sequences]
age_ids = [seq['age'] for seq in sequences]
segment_ids = [seq['segment'] for seq in sequences]
admission_loc_ids = [seq['admission_loc'] for seq in sequences]
discharge_loc_ids = [seq['discharge_loc'] for seq in sequences]
gender_ids = [seq['gender'] for seq in sequences]
ethnicity_ids = [seq['ethnicity'] for seq in sequences]
insurance_ids = [seq['insurance'] for seq in sequences]

max_len = max(len(seq) for seq in input_ids)

def pad_sequences(seqs, max_len):
    return [seq + [0]*(max_len - len(seq)) for seq in seqs]

input_ids_padded = pad_sequences(input_ids, max_len)
age_ids_padded = pad_sequences(age_ids, max_len)
segment_ids_padded = pad_sequences(segment_ids, max_len)
admission_loc_ids_padded = pad_sequences(admission_loc_ids, max_len)
discharge_loc_ids_padded = pad_sequences(discharge_loc_ids, max_len)
gender_ids_padded = pad_sequences(gender_ids, max_len)
ethnicity_ids_padded = pad_sequences(ethnicity_ids, max_len)
insurance_ids_padded = pad_sequences(insurance_ids, max_len)

# Convert lists to tensors.
input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long)
age_ids_tensor = torch.tensor(age_ids_padded, dtype=torch.long)
segment_ids_tensor = torch.tensor(segment_ids_padded, dtype=torch.long)
admission_loc_ids_tensor = torch.tensor(admission_loc_ids_padded, dtype=torch.long)
discharge_loc_ids_tensor = torch.tensor(discharge_loc_ids_padded, dtype=torch.long)
gender_ids_tensor = torch.tensor(gender_ids_padded, dtype=torch.long)
ethnicity_ids_tensor = torch.tensor(ethnicity_ids_padded, dtype=torch.long)
insurance_ids_tensor = torch.tensor(insurance_ids_padded, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.float)

# Create dataset and dataloader.
dataset = TensorDataset(
    input_ids_tensor, age_ids_tensor, segment_ids_tensor,
    admission_loc_ids_tensor, discharge_loc_ids_tensor,
    gender_ids_tensor, ethnicity_ids_tensor, insurance_ids_tensor,
    labels_tensor
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Define the BEHRT Model for Mechanical Ventilation Prediction.
class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs, 
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()
        config = BertConfig(
            vocab_size=num_diseases,
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=num_segments,
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
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, 
                gender_ids, ethnicity_ids, insurance_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        age_embeds = self.age_embedding(age_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        admission_loc_embeds = self.admission_loc_embedding(admission_loc_ids)
        discharge_loc_embeds = self.discharge_loc_embedding(discharge_loc_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        ethnicity_embeds = self.ethnicity_embedding(ethnicity_ids)
        insurance_embeds = self.insurance_embedding(insurance_ids)

        combined_output = (sequence_output + age_embeds + segment_embeds +
                           admission_loc_embeds + discharge_loc_embeds +
                           gender_embeds + ethnicity_embeds + insurance_embeds)
        cls_token = combined_output[:, 0, :]
        logits = self.classifier(cls_token).squeeze(1)
        return logits

num_diseases = len(disease_mapping)
num_ages = df['age_code'].max() + 1
num_segments = 2
num_admission_locs = df['first_ward_code'].max() + 1 if 'first_ward_code' in df.columns else 14
num_discharge_locs = df['last_ward_code'].max() + 1 if 'last_ward_code' in df.columns else 14
num_genders = df['GENDER'].max() + 1 if 'GENDER' in df.columns else 2
num_ethnicities = df['categorized_ethnicity_code'].max() + 1
num_insurances = df['INSURANCE'].max() + 1 if 'INSURANCE' in df.columns else 5

model = BEHRTModel(
    num_diseases=num_diseases,
    num_ages=num_ages,
    num_segments=num_segments,
    num_admission_locs=num_admission_locs,
    num_discharge_locs=num_discharge_locs,
    num_genders=num_genders,
    num_ethnicities=num_ethnicities,
    num_insurances=num_insurances,
    hidden_size=768
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)
# 4. Define Loss, Optimizer, and Scheduler.
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
# 5. Training Loop.
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, \
        gender_ids, ethnicity_ids, insurance_ids, labels = batch

        input_ids = input_ids.to(device)
        age_ids = age_ids.to(device)
        segment_ids = segment_ids.to(device)
        admission_loc_ids = admission_loc_ids.to(device)
        discharge_loc_ids = discharge_loc_ids.to(device)
        gender_ids = gender_ids.to(device)
        ethnicity_ids = ethnicity_ids.to(device)
        insurance_ids = insurance_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        attention_mask = (input_ids != 0).long().to(device)
        logits = model(
            input_ids=input_ids,
            age_ids=age_ids,
            segment_ids=segment_ids,
            admission_loc_ids=admission_loc_ids,
            discharge_loc_ids=discharge_loc_ids,
            gender_ids=gender_ids,
            ethnicity_ids=ethnicity_ids,
            insurance_ids=insurance_ids,
            attention_mask=attention_mask
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    avg_loss = total_loss / len(dataset)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}")
    
# 6. Evaluation Functions.
def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, \
            gender_ids, ethnicity_ids, insurance_ids, labels = batch
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            segment_ids = segment_ids.to(device)
            admission_loc_ids = admission_loc_ids.to(device)
            discharge_loc_ids = discharge_loc_ids.to(device)
            gender_ids = gender_ids.to(device)
            ethnicity_ids = ethnicity_ids.to(device)
            insurance_ids = insurance_ids.to(device)
            labels = labels.to(device)
            
            attention_mask = (input_ids != 0).long().to(device)
            logits = model(
                input_ids=input_ids,
                age_ids=age_ids,
                segment_ids=segment_ids,
                admission_loc_ids=admission_loc_ids,
                discharge_loc_ids=discharge_loc_ids,
                gender_ids=gender_ids,
                ethnicity_ids=ethnicity_ids,
                insurance_ids=insurance_ids,
                attention_mask=attention_mask
            )
            all_logits.extend(logits.cpu().numpy())
            preds = torch.sigmoid(logits)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_logits = np.array(all_logits)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    try:
        auroc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auroc = float('nan')
    auprc = average_precision_score(all_labels, all_logits)
    precision = precision_score(all_labels, (all_predictions >= 0.5).astype(int), zero_division=0)
    recall = recall_score(all_labels, (all_predictions >= 0.5).astype(int), zero_division=0)
    f1 = f1_score(all_labels, (all_predictions >= 0.5).astype(int), zero_division=0)
    
    return {
        'logits': all_logits,
        'predictions': all_predictions,
        'labels': all_labels,
        'auroc': auroc,
        'auprc': auprc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

evaluation_results = evaluate_model(model, dataloader, device)
print("Evaluation Results for Mechanical Ventilation Prediction:")
print("AUROC:", evaluation_results['auroc'])
print("AUPRC:", evaluation_results['auprc'])
print("Precision:", evaluation_results['precision'])
print("Recall:", evaluation_results['recall'])
print("F1 Score:", evaluation_results['f1'])

def get_model_predictions(model, dataloader, device):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, \
            gender_ids, ethnicity_ids, insurance_ids, labels = batch
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            segment_ids = segment_ids.to(device)
            admission_loc_ids = admission_loc_ids.to(device)
            discharge_loc_ids = discharge_loc_ids.to(device)
            gender_ids = gender_ids.to(device)
            ethnicity_ids = ethnicity_ids.to(device)
            insurance_ids = insurance_ids.to(device)
            labels = labels.to(device)
            logits = model(
                input_ids=input_ids,
                age_ids=age_ids,
                segment_ids=segment_ids,
                admission_loc_ids=admission_loc_ids,
                discharge_loc_ids=discharge_loc_ids,
                gender_ids=gender_ids,
                ethnicity_ids=ethnicity_ids,
                insurance_ids=insurance_ids,
                attention_mask=(input_ids != 0).long().to(device)
            )
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            all_predictions.extend(preds)
            all_labels.append(labels.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    return np.array(all_predictions), all_labels

preds, true_labels = get_model_predictions(model, dataloader, device)

# Create a filtered DataFrame for evaluation.
df_filtered = df.copy()

# Create age bins using pd.cut.
age_bins = [15, 30, 50, 70, 90]
age_labels_bins = ['15-29', '30-49', '50-69', '70-89']
df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=age_bins, labels=age_labels_bins, right=False)

# Get sensitive attribute arrays.
age_groups = df_filtered['age_group'].values
# Use the new ethnicity_group column for desired ethnicity labels.
ethnicity_groups = df_filtered['ethnicity_group'].values
# Use the new insurance_group column for desired insurance labels.
insurance_groups = df_filtered['insurance_group'].values

# Functions for computing EDDI.
def compute_eddi(sensitive_attr, true_labels, pred_labels):
    total_instances = len(true_labels)
    overall_errors = np.sum(true_labels != pred_labels)
    OER = overall_errors / total_instances if total_instances > 0 else 0

    subgroup_disparities = {}
    unique_groups = np.unique(sensitive_attr)
    for group in unique_groups:
        group_idx = (sensitive_attr == group)
        group_true = true_labels[group_idx]
        group_pred = pred_labels[group_idx]
        if len(group_true) == 0:
            continue
        group_errors = np.sum(group_true != group_pred)
        ER_s = group_errors / len(group_true)
        disparity = (ER_s - OER) / max(OER, 1 - OER)
        subgroup_disparities[group] = disparity
    return subgroup_disparities, OER

def compute_attribute_eddi(subgroup_disparities):
    disparities = np.array(list(subgroup_disparities.values()))
    if disparities.size == 0:
        return 0
    return np.sqrt(np.sum(disparities ** 2)) / len(disparities)

def print_subgroup_eddi(true, pred, sensitive_name, sensitive_values):
    subgroup_disparities, OER = compute_eddi(sensitive_values, true, pred)
    print(f"\nSensitive Attribute: {sensitive_name}")
    print(f"Overall Error Rate (OER): {OER:.4f}")
    for group, disparity in subgroup_disparities.items():
        print(f"  Subgroup {group}: d(s) = {disparity:.4f}")
    attribute_eddi = compute_attribute_eddi(subgroup_disparities)
    print(f"Attribute-level EDDI for {sensitive_name}: {attribute_eddi:.4f}\n")
    return subgroup_disparities, attribute_eddi

# Print subgroup disparities for each sensitive attribute.
print("=== EDDI for Age Groups ===")
age_subgroups, age_eddi = print_subgroup_eddi(true_labels, preds, "Age Groups", age_groups)

print("=== EDDI for Ethnicity Groups ===")
ethnicity_subgroups, ethnicity_eddi = print_subgroup_eddi(true_labels, preds, "Ethnicity Groups", ethnicity_groups)

print("=== EDDI for Insurance Groups ===")
insurance_subgroups, insurance_eddi = print_subgroup_eddi(true_labels, preds, "Insurance Groups", insurance_groups)

print("\nCombined Overall EDDI Calculation:")
overall_eddi = np.sqrt(age_eddi**2 + ethnicity_eddi**2 + insurance_eddi**2) / 3
print(f"Overall EDDI for Mechanical Ventilation: {overall_eddi:.4f}")
