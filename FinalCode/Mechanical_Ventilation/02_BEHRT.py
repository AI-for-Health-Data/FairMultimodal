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
    average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt

# 1. Load and Preprocess the Dataset
df = pd.read_csv('structured_dataset.csv')
print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())

# --- Map hospital admission IDs to disease codes ---
unique_diseases = df['hadm_id'].unique()
disease_mapping = {d: i for i, d in enumerate(unique_diseases)}
df['mapped_disease_id'] = df['hadm_id'].map(disease_mapping)

# --- Map age to an index ---
unique_ages = sorted(df['age'].unique())
age_mapping = {age: i for i, age in enumerate(unique_ages)}
df['age_code'] = df['age'].map(age_mapping)

# --- Create demographic codes for ethnicity ---
if 'categorized_ethnicity' not in df.columns:
    if 'ETHNICITY' in df.columns:
        df['categorized_ethnicity'] = df['ETHNICITY'].fillna('Other').str.upper().str.strip()
    elif 'ethnicity' in df.columns:
        df['categorized_ethnicity'] = df['ethnicity'].fillna('Other').str.upper().str.strip()
    else:
        df['categorized_ethnicity'] = 'OTHER'
        
if 'categorized_ethnicity_code' not in df.columns:
    df['categorized_ethnicity_code'] = df['categorized_ethnicity'].astype('category').cat.codes

# --- Convert other categorical features to codes ---
if 'GENDER' in df.columns:
    df['GENDER'] = df['GENDER'].astype('category').cat.codes
elif 'gender' in df.columns:
    df['gender'] = df['gender'].astype('category').cat.codes

if 'INSURANCE' in df.columns:
    df['INSURANCE'] = df['INSURANCE'].astype('category').cat.codes
elif 'insurance' in df.columns:
    df['insurance'] = df['insurance'].astype('category').cat.codes

# --- Map ward IDs to contiguous codes ---
# For FIRST_WARDID (admission location)
if 'FIRST_WARDID' in df.columns:
    unique_first_wards = df['FIRST_WARDID'].unique()
    first_ward_mapping = {ward: i for i, ward in enumerate(unique_first_wards)}
    df['first_ward_code'] = df['FIRST_WARDID'].map(first_ward_mapping)
else:
    df['first_ward_code'] = 0

# For LAST_WARDID (discharge location)
if 'LAST_WARDID' in df.columns:
    unique_last_wards = df['LAST_WARDID'].unique()
    last_ward_mapping = {ward: i for i, ward in enumerate(unique_last_wards)}
    df['last_ward_code'] = df['LAST_WARDID'].map(last_ward_mapping)
else:
    df['last_ward_code'] = 0
# 2. Prepare Sequences for Model Input
def prepare_sequences(df):
    patients = df['subject_id'].unique()
    sequences = []
    labels = []
    patient_ids = []
    
    for patient in patients:
        patient_data = df[df['subject_id'] == patient]
        # Use the mapped disease IDs as tokens.
        disease_sequence = patient_data['mapped_disease_id'].tolist()
        # Use the age_code as tokens.
        age_sequence = patient_data['age_code'].tolist()
        # Use remapped ward codes for admission and discharge locations.
        if 'first_ward_code' in patient_data.columns:
            admission_loc_sequence = patient_data['first_ward_code'].tolist()
        else:
            admission_loc_sequence = [0] * len(disease_sequence)
        if 'last_ward_code' in patient_data.columns:
            discharge_loc_sequence = patient_data['last_ward_code'].tolist()
        else:
            discharge_loc_sequence = [0] * len(disease_sequence)
        # Create a segment sequence (alternating 0 and 1).
        segment_sequence = [0 if i % 2 == 0 else 1 for i in range(len(disease_sequence))]
        # Get gender, ethnicity, and insurance codes.
        if 'GENDER' in patient_data.columns:
            gender_sequence = patient_data['GENDER'].tolist()
        elif 'gender' in patient_data.columns:
            gender_sequence = patient_data['gender'].tolist()
        else:
            gender_sequence = [0] * len(disease_sequence)
        ethnicity_sequence = patient_data['categorized_ethnicity_code'].tolist()
        if 'INSURANCE' in patient_data.columns:
            insurance_sequence = patient_data['INSURANCE'].tolist()
        elif 'insurance' in patient_data.columns:
            insurance_sequence = patient_data['insurance'].tolist()
        else:
            insurance_sequence = [0] * len(disease_sequence)
            
        sequences.append({
            'diseases': disease_sequence,
            'age': age_sequence,
            'admission_loc': admission_loc_sequence,
            'discharge_loc': discharge_loc_sequence,
            'segment': segment_sequence,
            'gender': gender_sequence,
            'ethnicity': ethnicity_sequence,
            'insurance': insurance_sequence
        })
        # For the target, predict mechanical ventilation (using the maximum value per patient)
        mechanical_ventilation_label = patient_data['mechanical_ventilation'].max()
        labels.append(mechanical_ventilation_label)
        patient_ids.append(patient)
    return sequences, labels, patient_ids

sequences, labels, patient_ids = prepare_sequences(df)

# Pad sequences to a fixed maximum length.
def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) for seq in sequences]

max_len = max(len(seq['diseases']) for seq in sequences)

input_ids = pad_sequences([seq['diseases'] for seq in sequences], max_len)
age_ids = pad_sequences([seq['age'] for seq in sequences], max_len)
segment_ids = pad_sequences([seq['segment'] for seq in sequences], max_len)
admission_loc_ids = pad_sequences([seq['admission_loc'] for seq in sequences], max_len)
discharge_loc_ids = pad_sequences([seq['discharge_loc'] for seq in sequences], max_len)
gender_ids = pad_sequences([seq['gender'] for seq in sequences], max_len)
ethnicity_ids = pad_sequences([seq['ethnicity'] for seq in sequences], max_len)
insurance_ids = pad_sequences([seq['insurance'] for seq in sequences], max_len)

# Convert lists to PyTorch tensors.
input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
age_ids_tensor = torch.tensor(age_ids, dtype=torch.long)
segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)
admission_loc_ids_tensor = torch.tensor(admission_loc_ids, dtype=torch.long)
discharge_loc_ids_tensor = torch.tensor(discharge_loc_ids, dtype=torch.long)
gender_ids_tensor = torch.tensor(gender_ids, dtype=torch.long)
ethnicity_ids_tensor = torch.tensor(ethnicity_ids, dtype=torch.long)
insurance_ids_tensor = torch.tensor(insurance_ids, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.float)

# Create a TensorDataset and DataLoader for training.
dataset = TensorDataset(
    input_ids_tensor, age_ids_tensor, segment_ids_tensor,
    admission_loc_ids_tensor, discharge_loc_ids_tensor,
    gender_ids_tensor, ethnicity_ids_tensor, insurance_ids_tensor,
    labels_tensor
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# 3. Define the BEHRT Model for Mechanical Ventilation Prediction
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
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
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
        
        cls_token = combined_output[:, 0, :]  # Use [CLS] token
        logits = self.classifier(cls_token).squeeze(1)
        return logits

# Determine sizes for embeddings.
num_diseases = len(disease_mapping)
num_ages = len(unique_ages)
num_segments = 2
num_admission_locs = df['first_ward_code'].nunique() if 'first_ward_code' in df.columns else 14
num_discharge_locs = df['last_ward_code'].nunique() if 'last_ward_code' in df.columns else 14
num_genders = df['GENDER'].nunique() if 'GENDER' in df.columns else df['gender'].nunique()
num_ethnicities = df['categorized_ethnicity_code'].nunique()
num_insurances = df['INSURANCE'].nunique() if 'INSURANCE' in df.columns else df['insurance'].nunique()

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
# 4. Define Loss, Optimizer, and Scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
# 5. Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        (input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids,
         gender_ids, ethnicity_ids, insurance_ids, labels) = batch

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
# 6. Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            (input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids,
             gender_ids, ethnicity_ids, insurance_ids, labels) = batch
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
        auroc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        auroc = float('nan')
    auprc = average_precision_score(all_labels, all_predictions)
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
import numpy as np
import pandas as pd

# DataFrame merging and demographic column standardization
results_df = pd.DataFrame({
    'subject_id': patient_ids,
    'label': evaluation_results['labels'],
    'pred_prob': evaluation_results['predictions']
})

demo_columns = []
if 'age' in df.columns:
    demo_columns.append('age')
if 'ethnicity' in df.columns:
    demo_columns.append('ethnicity')
elif 'ETHNICITY' in df.columns:
    demo_columns.append('ETHNICITY')
if 'insurance' in df.columns:
    demo_columns.append('insurance')
elif 'INSURANCE' in df.columns:
    demo_columns.append('INSURANCE')

demo_df = df.drop_duplicates(subset='subject_id')[['subject_id'] + demo_columns]
results_df = results_df.merge(demo_df, on='subject_id', how='left')

# Rename columns to standardize names
if 'ETHNICITY' in results_df.columns:
    results_df.rename(columns={'ETHNICITY': 'ethnicity'}, inplace=True)
if 'INSURANCE' in results_df.columns:
    results_df.rename(columns={'INSURANCE': 'insurance'}, inplace=True)

# Categorization functions 
def categorize_age(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    else:
        return '70-89'

def categorize_ethnicity(ethnicity):
    eth = ethnicity.upper()
    if eth in ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN']:
        return 'White'
    elif eth in ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN', 'BLACK/AFRICAN', 'CARIBBEAN ISLAND']:
        return 'Black'
    elif eth in ['HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - MEXICAN']:
        return 'Hispanic'
    elif eth in ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - INDIAN']:
        return 'Asian'
    else:
        return 'Other'

def categorize_insurance(insurance):
    ins = str(insurance).upper()
    if 'MEDICARE' in ins:
        return 'Medicare'
    elif 'MEDICAID' in ins:
        return 'Medicaid'
    elif 'PRIVATE' in ins:
        return 'Private'
    elif 'GOVERNMENT' in ins:
        return 'Government'
    else:
        return 'Other'

if 'age' in results_df.columns:
    results_df['age_bucket'] = results_df['age'].apply(categorize_age)
if 'ethnicity' in results_df.columns:
    results_df['ethnicity_group'] = results_df['ethnicity'].apply(categorize_ethnicity)
if 'insurance' in results_df.columns:
    results_df['insurance_group'] = results_df['insurance'].apply(categorize_insurance)

# Functions to calculate d(s) and EDDI 
def calculate_d_values(df, sensitive_attr, true_label='label', pred_prob='pred_prob', threshold=0.5):
    """
    Calculates d(s) for each subgroup in the sensitive attribute.
    d(s) = (ER_s - OER) / max(OER, 1 - OER)
    Returns a dictionary mapping subgroup -> d(s) and the overall error rate.
    """
    df_copy = df.copy()
    df_copy['predicted'] = (df_copy[pred_prob] >= threshold).astype(int)
    overall_error = np.mean(df_copy[true_label] != df_copy['predicted'])
    
    groups = df_copy[sensitive_attr].unique()
    d_dict = {}
    for group in groups:
        group_df = df_copy[df_copy[sensitive_attr] == group]
        group_error = np.mean(group_df[true_label] != group_df['predicted'])
        d_s = (group_error - overall_error) / max(overall_error, 1 - overall_error)
        d_dict[group] = d_s
    return d_dict, overall_error

def calculate_eddi_from_d(d_dict):
    """
    Computes EDDI for an attribute from its subgroup d(s) values.
    EDDI = (sqrt(sum_{s in S} (d(s))^2)) / (number of groups)
    """
    num_groups = len(d_dict)
    sum_sq = sum(d**2 for d in d_dict.values())
    eddi_attr = np.sqrt(sum_sq) / num_groups
    return eddi_attr

# Calculate EDDI for each sensitive attribute

# For Age Buckets
d_age, oer_age = calculate_d_values(results_df, sensitive_attr='age_bucket')
eddi_age = calculate_eddi_from_d(d_age)
print("d(s) for Age Buckets:", d_age)
print("EDDI for Age Buckets:", eddi_age)

# For Ethnicity Groups
d_ethnicity, oer_ethnicity = calculate_d_values(results_df, sensitive_attr='ethnicity_group')
eddi_ethnicity = calculate_eddi_from_d(d_ethnicity)
print("d(s) for Ethnicity Groups:", d_ethnicity)
print("EDDI for Ethnicity Groups:", eddi_ethnicity)

# For Insurance Groups
d_insurance, oer_insurance = calculate_d_values(results_df, sensitive_attr='insurance_group')
eddi_insurance = calculate_eddi_from_d(d_insurance)
print("d(s) for Insurance Groups:", d_insurance)
print("EDDI for Insurance Groups:", eddi_insurance)

# Calculate Overall EDDI
overall_eddi = np.sqrt(eddi_age**2 + eddi_ethnicity**2 + eddi_insurance**2) / 3
print("Overall EDDI for mechanical ventilation:", overall_eddi)
