import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertModel, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.calibration import calibration_curve
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, ttest_ind


# Data Preprocessing and Label Creation
# Read the structured common dataset.
df = pd.read_csv('final_structured_common.csv')

# Map hadm_id to a disease ID
unique_diseases = df['hadm_id'].unique()
disease_mapping = {d: i for i, d in enumerate(unique_diseases)}
df['mapped_disease_id'] = df['hadm_id'].map(disease_mapping)

# Function to categorize ethnicity.
def categorize_ethnicity(ethnicity):
    if pd.isna(ethnicity):
        return 'Other'
    ethnicity = ethnicity.upper().strip()
    # White category.
    if ethnicity in ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN']:
        return 'White'
    elif any(keyword in ethnicity for keyword in ['WHITE', 'EUROPEAN', 'RUSSIAN', 'BRAZILIAN', 'PORTUGUESE']):
        return 'White'
    # Black category.
    elif ethnicity in ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN', 'BLACK/AFRICAN', 'CARIBBEAN ISLAND']:
        return 'Black'
    elif any(keyword in ethnicity for keyword in ['BLACK', 'AFRICAN', 'CAPE VERDEAN', 'HAITIAN']):
        return 'Black'
    # Hispanic category.
    elif ethnicity in ['HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - MEXICAN']:
        return 'Hispanic'
    elif any(keyword in ethnicity for keyword in ['HISPANIC', 'LATINO', 'GUATEMALAN', 'PUERTO RICAN', 'DOMINICAN', 'SALVADORAN', 'COLOMBIAN', 'MEXICAN', 'CUBAN', 'HONDURAN']):
        return 'Hispanic'
    # Asian category.
    elif ethnicity in ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - INDIAN']:
        return 'Asian'
    elif any(keyword in ethnicity for keyword in ['ASIAN', 'CHINESE', 'JAPANESE', 'VIETNAMESE', 'FILIPINO', 'THAI', 'KOREAN', 'CAMBODIAN', 'ASIAN INDIAN']):
        return 'Asian'
    else:
        return 'Other'

# Create ethnicity columns if missing.
if 'categorized_ethnicity' not in df.columns:
    df['categorized_ethnicity'] = df['ETHNICITY'].apply(categorize_ethnicity)
if 'categorized_ethnicity_code' not in df.columns:
    df['categorized_ethnicity_code'] = df['categorized_ethnicity'].astype('category').cat.codes

# Convert time columns to datetime.
df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
df['DEATHTIME'] = pd.to_datetime(df['DEATHTIME'], errors='coerce')

# Calculate time differences (in hours).
df['time_to_discharge'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600
df['time_to_death'] = (df['DEATHTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600

# Encode categorical features.
df['GENDER'] = df['GENDER'].astype('category').cat.codes
df['INSURANCE'] = df['INSURANCE'].astype('category').cat.codes

# Fill missing age values.
df['age'] = df['age'].fillna(0)

# For LOS, assume that the column "los_binary" is our target for LOS > 3 days.
# Filter records: include only those where the record is at least 6 hours before discharge or death.
df_filtered = df[
    ((df['time_to_discharge'] > 6) & (df['short_term_mortality'] == 0)) |
    ((df['time_to_death'] > 6) & (df['short_term_mortality'] == 1))
].copy()

# Prepare sequences per patient.
def prepare_sequences(df):
    patients = df['subject_id'].unique()
    sequences = []
    labels = []
    patient_ids = []
    
    for patient in patients:
        patient_data = df[df['subject_id'] == patient].sort_values(by='ADMITTIME')
        
        # Feature sequences.
        age_sequence = patient_data['age'].tolist()
        disease_sequence = patient_data['mapped_disease_id'].tolist()
        segment_sequence = [0 if i % 2 == 0 else 1 for i in range(len(age_sequence))]
        
        # Admission and discharge location sequences (use placeholder 0 if missing).
        if 'FIRST_WARDID' in patient_data.columns:
            admission_loc_sequence = patient_data['FIRST_WARDID'].tolist()
        else:
            admission_loc_sequence = [0] * len(age_sequence)
        if 'LAST_WARDID' in patient_data.columns:
            discharge_loc_sequence = patient_data['LAST_WARDID'].tolist()
        else:
            discharge_loc_sequence = [0] * len(age_sequence)
        
        # Outcome labels: for each patient, take the maximum value over admissions.
        mortality_label = patient_data['short_term_mortality'].max()
        los_label = patient_data['los_binary'].max()
        mech_label = patient_data['mechanical_ventilation'].max()
        
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
        labels.append([mortality_label, los_label, mech_label])
        patient_ids.append(patient)
    return sequences, labels, patient_ids

sequences, labels, patient_ids = prepare_sequences(df_filtered)

# Collect feature sequences.
input_ids = [seq['diseases'] for seq in sequences]
age_ids = [seq['age'] for seq in sequences]
segment_ids = [seq['segment'] for seq in sequences]
admission_loc_ids = [seq['admission_loc'] for seq in sequences]
discharge_loc_ids = [seq['discharge_loc'] for seq in sequences]
gender_ids = [seq['gender'] for seq in sequences]
ethnicity_ids = [seq['ethnicity'] for seq in sequences]
insurance_ids = [seq['insurance'] for seq in sequences]

# Pad sequences to a uniform length.
max_len = max(len(seq) for seq in input_ids)
def pad_sequences(sequences, max_len):
    return [seq + [0]*(max_len - len(seq)) for seq in sequences]

input_ids_padded = pad_sequences(input_ids, max_len)
age_ids_padded = pad_sequences(age_ids, max_len)
segment_ids_padded = pad_sequences(segment_ids, max_len)
admission_loc_ids_padded = pad_sequences(admission_loc_ids, max_len)
discharge_loc_ids_padded = pad_sequences(discharge_loc_ids, max_len)
gender_ids_padded = pad_sequences(gender_ids, max_len)
ethnicity_ids_padded = pad_sequences(ethnicity_ids, max_len)
insurance_ids_padded = pad_sequences(insurance_ids, max_len)

# Convert padded sequences to PyTorch tensors.
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

class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs,
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()
        # Define a custom BERT configuration.
        config = BertConfig(
            vocab_size=num_diseases + num_ages + num_segments + num_admission_locs + num_discharge_locs + 2,
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
        
        # Three classifiers: for mortality, LOS and mechanical ventilation.
        self.classifier_mortality = nn.Linear(hidden_size, 1)
        self.classifier_los = nn.Linear(hidden_size, 1)
        self.classifier_mech = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids,
                gender_ids, ethnicity_ids, insurance_ids, attention_mask=None):
        # Clamp IDs to valid ranges.
        age_ids = torch.clamp(age_ids, min=0, max=self.age_embedding.num_embeddings - 1)
        segment_ids = torch.clamp(segment_ids, min=0, max=self.segment_embedding.num_embeddings - 1)
        admission_loc_ids = torch.clamp(admission_loc_ids, min=0, max=self.admission_loc_embedding.num_embeddings - 1)
        discharge_loc_ids = torch.clamp(discharge_loc_ids, min=0, max=self.discharge_loc_embedding.num_embeddings - 1)
        gender_ids = torch.clamp(gender_ids, min=0, max=self.gender_embedding.num_embeddings - 1)
        ethnicity_ids = torch.clamp(ethnicity_ids, min=0, max=self.ethnicity_embedding.num_embeddings - 1)
        insurance_ids = torch.clamp(insurance_ids, min=0, max=self.insurance_embedding.num_embeddings - 1)
        
        # Forward pass through BERT.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Get additional embeddings.
        age_embeds = self.age_embedding(age_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        admission_loc_embeds = self.admission_loc_embedding(admission_loc_ids)
        discharge_loc_embeds = self.discharge_loc_embedding(discharge_loc_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        ethnicity_embeds = self.ethnicity_embedding(ethnicity_ids)
        insurance_embeds = self.insurance_embedding(insurance_ids)
        
        # Combine embeddings (by summing).
        combined_output = (sequence_output + age_embeds + segment_embeds +
                           admission_loc_embeds + discharge_loc_embeds +
                           gender_embeds + ethnicity_embeds + insurance_embeds)
        
        # Use the [CLS] token (first token) for classification.
        logits_mortality = self.classifier_mortality(combined_output[:, 0, :])
        logits_los = self.classifier_los(combined_output[:, 0, :])
        logits_mech = self.classifier_mech(combined_output[:, 0, :])
        return logits_mortality, logits_los, logits_mech

# Define hyperparameters based on dataset statistics.
num_diseases = len(disease_mapping)
num_ages = int(df_filtered['age'].nunique() + 1)  
num_segments = 2
num_admission_locs = int(df_filtered['FIRST_WARDID'].nunique()) if 'FIRST_WARDID' in df_filtered.columns else 1
num_discharge_locs = int(df_filtered['LAST_WARDID'].nunique()) if 'LAST_WARDID' in df_filtered.columns else 1
num_genders = int(df_filtered['GENDER'].nunique())
num_ethnicities = int(df_filtered['categorized_ethnicity_code'].nunique())
num_insurances = int(df_filtered['INSURANCE'].nunique())

model = BEHRTModel(
    num_diseases=num_diseases,
    num_ages=num_ages,
    num_segments=num_segments,
    num_admission_locs=num_admission_locs,
    num_discharge_locs=num_discharge_locs,
    num_genders=num_genders,
    num_ethnicities=num_ethnicities,
    num_insurances=num_insurances
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

# Compute class weights for mortality, LOS, and mechanical ventilation.
class_weights_mortality = compute_class_weights(df_filtered, 'short_term_mortality')
class_weights_los = compute_class_weights(df_filtered, 'los_binary')
class_weights_mech = compute_class_weights(df_filtered, 'mechanical_ventilation')

# Convert class weights to tensors.
class_weights_tensor_mortality = torch.tensor(class_weights_mortality.values, dtype=torch.float, device=device)
class_weights_tensor_los = torch.tensor(class_weights_los.values, dtype=torch.float, device=device)
class_weights_tensor_mech = torch.tensor(class_weights_mech.values, dtype=torch.float, device=device)

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
bce_loss_mortality = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_mortality[1])
bce_loss_los = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_los[1])
bce_loss_mech = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_mech[1])

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids_b, age_ids_b, segment_ids_b, admission_loc_ids_b, discharge_loc_ids_b, \
        gender_ids_b, ethnicity_ids_b, insurance_ids_b, labels_b = batch
        
        input_ids_b = input_ids_b.to(device)
        age_ids_b = age_ids_b.to(device)
        segment_ids_b = segment_ids_b.to(device)
        admission_loc_ids_b = admission_loc_ids_b.to(device)
        discharge_loc_ids_b = discharge_loc_ids_b.to(device)
        gender_ids_b = gender_ids_b.to(device)
        ethnicity_ids_b = ethnicity_ids_b.to(device)
        insurance_ids_b = insurance_ids_b.to(device)
        labels_b = labels_b.to(device)
        
        optimizer.zero_grad()
        logits_mortality, logits_los, logits_mech = model(
            input_ids=input_ids_b,
            age_ids=age_ids_b,
            segment_ids=segment_ids_b,
            admission_loc_ids=admission_loc_ids_b,
            discharge_loc_ids=discharge_loc_ids_b,
            gender_ids=gender_ids_b,
            ethnicity_ids=ethnicity_ids_b,
            insurance_ids=insurance_ids_b
        )
        loss_mortality = bce_loss_mortality(logits_mortality, labels_b[:, 0].unsqueeze(1))
        loss_los = bce_loss_los(logits_los, labels_b[:, 1].unsqueeze(1))
        loss_mech = bce_loss_mech(logits_mech, labels_b[:, 2].unsqueeze(1))
        loss = loss_mortality + loss_los + loss_mech
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(total_loss)
    print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_logits = {'mortality': [], 'los': [], 'mech': []}
    all_predictions = {'mortality': [], 'los': [], 'mech': []}
    with torch.no_grad():
        for batch in dataloader:
            input_ids_b, age_ids_b, segment_ids_b, admission_loc_ids_b, discharge_loc_ids_b, \
            gender_ids_b, ethnicity_ids_b, insurance_ids_b, labels_b = batch
            
            input_ids_b = input_ids_b.to(device)
            age_ids_b = age_ids_b.to(device)
            segment_ids_b = segment_ids_b.to(device)
            admission_loc_ids_b = admission_loc_ids_b.to(device)
            discharge_loc_ids_b = discharge_loc_ids_b.to(device)
            gender_ids_b = gender_ids_b.to(device)
            ethnicity_ids_b = ethnicity_ids_b.to(device)
            insurance_ids_b = insurance_ids_b.to(device)
            labels_b = labels_b.to(device)
            
            logits_mortality, logits_los, logits_mech = model(
                input_ids=input_ids_b,
                age_ids=age_ids_b,
                segment_ids=segment_ids_b,
                admission_loc_ids=admission_loc_ids_b,
                discharge_loc_ids=discharge_loc_ids_b,
                gender_ids=gender_ids_b,
                ethnicity_ids=ethnicity_ids_b,
                insurance_ids=insurance_ids_b
            )
            
            all_logits['mortality'].append(logits_mortality.cpu().numpy())
            all_logits['los'].append(logits_los.cpu().numpy())
            all_logits['mech'].append(logits_mech.cpu().numpy())
            all_labels.append(labels_b.cpu().numpy())
            
            pred_mortality = (torch.sigmoid(logits_mortality) > 0.5).cpu().numpy().astype(int)
            pred_los = (torch.sigmoid(logits_los) > 0.5).cpu().numpy().astype(int)
            pred_mech = (torch.sigmoid(logits_mech) > 0.5).cpu().numpy().astype(int)
            all_predictions['mortality'].append(pred_mortality)
            all_predictions['los'].append(pred_los)
            all_predictions['mech'].append(pred_mech)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_logits['mortality'] = np.concatenate(all_logits['mortality'], axis=0)
    all_logits['los'] = np.concatenate(all_logits['los'], axis=0)
    all_logits['mech'] = np.concatenate(all_logits['mech'], axis=0)
    all_predictions['mortality'] = np.concatenate(all_predictions['mortality'], axis=0)
    all_predictions['los'] = np.concatenate(all_predictions['los'], axis=0)
    all_predictions['mech'] = np.concatenate(all_predictions['mech'], axis=0)
    
    auroc_mortality = roc_auc_score(all_labels[:, 0], all_logits['mortality'])
    auroc_los = roc_auc_score(all_labels[:, 1], all_logits['los'])
    auroc_mech = roc_auc_score(all_labels[:, 2], all_logits['mech'])
    
    precision_mortality, recall_mortality, _ = precision_recall_curve(all_labels[:, 0], all_logits['mortality'])
    auprc_mortality = auc(recall_mortality, precision_mortality)
    precision_los, recall_los, _ = precision_recall_curve(all_labels[:, 1], all_logits['los'])
    auprc_los = auc(recall_los, precision_los)
    precision_mech, recall_mech, _ = precision_recall_curve(all_labels[:, 2], all_logits['mech'])
    auprc_mech = auc(recall_mech, precision_mech)
    
    precision_mortality_score = precision_score(all_labels[:, 0], all_predictions['mortality'])
    recall_mortality_score = recall_score(all_labels[:, 0], all_predictions['mortality'])
    f1_mortality = f1_score(all_labels[:, 0], all_predictions['mortality'])
    
    precision_los_score = precision_score(all_labels[:, 1], all_predictions['los'])
    recall_los_score = recall_score(all_labels[:, 1], all_predictions['los'])
    f1_los = f1_score(all_labels[:, 1], all_predictions['los'])
    
    precision_mech_score = precision_score(all_labels[:, 2], all_predictions['mech'])
    recall_mech_score = recall_score(all_labels[:, 2], all_predictions['mech'])
    f1_mech = f1_score(all_labels[:, 2], all_predictions['mech'])
    
    return {
        'logits': all_logits,
        'predictions': all_predictions,
        'labels': all_labels,
        'auroc': {'mortality': auroc_mortality, 'los': auroc_los, 'mech': auroc_mech},
        'auprc': {'mortality': auprc_mortality, 'los': auprc_los, 'mech': auprc_mech},
        'precision': {'mortality': precision_mortality_score, 'los': precision_los_score, 'mech': precision_mech_score},
        'recall': {'mortality': recall_mortality_score, 'los': recall_los_score, 'mech': recall_mech_score},
        'f1': {'mortality': f1_mortality, 'los': f1_los, 'mech': f1_mech}
    }

evaluation_results = evaluate_model(model, dataloader, device)
print("Evaluation Results:")
print(evaluation_results)

def get_model_predictions(model, dataloader, device):
    all_predictions = {'mortality': [], 'los': [], 'mech': []}
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids_b, age_ids_b, segment_ids_b, admission_loc_ids_b, discharge_loc_ids_b, \
            gender_ids_b, ethnicity_ids_b, insurance_ids_b, labels_b = batch
            
            input_ids_b = input_ids_b.to(device)
            age_ids_b = age_ids_b.to(device)
            segment_ids_b = segment_ids_b.to(device)
            admission_loc_ids_b = admission_loc_ids_b.to(device)
            discharge_loc_ids_b = discharge_loc_ids_b.to(device)
            gender_ids_b = gender_ids_b.to(device)
            ethnicity_ids_b = ethnicity_ids_b.to(device)
            insurance_ids_b = insurance_ids_b.to(device)
            labels_b = labels_b.to(device)
            
            logits_mortality, logits_los, logits_mech = model(
                input_ids=input_ids_b,
                age_ids=age_ids_b,
                segment_ids=segment_ids_b,
                admission_loc_ids=admission_loc_ids_b,
                discharge_loc_ids=discharge_loc_ids_b,
                gender_ids=gender_ids_b,
                ethnicity_ids=ethnicity_ids_b,
                insurance_ids=insurance_ids_b,
                attention_mask=(input_ids_b != 0).long().to(device)
            )
            preds = (torch.sigmoid(logits_mortality) > 0.5).cpu().numpy().astype(int)
            preds_los = (torch.sigmoid(logits_los) > 0.5).cpu().numpy().astype(int)
            preds_mech = (torch.sigmoid(logits_mech) > 0.5).cpu().numpy().astype(int)
            all_predictions['mortality'].extend(preds)
            all_predictions['los'].extend(preds_los)
            all_predictions['mech'].extend(preds_mech)
            all_labels.append(labels_b.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    return np.array(all_predictions['mortality']), np.array(all_predictions['los']), np.array(all_predictions['mech']), all_labels

preds_mort, preds_los, preds_mech, labels_arr = get_model_predictions(model, dataloader, device)


# Fairness Evaluation Functions
def calculate_equalized_odds(labels, predictions, sensitive_attribute):
    unique_groups = np.unique(sensitive_attribute)
    equalized_odds = {}
    for group in unique_groups:
        group_idx = (sensitive_attribute == group)
        group_labels = labels[group_idx]
        group_predictions = predictions[group_idx]
        cm = confusion_matrix(group_labels, group_predictions)
        tn, fp, fn, tp = (0, 0, 0, 0)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        equalized_odds[group] = {'TPR': tpr, 'FPR': fpr}
    return equalized_odds

def calculate_equal_opportunity(labels, predictions, sensitive_attribute):
    unique_groups = np.unique(sensitive_attribute)
    equal_opportunity = {}
    for group in unique_groups:
        group_idx = (sensitive_attribute == group)
        group_labels = labels[group_idx]
        group_predictions = predictions[group_idx]
        cm = confusion_matrix(group_labels, group_predictions)
        tn, fp, fn, tp = (0, 0, 0, 0)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        equal_opportunity[group] = {'TPR': tpr}
    return equal_opportunity

def calculate_disparity(metric_values):
    return max(metric_values) - min(metric_values)

def evaluate_fairness(labels, predictions, sensitive_attribute):
    predictions = np.array(predictions)
    labels = np.array(labels)
    equalized_odds = calculate_equalized_odds(labels, predictions, sensitive_attribute)
    equal_opportunity = calculate_equal_opportunity(labels, predictions, sensitive_attribute)
    tpr_values = [v['TPR'] for v in equalized_odds.values()]
    fpr_values = [v['FPR'] for v in equalized_odds.values()]
    disparity_tpr = calculate_disparity(tpr_values)
    disparity_fpr = calculate_disparity(fpr_values)
    return {
        'equalized_odds': equalized_odds,
        'equal_opportunity': equal_opportunity,
        'disparity': {'TPR': disparity_tpr, 'FPR': disparity_fpr}
    }

# Define demographic mappings.
gender_mapping = {0: 'Female', 1: 'Male'}
ethnicity_mapping = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Hispanic/Latino', 4: 'Other'}
insurance_mapping = {0: 'Government', 1: 'Medicaid', 2: 'Medicare', 3: 'Private', 4: 'Self Pay'}

# Evaluate fairness for gender on mortality predictions.
sensitive_attribute_gender = df_filtered['GENDER'].values
fairness_results_gender = evaluate_fairness(labels_arr[:, 0], preds_mort, sensitive_attribute_gender)
print("Fairness Evaluation - Gender (Mortality):")
print(fairness_results_gender)

# Update age bins to cover all ages (using provided bins).
age_bins = [15, 30, 50, 70, 90]
age_labels_bins = ['15-29', '30-49', '50-69', '70-89']
df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=age_bins, labels=age_labels_bins, right=False)
df_filtered['age_group'] = df_filtered['age_group'].astype(str)

# Use existing categorized ethnicity.
df_filtered['ethnicity_group'] = df_filtered['categorized_ethnicity']

# For insurance, map the code to groups.
df_filtered['insurance_group'] = df_filtered['INSURANCE'].replace({
    0: 'Government', 1: 'Medicaid', 2: 'Medicare', 3: 'Private', 4: 'Self Pay'
})

age_groups = df_filtered['age_group'].values
ethnicity_groups = df_filtered['ethnicity_group'].values
insurance_groups = df_filtered['insurance_group'].values

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

# For Mortality (already computed above), we have:
print("=== EDDI for Age Groups (Mortality) ===")
age_subgroups_mort, age_eddi_mort = print_subgroup_eddi(labels_arr[:, 0], preds_mort, "Age Groups", age_groups)
print("=== EDDI for Ethnicity Groups (Mortality) ===")
ethnicity_subgroups_mort, ethnicity_eddi_mort = print_subgroup_eddi(labels_arr[:, 0], preds_mort, "Ethnicity Groups", ethnicity_groups)
print("=== EDDI for Insurance Groups (Mortality) ===")
insurance_subgroups_mort, insurance_eddi_mort = print_subgroup_eddi(labels_arr[:, 0], preds_mort, "Insurance Groups", insurance_groups)
overall_eddi_mort = np.sqrt(age_eddi_mort**2 + ethnicity_eddi_mort**2 + insurance_eddi_mort**2) / 3
print(f"\nOverall EDDI for Mortality Predictions: {overall_eddi_mort:.4f}\n")

# For LOS outcome:
print("=== EDDI for Age Groups (LOS) ===")
age_subgroups_los, age_eddi_los = print_subgroup_eddi(labels_arr[:, 1], preds_los, "Age Groups", age_groups)
print("=== EDDI for Ethnicity Groups (LOS) ===")
ethnicity_subgroups_los, ethnicity_eddi_los = print_subgroup_eddi(labels_arr[:, 1], preds_los, "Ethnicity Groups", ethnicity_groups)
print("=== EDDI for Insurance Groups (LOS) ===")
insurance_subgroups_los, insurance_eddi_los = print_subgroup_eddi(labels_arr[:, 1], preds_los, "Insurance Groups", insurance_groups)
overall_eddi_los = np.sqrt(age_eddi_los**2 + ethnicity_eddi_los**2 + insurance_eddi_los**2) / 3
print(f"\nOverall EDDI for LOS Predictions: {overall_eddi_los:.4f}\n")

# For Mechanical Ventilation outcome:
print("=== EDDI for Age Groups (Mechanical Ventilation) ===")
age_subgroups_mech, age_eddi_mech = print_subgroup_eddi(labels_arr[:, 2], preds_mech, "Age Groups", age_groups)
print("=== EDDI for Ethnicity Groups (Mechanical Ventilation) ===")
ethnicity_subgroups_mech, ethnicity_eddi_mech = print_subgroup_eddi(labels_arr[:, 2], preds_mech, "Ethnicity Groups", ethnicity_groups)
print("=== EDDI for Insurance Groups (Mechanical Ventilation) ===")
insurance_subgroups_mech, insurance_eddi_mech = print_subgroup_eddi(labels_arr[:, 2], preds_mech, "Insurance Groups", insurance_groups)
overall_eddi_mech = np.sqrt(age_eddi_mech**2 + ethnicity_eddi_mech**2 + insurance_eddi_mech**2) / 3
print(f"\nOverall EDDI for Mechanical Ventilation Predictions: {overall_eddi_mech:.4f}\n")

# Detailed EDDI for each age group.
age_group_list = ['15-29', '30-49', '50-69', '70-89']
print("Detailed EDDI values for each Age Group:")
for group in age_group_list:
    disparity = age_subgroups.get(group, None)
    if disparity is not None:
        print(f"  Age Group {group}: EDDI = {disparity:.4f}")
    else:
        print(f"  Age Group {group}: No data available")

# Combine attribute-level EDDI scores.
overall_eddi = np.sqrt(age_eddi**2 + ethnicity_eddi**2 + insurance_eddi**2) / 3
print(f"\nOverall EDDI for Model Predictions (Mortality): {overall_eddi:.4f}")

# Jitter Plot for Prediction Probabilities
def plot_prediction_jitter(predictions, labels, outcome_name):
    jitter = np.random.normal(0, 0.02, size=len(predictions))
    colors = ['blue' if label == 1 else 'red' for label in labels]
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, jitter, c=colors, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
    plt.title(f'Jitter Plot of Prediction Probabilities for {outcome_name}', fontsize=16)
    plt.xlabel('Prediction Probability', fontsize=14)
    plt.ylabel('Jitter (for visualization)', fontsize=14)
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.7)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='True Positive (1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='True Negative (0)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot jitter for each outcome.
outcome_names = {'mortality': 'Short-term Mortality', 'los': 'LOS > 3 Days', 'mech': 'Mechanical Ventilation'}
outcome_indices = {'mortality': 0, 'los': 1, 'mech': 2}

for key, name in outcome_names.items():
    print(f"\nJitter Plot for {name}")
    probs = expit(evaluation_results['logits'][key]).flatten()
    idx = outcome_indices[key]
    plot_prediction_jitter(predictions=probs,
                           labels=evaluation_results['labels'][:, idx],
                           outcome_name=name)
