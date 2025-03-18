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

df = pd.read_csv('final_structured_common.csv')

unique_diseases = df['hadm_id'].unique()
disease_mapping = {d: i for i, d in enumerate(unique_diseases)}
df['mapped_disease_id'] = df['hadm_id'].map(disease_mapping)

def categorize_ethnicity(ethnicity):
    if pd.isna(ethnicity):
        return 'Other'
    ethnicity = ethnicity.upper().strip()
    if ethnicity in ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN']:
        return 'White'
    elif any(keyword in ethnicity for keyword in ['WHITE', 'EUROPEAN', 'RUSSIAN', 'BRAZILIAN', 'PORTUGUESE']):
        return 'White'
    elif ethnicity in ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN', 'BLACK/AFRICAN', 'CARIBBEAN ISLAND']:
        return 'Black'
    elif any(keyword in ethnicity for keyword in ['BLACK', 'AFRICAN', 'CAPE VERDEAN', 'HAITIAN']):
        return 'Black'
    elif ethnicity in ['HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - MEXICAN']:
        return 'Hispanic'
    elif any(keyword in ethnicity for keyword in ['HISPANIC', 'LATINO', 'GUATEMALAN', 'PUERTO RICAN', 'DOMINICAN', 'SALVADORAN', 'COLOMBIAN', 'MEXICAN', 'CUBAN', 'HONDURAN']):
        return 'Hispanic'
    elif ethnicity in ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - INDIAN']:
        return 'Asian'
    elif any(keyword in ethnicity for keyword in ['ASIAN', 'CHINESE', 'JAPANESE', 'VIETNAMESE', 'FILIPINO', 'THAI', 'KOREAN', 'CAMBODIAN', 'ASIAN INDIAN']):
        return 'Asian'
    else:
        return 'Other'

df['categorized_ethnicity'] = df['ETHNICITY'].apply(categorize_ethnicity)
df['categorized_ethnicity_code'] = df['categorized_ethnicity'].astype('category').cat.codes

df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
df['DEATHTIME'] = pd.to_datetime(df['DEATHTIME'], errors='coerce')

df['time_to_discharge'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600
df['time_to_death'] = (df['DEATHTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600

df['GENDER_CODE'] = df['GENDER'].astype('category').cat.codes
df['INSURANCE_CODE'] = df['INSURANCE'].astype('category').cat.codes

df['age'] = df['age'].fillna(0)

# Filter data (only include records with sufficient time span).
df_filtered = df[
    ((df['time_to_discharge'] > 6) & (df['short_term_mortality'] == 0)) |
    ((df['time_to_death'] > 6) & (df['short_term_mortality'] == 1))
].copy()

# Prepare Sequences Per Patient (For DfC, we exclude sensitive features from input)
def prepare_sequences(df):
    patients = df['subject_id'].unique()
    sequences = []
    labels = []
    sensitive_features = []  # Store sensitive attributes for evaluation only.
    patient_ids = []
    for patient in patients:
        patient_data = df[df['subject_id'] == patient].sort_values(by='ADMITTIME')
        # For DfC, use only non-sensitive features:
        # Use disease sequence and segment, admission, and discharge locations.
        disease_sequence = patient_data['mapped_disease_id'].tolist()
        segment_sequence = [0 if i % 2 == 0 else 1 for i in range(len(disease_sequence))]
        if 'FIRST_WARDID' in patient_data.columns:
            admission_loc_sequence = patient_data['FIRST_WARDID'].tolist()
        else:
            admission_loc_sequence = [0] * len(disease_sequence)
        if 'LAST_WARDID' in patient_data.columns:
            discharge_loc_sequence = patient_data['LAST_WARDID'].tolist()
        else:
            discharge_loc_sequence = [0] * len(disease_sequence)
        sequences.append({
            'diseases': disease_sequence,
            'segment': segment_sequence,
            'admission_loc': admission_loc_sequence,
            'discharge_loc': discharge_loc_sequence
        })
        mortality_label = patient_data['short_term_mortality'].max()
        los_label = patient_data['los_binary'].max()
        mech_label = patient_data['mechanical_ventilation'].max()
        labels.append([mortality_label, los_label, mech_label])
        # Store sensitive attributes (for evaluation only).
        age_val = patient_data['age'].iloc[0]
        gender_val = patient_data['GENDER_CODE'].iloc[0]
        ethnicity_val = patient_data['categorized_ethnicity_code'].iloc[0]
        insurance_val = patient_data['INSURANCE_CODE'].iloc[0]
        sensitive_features.append([age_val, gender_val, ethnicity_val, insurance_val])
        patient_ids.append(patient)
    return sequences, labels, sensitive_features, patient_ids

sequences, labels, sensitive_features, patient_ids = prepare_sequences(df_filtered)

# For model input, only use non-sensitive features.
input_ids = [seq['diseases'] for seq in sequences]
segment_ids = [seq['segment'] for seq in sequences]
admission_loc_ids = [seq['admission_loc'] for seq in sequences]
discharge_loc_ids = [seq['discharge_loc'] for seq in sequences]

# Convert sensitive features to a NumPy array (for later evaluation).
sensitive_features = np.array(sensitive_features)  # Columns: age, gender, ethnicity, insurance

# Pad sequences to a uniform length.
max_len = max(len(seq) for seq in input_ids)
def pad_sequences(sequences, max_len):
    return [seq + [0]*(max_len - len(seq)) for seq in sequences]

input_ids_padded = pad_sequences(input_ids, max_len)
segment_ids_padded = pad_sequences(segment_ids, max_len)
admission_loc_ids_padded = pad_sequences(admission_loc_ids, max_len)
discharge_loc_ids_padded = pad_sequences(discharge_loc_ids, max_len)

# Convert padded sequences to PyTorch tensors.
input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long)
segment_ids_tensor = torch.tensor(segment_ids_padded, dtype=torch.long)
admission_loc_ids_tensor = torch.tensor(admission_loc_ids_padded, dtype=torch.long)
discharge_loc_ids_tensor = torch.tensor(discharge_loc_ids_padded, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.float)

# Create dataset and dataloader.
dataset = TensorDataset(
    input_ids_tensor, segment_ids_tensor, admission_loc_ids_tensor, discharge_loc_ids_tensor, labels_tensor
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# DfC BEHRT Model (Demographic-Free Classification)
class DFC_BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_segments, num_admission_locs, num_discharge_locs, hidden_size=768):
        super(DFC_BEHRTModel, self).__init__()
        # Define a custom BERT configuration using only non-sensitive vocab.
        config = BertConfig(
            vocab_size = num_diseases + num_segments + num_admission_locs + num_discharge_locs + 2,
            hidden_size = hidden_size,
            num_hidden_layers = 12,
            num_attention_heads = 12,
            intermediate_size = 3072,
            max_position_embeddings = 512,
            type_vocab_size = 2,
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1
        )
        self.bert = BertModel(config)
        # Only use embeddings for non-sensitive features.
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.admission_loc_embedding = nn.Embedding(num_admission_locs, hidden_size)
        self.discharge_loc_embedding = nn.Embedding(num_discharge_locs, hidden_size)
        # Three classifiers for the outcomes.
        self.classifier_mortality = nn.Linear(hidden_size, 1)
        self.classifier_los = nn.Linear(hidden_size, 1)
        self.classifier_mech = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, segment_ids, admission_loc_ids, discharge_loc_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        # BERT forward pass.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        # Get non-sensitive embeddings.
        segment_embeds = self.segment_embedding(segment_ids)
        admission_embeds = self.admission_loc_embedding(admission_loc_ids)
        discharge_embeds = self.discharge_loc_embedding(discharge_loc_ids)
        # Combine by summing.
        combined_output = sequence_output + segment_embeds + admission_embeds + discharge_embeds
        # Use [CLS] token (first token) for classification.
        cls_rep = combined_output[:, 0, :]
        logits_mortality = self.classifier_mortality(cls_rep)
        logits_los = self.classifier_los(cls_rep)
        logits_mech = self.classifier_mech(cls_rep)
        return logits_mortality, logits_los, logits_mech


num_diseases = len(disease_mapping)
num_segments = 2
num_admission_locs = int(df_filtered['FIRST_WARDID'].nunique()) if 'FIRST_WARDID' in df_filtered.columns else 1
num_discharge_locs = int(df_filtered['LAST_WARDID'].nunique()) if 'LAST_WARDID' in df_filtered.columns else 1

model = DFC_BEHRTModel(
    num_diseases=num_diseases,
    num_segments=num_segments,
    num_admission_locs=num_admission_locs,
    num_discharge_locs=num_discharge_locs,
    hidden_size=768
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

class_weights_mortality = compute_class_weights(df_filtered, 'short_term_mortality')
class_weights_los = compute_class_weights(df_filtered, 'los_binary')
class_weights_mech = compute_class_weights(df_filtered, 'mechanical_ventilation')

class_weights_tensor_mortality = torch.tensor(class_weights_mortality.values, dtype=torch.float, device=device)
class_weights_tensor_los = torch.tensor(class_weights_los.values, dtype=torch.float, device=device)
class_weights_tensor_mech = torch.tensor(class_weights_mech.values, dtype=torch.float, device=device)

loss_fn_mortality = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_mortality[1])
loss_fn_los = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_los[1])
loss_fn_mech = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_mech[1])

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids_b, segment_ids_b, admission_loc_ids_b, discharge_loc_ids_b, labels_b = batch
        input_ids_b = input_ids_b.to(device)
        segment_ids_b = segment_ids_b.to(device)
        admission_loc_ids_b = admission_loc_ids_b.to(device)
        discharge_loc_ids_b = discharge_loc_ids_b.to(device)
        labels_b = labels_b.to(device)
        
        optimizer.zero_grad()
        logits_mort, logits_los, logits_mech = model(
            input_ids=input_ids_b,
            segment_ids=segment_ids_b,
            admission_loc_ids=admission_loc_ids_b,
            discharge_loc_ids=discharge_loc_ids_b
        )
        loss_mort = loss_fn_mortality(logits_mort, labels_b[:, 0].unsqueeze(1))
        loss_los = loss_fn_los(logits_los, labels_b[:, 1].unsqueeze(1))
        loss_mech = loss_fn_mech(logits_mech, labels_b[:, 2].unsqueeze(1))
        loss = loss_mort + loss_los + loss_mech
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
            input_ids_b, segment_ids_b, admission_loc_ids_b, discharge_loc_ids_b, labels_b = batch
            input_ids_b = input_ids_b.to(device)
            segment_ids_b = segment_ids_b.to(device)
            admission_loc_ids_b = admission_loc_ids_b.to(device)
            discharge_loc_ids_b = discharge_loc_ids_b.to(device)
            labels_b = labels_b.to(device)
            
            logits_mort, logits_los, logits_mech = model(
                input_ids=input_ids_b,
                segment_ids=segment_ids_b,
                admission_loc_ids=admission_loc_ids_b,
                discharge_loc_ids=discharge_loc_ids_b
            )
            all_logits['mortality'].append(logits_mort.cpu().numpy())
            all_logits['los'].append(logits_los.cpu().numpy())
            all_logits['mech'].append(logits_mech.cpu().numpy())
            all_labels.append(labels_b.cpu().numpy())
            
            preds_mort = (torch.sigmoid(logits_mort) > 0.5).cpu().numpy().astype(int)
            preds_los = (torch.sigmoid(logits_los) > 0.5).cpu().numpy().astype(int)
            preds_mech = (torch.sigmoid(logits_mech) > 0.5).cpu().numpy().astype(int)
            all_predictions['mortality'].append(preds_mort)
            all_predictions['los'].append(preds_los)
            all_predictions['mech'].append(preds_mech)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_logits['mortality'] = np.concatenate(all_logits['mortality'], axis=0)
    all_logits['los'] = np.concatenate(all_logits['los'], axis=0)
    all_logits['mech'] = np.concatenate(all_logits['mech'], axis=0)
    all_predictions['mortality'] = np.concatenate(all_predictions['mortality'], axis=0)
    all_predictions['los'] = np.concatenate(all_predictions['los'], axis=0)
    all_predictions['mech'] = np.concatenate(all_predictions['mech'], axis=0)
    
    auroc_mort = roc_auc_score(all_labels[:, 0], all_logits['mortality'])
    auroc_los = roc_auc_score(all_labels[:, 1], all_logits['los'])
    auroc_mech = roc_auc_score(all_labels[:, 2], all_logits['mech'])
    
    precision_mort, recall_mort, _ = precision_recall_curve(all_labels[:, 0], all_logits['mortality'])
    auprc_mort = auc(recall_mort, precision_mort)
    precision_los, recall_los, _ = precision_recall_curve(all_labels[:, 1], all_logits['los'])
    auprc_los = auc(recall_los, precision_los)
    precision_mech, recall_mech, _ = precision_recall_curve(all_labels[:, 2], all_logits['mech'])
    auprc_mech = auc(recall_mech, precision_mech)
    
    precision_mort_score = precision_score(all_labels[:, 0], all_predictions['mortality'])
    recall_mort_score = recall_score(all_labels[:, 0], all_predictions['mortality'])
    f1_mort = f1_score(all_labels[:, 0], all_predictions['mortality'])
    
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
        'auroc': {'mortality': auroc_mort, 'los': auroc_los, 'mech': auroc_mech},
        'auprc': {'mortality': auprc_mort, 'los': auprc_los, 'mech': auprc_mech},
        'precision': {'mortality': precision_mort_score, 'los': precision_los_score, 'mech': precision_mech_score},
        'recall': {'mortality': recall_mort_score, 'los': recall_los_score, 'mech': recall_mech_score},
        'f1': {'mortality': f1_mort, 'los': f1_los, 'mech': f1_mech}
    }

evaluation_results = evaluate_model(model, dataloader, device)
print("Evaluation Results:")
print(evaluation_results)
