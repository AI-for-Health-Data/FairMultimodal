import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertModel, BertConfig
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import chi2_contingency, ttest_ind

# 1. Load Dataset and Preprocessing
df = pd.read_csv('structured_first_admissions.csv')

# Encode disease codes as numerical values
unique_diseases = df['DIAGNOSIS'].unique()
disease_mapping = {disease: i for i, disease in enumerate(unique_diseases)}
df['disease_code'] = df['DIAGNOSIS'].map(disease_mapping)

# Encode categorical features
df['ADMISSION_LOCATION'] = df['ADMISSION_LOCATION'].astype('category').cat.codes
df['DISCHARGE_LOCATION'] = df['DISCHARGE_LOCATION'].astype('category').cat.codes
df['INSURANCE'] = df['INSURANCE'].astype('category').cat.codes
df['GENDER'] = df['GENDER'].astype('category').cat.codes

# Convert time columns to datetime
df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
df['DEATHTIME'] = pd.to_datetime(df['DEATHTIME'], errors='coerce')

# Calculate time differences in hours
df['time_to_discharge'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600
df['time_to_death'] = (df['DEATHTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600

# Ensure consistency with the original dataset logic
def preprocess_data(df):
    # Maintain the first admission per patient
    df = df.sort_values(by=['subject_id', 'ADMITTIME']).groupby('subject_id').first().reset_index()
    
    # Ensure consistent encoding for categories
    df['categorized_ethnicity'] = df[['categorized_ethnicity_White',
                                       'categorized_ethnicity_Black',
                                       'categorized_ethnicity_Hispanic',
                                       'categorized_ethnicity_Asian',
                                       'categorized_ethnicity_Other']].idxmax(axis=1)

    df['categorized_insurance'] = df[['categorized_insurance_Private',
                                       'categorized_insurance_Medicaid',
                                       'categorized_insurance_Medicare',
                                       'categorized_insurance_Self Pay',
                                       'categorized_insurance_Government']].idxmax(axis=1)

    return df

df = preprocess_data(df)

# Filter records up to 6 hours before discharge or death
df_filtered = df[
    (
        (df['time_to_discharge'] > 6) & 
        (df['short_term_mortality'] == 0) &
        (df['ventilation_within_6_hours'] == 0)
    ) | 
    (
        (df['time_to_death'] > 6) & 
        (df['short_term_mortality'] == 1)
    ) |
    (
        (df['ventilation_within_6_hours'] == 1)
    )
]

# Function to compute class weights using the Inverse of Number of Samples (INS)
def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

# Compute class weights for each outcome using INS
class_weights_mortality = compute_class_weights(df_filtered, 'short_term_mortality')
class_weights_readmission = compute_class_weights(df_filtered, 'readmitted_within_30_days')
class_weights_ventilation = compute_class_weights(df_filtered, 'ventilation_within_6_hours')

# Convert class weights to tensors for PyTorch loss functions
class_weights_tensor_mortality = torch.tensor(class_weights_mortality.values, dtype=torch.float).to('cpu')
class_weights_tensor_readmission = torch.tensor(class_weights_readmission.values, dtype=torch.float).to('cpu')
class_weights_tensor_ventilation = torch.tensor(class_weights_ventilation.values, dtype=torch.float).to('cpu')

# Function to count positive and negative cases
def count_positive_negative_cases(df, columns):
    for column in columns:
        positive_cases = df[column].sum()
        negative_cases = len(df) - positive_cases
        print(f"{column} - Positive Cases: {positive_cases}, Negative Cases: {negative_cases}")

# Define outcome columns
outcome_columns = ['short_term_mortality', 'readmitted_within_30_days', 'ventilation_within_6_hours']

# Count positive and negative cases for each outcome
count_positive_negative_cases(df_filtered, outcome_columns)

# Ensure numerical encoding for ethnicity and insurance fields
df_filtered = df_filtered.copy()  # Create a copy to avoid the warning
df_filtered.loc[:, 'categorized_ethnicity'] = df_filtered['categorized_ethnicity'].astype('category').cat.codes
df_filtered.loc[:, 'categorized_insurance'] = df_filtered['categorized_insurance'].astype('category').cat.codes

# 2. Prepare Sequences for Model Input
def prepare_sequences(df):
    patients = df['subject_id'].unique()
    sequences = []
    labels = []
    patient_ids = []

    for patient in patients:
        patient_data = df[df['subject_id'] == patient].sort_values(by='ADMITTIME')
        
        age_sequence = patient_data['age'].tolist()
        disease_sequence = patient_data['disease_code'].tolist()
        admission_loc_sequence = patient_data['ADMISSION_LOCATION'].tolist()
        discharge_loc_sequence = patient_data['DISCHARGE_LOCATION'].tolist()
        segment_sequence = [0 if i % 2 == 0 else 1 for i in range(len(age_sequence))]

        short_term_mortality_label = patient_data['short_term_mortality'].max()
        readmission_label = patient_data['readmitted_within_30_days'].max()
        mechanical_ventilation_label = patient_data['ventilation_within_6_hours'].max()

        sequences.append({
            'age': age_sequence,
            'diseases': disease_sequence,
            'admission_loc': admission_loc_sequence,
            'discharge_loc': discharge_loc_sequence,
            'segment': segment_sequence,
            'gender': patient_data['GENDER'].tolist(),
            'ethnicity': patient_data['categorized_ethnicity'].tolist(),
            'insurance': patient_data['categorized_insurance'].tolist()
        })
        labels.append([short_term_mortality_label, readmission_label, mechanical_ventilation_label])
        patient_ids.append(patient)

    return sequences, labels, patient_ids

sequences, labels, patient_ids = prepare_sequences(df_filtered)

# Tokenize sequences and encode them
input_ids = []
age_ids = []
segment_ids = []
admission_loc_ids = []
discharge_loc_ids = []
gender_ids = []
ethnicity_ids = []
insurance_ids = []

for seq in sequences:
    token_ids = seq['diseases']
    age_sequence = seq['age']
    segment_sequence = seq['segment']
    admission_loc_sequence = seq['admission_loc']
    discharge_loc_sequence = seq['discharge_loc']
    gender_sequence = seq['gender']
    ethnicity_sequence = seq['ethnicity']
    insurance_sequence = seq['insurance']
    
    input_ids.append(token_ids)
    age_ids.append(age_sequence)
    segment_ids.append(segment_sequence)
    admission_loc_ids.append(admission_loc_sequence)
    discharge_loc_ids.append(discharge_loc_sequence)
    gender_ids.append(gender_sequence)
    ethnicity_ids.append(ethnicity_sequence)
    insurance_ids.append(insurance_sequence)

# Determine maximum sequence length and pad all sequences
max_len = max(len(seq) for seq in input_ids)

def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) for seq in sequences]

input_ids_padded = pad_sequences(input_ids, max_len)
age_ids_padded = pad_sequences(age_ids, max_len)
segment_ids_padded = pad_sequences(segment_ids, max_len)
admission_loc_ids_padded = pad_sequences(admission_loc_ids, max_len)
discharge_loc_ids_padded = pad_sequences(discharge_loc_ids, max_len)
gender_ids_padded = pad_sequences(gender_ids, max_len)
ethnicity_ids_padded = pad_sequences(ethnicity_ids, max_len)
insurance_ids_padded = pad_sequences(insurance_ids, max_len)

# Convert to PyTorch tensors
input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long)
age_ids_tensor = torch.tensor(age_ids_padded, dtype=torch.long)
segment_ids_tensor = torch.tensor(segment_ids_padded, dtype=torch.long)
admission_loc_ids_tensor = torch.tensor(admission_loc_ids_padded, dtype=torch.long)
discharge_loc_ids_tensor = torch.tensor(discharge_loc_ids_padded, dtype=torch.long)
gender_ids_tensor = torch.tensor(gender_ids_padded, dtype=torch.long)
ethnicity_ids_tensor = torch.tensor(ethnicity_ids_padded, dtype=torch.long)
insurance_ids_tensor = torch.tensor(insurance_ids_padded, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.float)

# Creating dataset and dataloader
dataset = TensorDataset(
    input_ids_tensor, age_ids_tensor, segment_ids_tensor,
    admission_loc_ids_tensor, discharge_loc_ids_tensor, 
    gender_ids_tensor, ethnicity_ids_tensor, insurance_ids_tensor, 
    labels_tensor
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define BEHRT Model
class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs, 
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()

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
        
        # Add embeddings for new demographic features
        self.gender_embedding = nn.Embedding(num_genders, hidden_size)
        self.ethnicity_embedding = nn.Embedding(num_ethnicities, hidden_size)
        self.insurance_embedding = nn.Embedding(num_insurances, hidden_size)

        # Three classifiers for three different binary target variables
        self.classifier_mortality = nn.Linear(hidden_size, 1)
        self.classifier_readmission = nn.Linear(hidden_size, 1)
        self.classifier_ventilation = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, 
                gender_ids, ethnicity_ids, insurance_ids, attention_mask=None):
        
        # Clamp IDs to valid ranges
        age_ids = torch.clamp(age_ids, min=0, max=self.age_embedding.num_embeddings - 1)
        segment_ids = torch.clamp(segment_ids, min=0, max=self.segment_embedding.num_embeddings - 1)
        admission_loc_ids = torch.clamp(admission_loc_ids, min=0, max=self.admission_loc_embedding.num_embeddings - 1)
        discharge_loc_ids = torch.clamp(discharge_loc_ids, min=0, max=self.discharge_loc_embedding.num_embeddings - 1)
        gender_ids = torch.clamp(gender_ids, min=0, max=self.gender_embedding.num_embeddings - 1)
        ethnicity_ids = torch.clamp(ethnicity_ids, min=0, max=self.ethnicity_embedding.num_embeddings - 1)
        insurance_ids = torch.clamp(insurance_ids, min=0, max=self.insurance_embedding.num_embeddings - 1)

        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Get embeddings
        age_embeds = self.age_embedding(age_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        admission_loc_embeds = self.admission_loc_embedding(admission_loc_ids)
        discharge_loc_embeds = self.discharge_loc_embedding(discharge_loc_ids)
        gender_embeds = self.gender_embedding(gender_ids)
        ethnicity_embeds = self.ethnicity_embedding(ethnicity_ids)
        insurance_embeds = self.insurance_embedding(insurance_ids)

        # Combine embeddings
        combined_output = (sequence_output + age_embeds + segment_embeds + 
                           admission_loc_embeds + discharge_loc_embeds + 
                           gender_embeds + ethnicity_embeds + insurance_embeds)

        # Classifier outputs
        logits_mortality = self.classifier_mortality(combined_output[:, 0, :])
        logits_readmission = self.classifier_readmission(combined_output[:, 0, :])
        logits_ventilation = self.classifier_ventilation(combined_output[:, 0, :])

        return logits_mortality, logits_readmission, logits_ventilation

# Initialize the model
num_diseases = len(disease_mapping)
num_ages = df_filtered['age'].nunique()
num_segments = 2
num_admission_locs = df_filtered['ADMISSION_LOCATION'].nunique()
num_discharge_locs = df_filtered['DISCHARGE_LOCATION'].nunique()
num_genders = df_filtered['GENDER'].nunique()
num_ethnicities = df_filtered['ETHNICITY'].nunique()
num_insurances = df_filtered['INSURANCE'].nunique()

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

# Move the model to the available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

# Define weighted binary cross-entropy loss for the imbalanced classes
bce_loss_mortality = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_mortality[1])  
bce_loss_readmission = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_readmission[1])  
bce_loss_ventilation = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_ventilation[1])  

# 4. Training Loop
epochs = 30
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

        # Generate attention mask
        attention_mask = (input_ids != 0).long().to(device)

        optimizer.zero_grad()

        # Forward pass
        logits_mortality, logits_readmission, logits_ventilation = model(
            input_ids=input_ids, 
            age_ids=age_ids, 
            segment_ids=segment_ids, 
            admission_loc_ids=admission_loc_ids, 
            discharge_loc_ids=discharge_loc_ids,
            gender_ids=gender_ids,
            ethnicity_ids=ethnicity_ids,
            insurance_ids=insurance_ids,
            attention_mask=attention_mask  # Include attention mask
        )

        # Compute weighted loss for each target
        loss_mortality = bce_loss_mortality(logits_mortality, labels[:, 0].unsqueeze(1))
        loss_readmission = bce_loss_readmission(logits_readmission, labels[:, 1].unsqueeze(1))
        loss_ventilation = bce_loss_ventilation(logits_ventilation, labels[:, 2].unsqueeze(1))

        # Combine losses
        total_loss_batch = loss_mortality + loss_readmission + loss_ventilation
        total_loss_batch.backward()

        optimizer.step()
        total_loss += total_loss_batch.item()

    print(f"Epoch {epoch + 1} - Total Loss: {total_loss:.4f}")

# 5. Evaluation Function

def evaluate_model(model, dataloader, device):
    model.eval()  
    all_labels = []
    all_logits = {'mortality': [], 'readmission': [], 'ventilation': []}
    all_predictions = {'mortality': [], 'readmission': [], 'ventilation': []}

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

            # Forward pass through the model
            logits_mortality, logits_readmission, logits_ventilation = model(
                input_ids=input_ids, 
                age_ids=age_ids, 
                segment_ids=segment_ids, 
                admission_loc_ids=admission_loc_ids, 
                discharge_loc_ids=discharge_loc_ids,
                gender_ids=gender_ids,
                ethnicity_ids=ethnicity_ids,
                insurance_ids=insurance_ids
            )

            # Collect logits and labels
            all_logits['mortality'].append(logits_mortality.cpu().numpy())
            all_logits['readmission'].append(logits_readmission.cpu().numpy())
            all_logits['ventilation'].append(logits_ventilation.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

            # Convert logits to class predictions
            pred_mortality = (torch.sigmoid(logits_mortality) > 0.5).cpu().numpy().astype(int)
            pred_readmission = (torch.sigmoid(logits_readmission) > 0.5).cpu().numpy().astype(int)
            pred_ventilation = (torch.sigmoid(logits_ventilation) > 0.5).cpu().numpy().astype(int)

            # Append predictions
            all_predictions['mortality'].append(pred_mortality)
            all_predictions['readmission'].append(pred_readmission)
            all_predictions['ventilation'].append(pred_ventilation)

    # Concatenate all collected data
    all_labels = np.concatenate(all_labels, axis=0)
    all_logits['mortality'] = np.concatenate(all_logits['mortality'], axis=0)
    all_logits['readmission'] = np.concatenate(all_logits['readmission'], axis=0)
    all_logits['ventilation'] = np.concatenate(all_logits['ventilation'], axis=0)

    all_predictions['mortality'] = np.concatenate(all_predictions['mortality'], axis=0)
    all_predictions['readmission'] = np.concatenate(all_predictions['readmission'], axis=0)
    all_predictions['ventilation'] = np.concatenate(all_predictions['ventilation'], axis=0)

    # Compute AUROC for each task
    auroc_mortality = roc_auc_score(all_labels[:, 0], all_logits['mortality'])
    auroc_readmission = roc_auc_score(all_labels[:, 1], all_logits['readmission'])
    auroc_ventilation = roc_auc_score(all_labels[:, 2], all_logits['ventilation'])

    # Compute AUPRC for each task
    precision_mortality, recall_mortality, _ = precision_recall_curve(all_labels[:, 0], all_logits['mortality'])
    auprc_mortality = auc(recall_mortality, precision_mortality)

    precision_readmission, recall_readmission, _ = precision_recall_curve(all_labels[:, 1], all_logits['readmission'])
    auprc_readmission = auc(recall_readmission, precision_readmission)

    precision_ventilation, recall_ventilation, _ = precision_recall_curve(all_labels[:, 2], all_logits['ventilation'])
    auprc_ventilation = auc(recall_ventilation, precision_ventilation)

    # Precision, Recall, and F1 scores for each target
    precision_mortality_score = precision_score(all_labels[:, 0], all_predictions['mortality'])
    recall_mortality_score = recall_score(all_labels[:, 0], all_predictions['mortality'])
    f1_mortality = f1_score(all_labels[:, 0], all_predictions['mortality'])

    precision_readmission_score = precision_score(all_labels[:, 1], all_predictions['readmission'])
    recall_readmission_score = recall_score(all_labels[:, 1], all_predictions['readmission'])
    f1_readmission = f1_score(all_labels[:, 1], all_predictions['readmission'])

    precision_ventilation_score = precision_score(all_labels[:, 2], all_predictions['ventilation'])
    recall_ventilation_score = recall_score(all_labels[:, 2], all_predictions['ventilation'])
    f1_ventilation = f1_score(all_labels[:, 2], all_predictions['ventilation'])

    # Return results, including 'labels'
    return {
        'logits': all_logits,
        'predictions': all_predictions,
        'labels': all_labels,
        'auroc': {
            'mortality': auroc_mortality,
            'readmission': auroc_readmission,
            'ventilation': auroc_ventilation
        },
        'auprc': {
            'mortality': auprc_mortality,
            'readmission': auprc_readmission,
            'ventilation': auprc_ventilation
        },
        'precision': {
            'mortality': precision_mortality_score,
            'readmission': precision_readmission_score,
            'ventilation': precision_ventilation_score
        },
        'recall': {
            'mortality': recall_mortality_score,
            'readmission': recall_readmission_score,
            'ventilation': recall_ventilation_score
        },
        'f1': {
            'mortality': f1_mortality,
            'readmission': f1_readmission,
            'ventilation': f1_ventilation
        }
    }
    
# Run evaluation
evaluation_results = evaluate_model(model, dataloader, device)
print(evaluation_results)

# Helper function to get predictions from the model
def get_model_predictions(model, dataloader, device):
    all_predictions = {'mortality': [], 'readmission': [], 'ventilation': []}
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack 7 tensors: input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, gender_ids, ethnicity_ids, insurance_ids, and labels
            input_ids, age_ids, segment_ids, admission_loc_ids, discharge_loc_ids, gender_ids, ethnicity_ids, insurance_ids, labels = batch
            
            # Move tensors to the appropriate device
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            segment_ids = segment_ids.to(device)
            admission_loc_ids = admission_loc_ids.to(device)
            discharge_loc_ids = discharge_loc_ids.to(device)
            gender_ids = gender_ids.to(device)
            ethnicity_ids = ethnicity_ids.to(device)
            insurance_ids = insurance_ids.to(device)
            labels = labels.to(device)

            # Forward pass through the model
            logits_mortality, logits_readmission, logits_ventilation = model(
                input_ids=input_ids, 
                age_ids=age_ids, 
                segment_ids=segment_ids, 
                admission_loc_ids=admission_loc_ids, 
                discharge_loc_ids=discharge_loc_ids, 
                gender_ids=gender_ids, 
                ethnicity_ids=ethnicity_ids,
                insurance_ids=insurance_ids
            )
            
            # Convert logits to binary predictions (0 or 1)
            pred_mortality = (torch.sigmoid(logits_mortality) > 0.5).cpu().numpy().astype(int)
            pred_readmission = (torch.sigmoid(logits_readmission) > 0.5).cpu().numpy().astype(int)
            pred_ventilation = (torch.sigmoid(logits_ventilation) > 0.5).cpu().numpy().astype(int)

            all_predictions['mortality'].extend(pred_mortality)
            all_predictions['readmission'].extend(pred_readmission)
            all_predictions['ventilation'].extend(pred_ventilation)
            all_labels.append(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_predictions, all_labels

# Run model prediction after training
predictions, labels = get_model_predictions(model, dataloader, device)

# Define the mapping for detailed ethnicity classifications
ethnicity_mapping = {
    'WHITE': 'White',
    'WHITE - BRAZILIAN': 'White',
    'WHITE - EASTERN EUROPEAN': 'White',
    'WHITE - OTHER EUROPEAN': 'White',
    'WHITE - RUSSIAN': 'White',
    'BLACK/AFRICAN': 'Black',
    'BLACK/AFRICAN AMERICAN': 'Black',
    'BLACK/CAPE VERDEAN': 'Black',
    'BLACK/HAITIAN': 'Black',
    'ASIAN': 'Asian',
    'ASIAN - ASIAN INDIAN': 'Asian',
    'ASIAN - CAMBODIAN': 'Asian',
    'ASIAN - CHINESE': 'Asian',
    'ASIAN - FILIPINO': 'Asian',
    'ASIAN - JAPANESE': 'Asian',
    'ASIAN - KOREAN': 'Asian',
    'ASIAN - OTHER': 'Asian',
    'ASIAN - THAI': 'Asian',
    'ASIAN - VIETNAMESE': 'Asian',
    'HISPANIC OR LATINO': 'Hispanic',
    'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 'Hispanic',
    'HISPANIC/LATINO - COLOMBIAN': 'Hispanic',
    'HISPANIC/LATINO - CUBAN': 'Hispanic',
    'HISPANIC/LATINO - DOMINICAN': 'Hispanic',
    'HISPANIC/LATINO - GUATEMALAN': 'Hispanic',
    'HISPANIC/LATINO - HONDURAN': 'Hispanic',
    'HISPANIC/LATINO - MEXICAN': 'Hispanic',
    'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic',
    'HISPANIC/LATINO - SALVADORAN': 'Hispanic',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other',
    'AMERICAN INDIAN/ALASKA NATIVE': 'Other',
    'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 'Other',
    'MIDDLE EASTERN': 'Other',
    'MULTI RACE ETHNICITY': 'Other',
    'PORTUGUESE': 'Other',
    'SOUTH AMERICAN': 'Other',
    'CARIBBEAN ISLAND': 'Other',
    'OTHER': 'Other',
    'PATIENT DECLINED TO ANSWER': 'Other',
    'UNABLE TO OBTAIN': 'Other',
    'UNKNOWN/NOT SPECIFIED': 'Other'
}

# Apply the mapping to the ETHNICITY column
df_filtered['categorized_ethnicity'] = df_filtered['ETHNICITY'].map(ethnicity_mapping)

# Check for any unmapped values and assign them to 'Other'
df_filtered['categorized_ethnicity'] = df_filtered['categorized_ethnicity'].fillna('Other')

# Verify the mapping
print(df_filtered['categorized_ethnicity'].value_counts())

import numpy as np
from sklearn.metrics import confusion_matrix

# Ensure predictions and labels are NumPy arrays
predictions['mortality'] = np.array(predictions['mortality'])
predictions['readmission'] = np.array(predictions['readmission'])
predictions['ventilation'] = np.array(predictions['ventilation'])
labels = np.array(labels)

# Function to calculate Equalized Odds (TPR and FPR) with error handling
def calculate_equalized_odds(labels, predictions, sensitive_attribute):
    unique_groups = np.unique(sensitive_attribute)
    equalized_odds = {}

    for group in unique_groups:
        group_idx = (sensitive_attribute == group)  # Boolean indexing for the group
        group_labels = labels[group_idx]
        group_predictions = predictions[group_idx]

        # Calculate confusion matrix with handling for fewer than 4 values
        cm = confusion_matrix(group_labels, group_predictions)
        tn, fp, fn, tp = (0, 0, 0, 0)  # Default values in case of missing classes
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # Single class case (either all true negatives or all true positives)
            if group_labels[0] == 0:
                tn = cm[0][0]
            else:
                tp = cm[0][0]

        # Calculate TPR and FPR with safeguards for division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

        equalized_odds[group] = {'TPR': tpr, 'FPR': fpr}

    return equalized_odds

# Function to calculate Equal Opportunity (TPR only)
def calculate_equal_opportunity(labels, predictions, sensitive_attribute):
    unique_groups = np.unique(sensitive_attribute)
    equal_opportunity = {}

    for group in unique_groups:
        group_idx = (sensitive_attribute == group)  # Boolean indexing for the group
        group_labels = labels[group_idx]
        group_predictions = predictions[group_idx]

        # Calculate confusion matrix and handle missing values
        cm = confusion_matrix(group_labels, group_predictions)
        tn, fp, fn, tp = (0, 0, 0, 0)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            if group_labels[0] == 0:
                tn = cm[0][0]
            else:
                tp = cm[0][0]

        # Calculate TPR with a safeguard for division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate

        equal_opportunity[group] = {'TPR': tpr}

    return equal_opportunity

# Function to calculate Disparity (Max - Min) across groups
def calculate_disparity(metric_values):
    return max(metric_values) - min(metric_values)

# Define function for fairness evaluation
def evaluate_fairness(labels, predictions, sensitive_attribute):
    unique_groups = np.unique(sensitive_attribute)
    fairness_results = {"equalized_odds": {}, "equal_opportunity": {}, "disparity": {}}

    tpr_values = {}
    fpr_values = {}

    for group in unique_groups:
        group_indices = sensitive_attribute == group
        group_labels = labels[group_indices]
        group_predictions = predictions[group_indices]

        # Compute TPR (True Positive Rate) and FPR (False Positive Rate)
        tn, fp, fn, tp = confusion_matrix(group_labels, group_predictions, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        fairness_results["equalized_odds"][group] = {"TPR": tpr, "FPR": fpr}
        fairness_results["equal_opportunity"][group] = {"TPR": tpr}

        # Store for disparity calculation
        tpr_values[group] = tpr
        fpr_values[group] = fpr

    # Calculate disparities
    tpr_min, tpr_max = min(tpr_values.values()), max(tpr_values.values())
    fpr_min, fpr_max = min(fpr_values.values()), max(fpr_values.values())
    fairness_results["disparity"]["TPR"] = tpr_max - tpr_min
    fairness_results["disparity"]["FPR"] = fpr_max - fpr_min

    return fairness_results

# Fairness Evaluation for Gender
sensitive_attribute_gender = df_filtered['GENDER'].values
fairness_results_gender = evaluate_fairness(
    labels[:, 0], predictions['mortality'], sensitive_attribute_gender
)
print("Fairness Evaluation - Gender:")
print(fairness_results_gender)

# Fairness Evaluation for Categorized Ethnicity
sensitive_attribute_ethnicity = df_filtered['categorized_ethnicity'].values
fairness_results_ethnicity = evaluate_fairness(
    labels[:, 0], predictions['mortality'], sensitive_attribute_ethnicity
)
print("\nFairness Evaluation - Ethnicity:")
print(fairness_results_ethnicity)

# Fairness Evaluation for Insurance
sensitive_attribute_insurance = df_filtered['INSURANCE'].values
fairness_results_insurance = evaluate_fairness(
    labels[:, 0], predictions['mortality'], sensitive_attribute_insurance
)
print("\nFairness Evaluation - Insurance:")
print(fairness_results_insurance)

