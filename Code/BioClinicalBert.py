import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from tqdm import tqdm

# Define the dataset class
class ICUNotesDataset(Dataset):
    def __init__(self, data, label_column, tokenizer, max_length=512):
        self.data = data
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        note_chunks = [row[col] for col in self.data.columns if col.startswith('note_') and not pd.isnull(row[col])]
        label = row[self.label_column]

        # Tokenize each note chunk
        tokenized_chunks = [
            self.tokenizer(
                chunk,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            for chunk in note_chunks
        ]

        return {
            'input_ids': [tc['input_ids'].squeeze(0) for tc in tokenized_chunks],
            'attention_mask': [tc['attention_mask'].squeeze(0) for tc in tokenized_chunks],
            'label': torch.tensor(label, dtype=torch.float)
        }

# Define the BioClinicalBERT model
class BioClinicalBERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_chunks, num_classes=1):
        super(BioClinicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size * num_chunks, num_classes)

    def forward(self, input_ids_list, attention_mask_list):
        embeddings = []

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(pooled_output)

        # Concatenate embeddings from all note chunks
        concatenated_embeddings = torch.cat(embeddings, dim=1)

        # Pass concatenated embeddings through the classification layer
        logits = self.fc(concatenated_embeddings)
        return logits

# Function to compute class weights
def compute_class_weights(labels):
    unique_classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)

# Function to train the model
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids_list = [x.to(device) for x in batch['input_ids']]
        attention_mask_list = [x.to(device) for x in batch['attention_mask']]
        labels = batch['label'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(input_ids_list, attention_mask_list)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids_list = [x.to(device) for x in batch['input_ids']]
            attention_mask_list = [x.to(device) for x in batch['attention_mask']]
            labels = batch['label'].to(device).unsqueeze(1)

            outputs = model(input_ids_list, attention_mask_list)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds)

    # Calculate evaluation metrics
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'predictions': all_preds,
        'labels': all_labels
    }

# Function to calculate fairness metrics
def calculate_fairness_metrics(labels, predictions, sensitive_attribute):
    labels = np.array(labels)
    predictions = np.array(predictions)
    unique_groups = np.unique(sensitive_attribute)

    fairness_results = {}
    for group in unique_groups:
        group_idx = (sensitive_attribute == group)
        group_labels = labels[group_idx]
        group_predictions = predictions[group_idx]

        # Confusion Matrix
        cm = confusion_matrix(group_labels, (group_predictions > 0.5).astype(int))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calculate TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        fairness_results[group] = {'TPR': tpr, 'FPR': fpr}

    # Calculate disparities
    tpr_values = [v['TPR'] for v in fairness_results.values()]
    fpr_values = [v['FPR'] for v in fairness_results.values()]
    disparity_tpr = max(tpr_values) - min(tpr_values)
    disparity_fpr = max(fpr_values) - min(fpr_values)

    return {
        'equalized_odds': fairness_results,
        'disparity': {'TPR': disparity_tpr, 'FPR': disparity_fpr}
    }

# Load and preprocess the dataset
processed_data = pd.read_csv("processed_icu_notes.csv")
processed_data['note'] = processed_data[[col for col in processed_data.columns if col.startswith('note_')]].fillna('').agg(' '.join, axis=1)
label_column = 'short_term_mortality'
processed_data = processed_data.dropna(subset=[label_column])
processed_data[label_column] = processed_data[label_column].astype(int)

# Split the dataset
train_data = processed_data.sample(frac=0.7, random_state=42)
remaining_data = processed_data.drop(train_data.index)
val_data = remaining_data.sample(frac=0.5, random_state=42)
test_data = remaining_data.drop(val_data.index)

# Prepare datasets and data loaders
batch_size = 16
max_length = 512
num_chunks = len([col for col in processed_data.columns if col.startswith('note_')])  # Number of note chunks per patient
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
train_dataset = ICUNotesDataset(train_data, label_column, tokenizer, max_length)
val_dataset = ICUNotesDataset(val_data, label_column, tokenizer, max_length)
test_dataset = ICUNotesDataset(test_data, label_column, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BioClinicalBERTClassifier("emilyalsentzer/Bio_ClinicalBERT", num_chunks=num_chunks).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(train_data[label_column])[1])

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
    print(f"Validation Metrics: AUROC={val_metrics['auroc']:.4f}, AUPRC={val_metrics['auprc']:.4f}, "
          f"F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}, Precision={val_metrics['precision']:.4f}")

# Testing and Fairness Evaluation
test_metrics = evaluate_model(model, test_loader, device)
print(f"Test Metrics: AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}, "
      f"F1={test_metrics['f1']:.4f}, Recall={test_metrics['recall']:.4f}, Precision={test_metrics['precision']:.4f}")

# Fairness Evaluation
sensitive_attributes = {'Gender': 'GENDER', 'Ethnicity': 'ETHNICITY', 'Insurance': 'INSURANCE'}
for attr, col in sensitive_attributes.items():
    if col in processed_data.columns:
        sensitive_attribute = processed_data.loc[test_data.index, col].values
        fairness_results = calculate_fairness_metrics(test_metrics['labels'], test_metrics['predictions'], sensitive_attribute)
        print(f"\nFairness Evaluation for {attr}:")
        print("Equalized Odds (TPR, FPR):", fairness_results['equalized_odds'])
        print("Disparity:", fairness_results['disparity']")
