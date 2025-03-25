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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.special import expit  # for logistic sigmoid
from skmultilearn.model_selection import iterative_train_test_split


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
        # Extract the CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

# Apply BioClinicalBERT on Patient Notes
def apply_bioclinicalbert_on_patient_notes(df, note_columns, tokenizer, model, device, aggregation="mean", max_length=512):
    patient_ids = df["subject_id"].unique()
    aggregated_embeddings = []
    for pid in tqdm(patient_ids, desc="Aggregating text embeddings"):
        patient_data = df[df["subject_id"] == pid]
        notes = []
        for col in note_columns:
            vals = patient_data[col].dropna().tolist()
            notes.extend([v for v in vals if isinstance(v, str) and v.strip() != ""])
        if len(notes) == 0:
            # Use a vector of zeros if no valid note exists
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

# Unstructured Dataset Definition
class UnstructuredDataset(Dataset):
    def __init__(self, embeddings, mortality_labels, los_labels, mech_labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.mortality_labels = torch.tensor(mortality_labels, dtype=torch.float32).unsqueeze(1)
        self.los_labels = torch.tensor(los_labels, dtype=torch.float32).unsqueeze(1)
        self.mech_labels = torch.tensor(mech_labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.mortality_labels[idx], self.los_labels[idx], self.mech_labels[idx]

# Unstructured Classifier Definition
class UnstructuredClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super(UnstructuredClassifier, self).__init__()
        # Output 3 logits: mortality, los, mechanical ventilation.
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)  
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits

# Training function for one epoch
def train_model(model, dataloader, optimizer, device, criterion_mort, criterion_los, criterion_mech):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        embeddings, labels_mort, labels_los, labels_mech = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(embeddings)  # Shape: (batch_size, 3)
        loss_mort = criterion_mort(logits[:, 0].unsqueeze(1), labels_mort)
        loss_los = criterion_los(logits[:, 1].unsqueeze(1), labels_los)
        loss_mech = criterion_mech(logits[:, 2].unsqueeze(1), labels_mech)
        loss = loss_mort + loss_los + loss_mech
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Evaluation function to compute metrics
def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels_mort, labels_los, labels_mech = [x.to(device) for x in batch]
            logits = model(embeddings)
            all_logits.append(logits.cpu())
            batch_labels = torch.cat((labels_mort.cpu(), labels_los.cpu(), labels_mech.cpu()), dim=1)
            all_labels.append(batch_labels)
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = expit(all_logits)
    preds = (probs >= threshold).astype(int)
    
    metrics = {}
    tasks = ["mortality", "los", "mech"]
    for i, task in enumerate(tasks):
        # Compute AUROC and AUPRC (if possible)
        try:
            aucroc = roc_auc_score(all_labels[:, i], probs[:, i])
        except Exception:
            aucroc = float('nan')
        try:
            auprc = average_precision_score(all_labels[:, i], probs[:, i])
        except Exception:
            auprc = float('nan')
        # Compute F1, Recall, and Precision using predictions
        f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        recall_metric = recall_score(all_labels[:, i], preds[:, i], zero_division=0)
        precision = precision_score(all_labels[:, i], preds[:, i], zero_division=0)
        # Calculate confusion matrix for TPR and FPR
        tn, fp, fn, tp = confusion_matrix(all_labels[:, i], preds[:, i]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics[task] = {
            "aucroc": aucroc,
            "auprc": auprc,
            "f1": f1,
            "recall": recall_metric,
            "precision": precision,
            "tpr": tpr,
            "fpr": fpr
        }
    return metrics

# Evaluation loss function for validation during training
def evaluate_loss(model, dataloader, device, criterion_mort, criterion_los, criterion_mech):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels_mort, labels_los, labels_mech = [x.to(device) for x in batch]
            logits = model(embeddings)
            loss_mort = criterion_mort(logits[:, 0].unsqueeze(1), labels_mort)
            loss_los = criterion_los(logits[:, 1].unsqueeze(1), labels_los)
            loss_mech = criterion_mech(logits[:, 2].unsqueeze(1), labels_mech)
            loss = loss_mort + loss_los + loss_mech
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else None

# Utility function to get patient probabilities
def get_patient_probabilities(model, dataloader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, _, _, _ = [x.to(device) for x in batch]
            logits = model(embeddings)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    probs = expit(all_logits)
    return probs

# Demographic and Fairness Utilities
def assign_age_bucket(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    elif 70 <= age <= 89:
        return '70-89'
    else:
        return 'other'

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

def print_detailed_eddi(y_true, y_pred, outcome_name, df_results, age_order, ethnicity_order, insurance_order):
    print(f"\n--- Detailed EDDI for {outcome_name} Outcome ---")
    overall_age, age_sub = compute_eddi(y_true, y_pred, df_results['age_bucket'].values)
    overall_eth, eth_sub = compute_eddi(y_true, y_pred, df_results['ethnicity_category'].values)
    overall_ins, ins_sub = compute_eddi(y_true, y_pred, df_results['insurance_category'].values)
    
    print("Age Buckets EDDI (per subgroup):")
    for bucket in age_order:
        print(f"  {bucket}: {age_sub.get(bucket, np.nan):.4f}")
    print("Overall Age EDDI:", overall_age)
    
    print("Ethnicity Groups EDDI (per subgroup):")
    for group in ethnicity_order:
        print(f"  {group}: {eth_sub.get(group, np.nan):.4f}")
    print("Overall Ethnicity EDDI:", overall_eth)
    
    print("Insurance Groups EDDI (per subgroup):")
    for group in insurance_order:
        print(f"  {group}: {ins_sub.get(group, np.nan):.4f}")
    print("Overall Insurance EDDI:", overall_ins)
    
    total_eddi = np.sqrt((overall_age**2 + overall_eth**2 + overall_ins**2)) / 3
    print(f"Final Overall {outcome_name} EDDI: {total_eddi:.4f}")
    return total_eddi, {"age": age_sub, "ethnicity": eth_sub, "insurance": ins_sub}


def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("Data shape:", df.shape)
    
    note_columns = [col for col in df.columns if col.startswith("note_")]
    print("Note columns found:", note_columns)
    
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
        df_filtered, note_columns, tokenizer, bioclinical_bert_ft, device, aggregation="mean", max_length=512
    )
    print("Aggregated text embeddings shape:", aggregated_embeddings_np.shape)
    
    # Use unique patients for labels and demographics.
    df_unique = df_filtered.drop_duplicates(subset="subject_id")
    
    # Multi-Label Iterative Stratification Splitting
    X = df_unique[['subject_id']].values  
    # Create the multi-label target matrix with the three outcomes.
    y = df_unique[['short_term_mortality', 'los_binary', 'mechanical_ventilation']].values 
    
    # First split: 80% train+val and 20% test.
    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)
    
    # Create dataframes based on subject_ids (converted to strings for consistency)
    subject_ids_train_val = set(X_train_val.flatten().astype(str))
    subject_ids_test = set(X_test.flatten().astype(str))
    
    df_train_val = df_unique[df_unique['subject_id'].astype(str).isin(subject_ids_train_val)]
    df_test = df_unique[df_unique['subject_id'].astype(str).isin(subject_ids_test)]
    
    # Second split: split train_val into train and validation.
    val_fraction = 0.05 / 0.8  # ~0.0625
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size=val_fraction)
    
    subject_ids_train = set(X_train.flatten().astype(str))
    subject_ids_val = set(X_val.flatten().astype(str))
    
    df_train = df_train_val[df_train_val['subject_id'].astype(str).isin(subject_ids_train)]
    df_val = df_train_val[df_train_val['subject_id'].astype(str).isin(subject_ids_val)]
    
    print("Train set size:", len(df_train))
    print("Validation set size:", len(df_val))
    print("Test set size:", len(df_test))
    
    # Convert subject_id in df_test to string to avoid merge errors.
    df_test['subject_id'] = df_test['subject_id'].astype(str)
    
    # Create a mapping from subject_id to aggregated embedding.
    patient_embedding_dict = {str(sid): emb for sid, emb in zip(patient_ids, aggregated_embeddings_np)}
    
    def create_dataset(df_subset):
        embeddings_list = []
        mortality_labels = []
        los_labels = []
        mech_labels = []
        for _, row in df_subset.iterrows():
            sid = str(row["subject_id"])
            if sid in patient_embedding_dict:
                embeddings_list.append(patient_embedding_dict[sid])
                mortality_labels.append(row["short_term_mortality"])
                los_labels.append(row["los_binary"])
                mech_labels.append(row["mechanical_ventilation"])
        embeddings_array = np.array(embeddings_list)
        return UnstructuredDataset(embeddings_array, mortality_labels, los_labels, mech_labels)
    
    train_dataset = create_dataset(df_train)
    val_dataset = create_dataset(df_val)
    test_dataset = create_dataset(df_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    classifier = UnstructuredClassifier(input_size=768, hidden_size=256).to(device)
    
    # Compute class weights for each outcome using the training set.
    class_weights_mort = compute_class_weights(df_train, "short_term_mortality")
    class_weights_los = compute_class_weights(df_train, "los_binary")
    class_weights_mech = compute_class_weights(df_train, "mechanical_ventilation")
    pos_weight_mort = torch.tensor(class_weights_mort.iloc[1], dtype=torch.float, device=device)
    pos_weight_los = torch.tensor(class_weights_los.iloc[1], dtype=torch.float, device=device)
    pos_weight_mech = torch.tensor(class_weights_mech.iloc[1], dtype=torch.float, device=device)
    
    # Use Focal Loss for each task.
    criterion_mort = FocalLoss(gamma=2, pos_weight=pos_weight_mort, reduction='mean')
    criterion_los = FocalLoss(gamma=2, pos_weight=pos_weight_los, reduction='mean')
    criterion_mech = FocalLoss(gamma=2, pos_weight=pos_weight_mech, reduction='mean')
    
    optimizer = AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training settings.
    max_epochs = 50
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = "best_classifier_model.pt"
    
    print("\nStarting training...")
    for epoch in range(max_epochs):
        train_loss = train_model(classifier, train_loader, optimizer, device, criterion_mort, criterion_los, criterion_mech)
        val_loss = evaluate_loss(classifier, val_loader, device, criterion_mort, criterion_los, criterion_mech)
        print(f"\n[Epoch {epoch+1}/{max_epochs}] Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model.
            torch.save(classifier.state_dict(), best_model_path)
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
        
        # Early stopping.
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break
    
    print("Training complete.")
    
    classifier.load_state_dict(torch.load(best_model_path))
    

    print("\nEvaluating on the test set:")
    test_metrics = evaluate_model(classifier, test_loader, device, threshold=0.5)
    for outcome, m in test_metrics.items():
        print(f"\nOutcome: {outcome.capitalize()}")
        print(f"  AUROC     : {m['aucroc']:.4f}")
        print(f"  AUPRC     : {m['auprc']:.4f}")
        print(f"  F1-Score  : {m['f1']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  TPR       : {m['tpr']:.4f}")
        print(f"  FPR       : {m['fpr']:.4f}")
    
    
    demo_cols = ['subject_id', 'age', 'ethnicity_category', 'insurance_category']
    df_demo = df_test[demo_cols].copy()
    df_demo['subject_id'] = df_demo['subject_id'].astype(str)
    
    probs = get_patient_probabilities(classifier, test_loader, device)
    
    df_probs = pd.DataFrame({
        'subject_id': df_demo['subject_id'].values,
        'mortality_prob': probs[:, 0],
        'los_prob': probs[:, 1],
        'mech_prob': probs[:, 2]
    })
    
    df_results = pd.merge(df_demo, df_probs, on='subject_id', how='inner')
    
    df_results["ethnicity_category"] = df_results["ethnicity_category"].str.lower().fillna("others")
    df_results["ethnicity_category"] = df_results["ethnicity_category"].apply(
        lambda x: x if x in ["white", "black", "hispanic", "asian"] else "others"
    )
    
    df_results["insurance_category"] = df_results["insurance_category"].str.lower().fillna("others")
    df_results["insurance_category"] = df_results["insurance_category"].apply(
        lambda x: x if x in ["government", "medicare", "medicaid", "private", "self pay"] else "others"
    )
    
    # Fixed subgroup orders.
    age_order = ["15-29", "30-49", "50-69", "70-89", "other"]
    ethnicity_order = ["white", "black", "hispanic", "asian", "others"]
    insurance_order = ["government", "medicare", "medicaid", "private", "self pay", "others"]
    
    # Add ground truth labels.
    df_results['mortality_pred'] = (df_results['mortality_prob'] >= 0.5).astype(int)
    df_results['los_pred'] = (df_results['los_prob'] >= 0.5).astype(int)
    df_results['mech_pred'] = (df_results['mech_prob'] >= 0.5).astype(int)
    
    df_results = pd.merge(df_results,
                          df_test[['subject_id', 'short_term_mortality', 'los_binary', 'mechanical_ventilation']],
                          on='subject_id', how='inner')
    df_results.rename(columns={
        'short_term_mortality': 'mortality_true',
        'los_binary': 'los_true',
        'mechanical_ventilation': 'mech_true'
    }, inplace=True)
    
    # Create age buckets for fairness evaluation.
    df_results['age_bucket'] = df_results['age'].apply(assign_age_bucket)
    
    # For each outcome, compute and print detailed EDDI values.
    y_true_mort = df_results["mortality_true"].values.astype(int)
    y_pred_mort = df_results["mortality_pred"].values.astype(int)
    y_true_los = df_results["los_true"].values.astype(int)
    y_pred_los = df_results["los_pred"].values.astype(int)
    y_true_mech = df_results["mech_true"].values.astype(int)
    y_pred_mech = df_results["mech_pred"].values.astype(int)
    
    print("\nCalculating detailed fairness metrics (EDDI) on the test set:")
    eddi_mort, details_mort = print_detailed_eddi(y_true_mort, y_pred_mort, "Mortality", df_results, age_order, ethnicity_order, insurance_order)
    eddi_los, details_los = print_detailed_eddi(y_true_los, y_pred_los, "LOS", df_results, age_order, ethnicity_order, insurance_order)
    eddi_mech, details_mech = print_detailed_eddi(y_true_mech, y_pred_mech, "Mechanical Ventilation", df_results, age_order, ethnicity_order, insurance_order)
    
    eddi_all = {
        "mortality": {"overall_EDDI": eddi_mort, "subgroup_EDDI": details_mort},
        "los": {"overall_EDDI": eddi_los, "subgroup_EDDI": details_los},
        "mechanical_ventilation": {"overall_EDDI": eddi_mech, "subgroup_EDDI": details_mech}
    }
    print("\nAll EDDI Results on the test set:")
    for outcome, stats in eddi_all.items():
        print(f"{outcome.capitalize()} Overall EDDI: {stats['overall_EDDI']:.4f}")
    


if __name__ == "__main__":
    train_pipeline()
