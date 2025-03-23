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
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.special import expit  # for logistic sigmoid

# Utility function to compute class weights using the Inverse Number of Samples (INS) method.
def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

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
    patient_ids = df["subject_id"].drop_duplicates().values
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
        # Output 3 logits: mortality, length-of-stay (LOS), and mechanical ventilation.
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

# Evaluation function for metrics
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
    # Use logistic sigmoid to convert logits to probabilities
    probs = expit(all_logits)
    preds = (probs >= threshold).astype(int)
    
    metrics = {}
    tasks = ["mortality", "los", "mech"]
    for i, task in enumerate(tasks):
        try:
            aucroc = roc_auc_score(all_labels[:, i], probs[:, i])
        except Exception:
            aucroc = float('nan')
        try:
            auprc = average_precision_score(all_labels[:, i], probs[:, i])
        except Exception:
            auprc = float('nan')
        f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        recall_metric = recall_score(all_labels[:, i], preds[:, i], zero_division=0)
        precision = precision_score(all_labels[:, i], preds[:, i], zero_division=0)
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

# Evaluation function to compute loss on a given dataset (e.g., validation/test)
def compute_loss(model, dataloader, device, criterion_mort, criterion_los, criterion_mech):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels_mort, labels_los, labels_mech = [x.to(device) for x in batch]
            logits = model(embeddings)
            loss_mort = criterion_mort(logits[:, 0].unsqueeze(1), labels_mort)
            loss_los = criterion_los(logits[:, 1].unsqueeze(1), labels_los)
            loss_mech = criterion_mech(logits[:, 2].unsqueeze(1), labels_mech)
            loss = loss_mort + loss_los + loss_mech
            total_loss += loss.item()
    return total_loss / len(dataloader)

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

# Main Training and Evaluation Pipeline
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Read the unstructured data CSV.
    df = pd.read_csv("final_unstructured_common.csv", low_memory=False)
    print("Original data shape:", df.shape)
    
    # Identify note columns (columns starting with 'note_').
    note_columns = [col for col in df.columns if col.startswith("note_")]
    print("Note columns found:", note_columns)
    
    # Filter rows that contain at least one valid note.
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
    df_unique = df_unique.set_index("subject_id").loc[patient_ids].reset_index()
    
    # Create a stratification label by concatenating outcome columns.
    df_unique['strat_labels'] = (
        df_unique['short_term_mortality'].astype(str) + "_" +
        df_unique['los_binary'].astype(str) + "_" +
        df_unique['mechanical_ventilation'].astype(str)
    )
    
    # Outcome labels.
    mortality_labels = df_unique["short_term_mortality"].values
    los_labels = df_unique["los_binary"].values
    mech_labels = df_unique["mechanical_ventilation"].values

    # Build dataset.
    full_dataset = UnstructuredDataset(aggregated_embeddings_np, mortality_labels, los_labels, mech_labels)
    
    # Get stratification labels.
    strat_labels = df_unique['strat_labels'].values

    # Create indices for the full dataset.
    indices = np.arange(len(full_dataset))
    
    # Stratified split: 80% train, 20% test.
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=strat_labels
    )
    
    # From the training data, set aside 5% for validation (again, stratified).
    train_strat_labels = strat_labels[train_indices]
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.05,
        random_state=42,
        shuffle=True,
        stratify=train_strat_labels
    )
    
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=16, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=16, shuffle=False)
    
    # Instantiate classifier.
    classifier = UnstructuredClassifier(input_size=768, hidden_size=256).to(device)
    
    # Compute class weights for each outcome using INS method.
    class_weights_mort = compute_class_weights(df_unique, "short_term_mortality")
    class_weights_los = compute_class_weights(df_unique, "los_binary")
    class_weights_mech = compute_class_weights(df_unique, "mechanical_ventilation")
    pos_weight_mort = torch.tensor(class_weights_mort.iloc[1], dtype=torch.float, device=device)
    pos_weight_los = torch.tensor(class_weights_los.iloc[1], dtype=torch.float, device=device)
    pos_weight_mech = torch.tensor(class_weights_mech.iloc[1], dtype=torch.float, device=device)
    
    # Use weighted BCEWithLogitsLoss for each task.
    criterion_mort = nn.BCEWithLogitsLoss(pos_weight=pos_weight_mort)
    criterion_los = nn.BCEWithLogitsLoss(pos_weight=pos_weight_los)
    criterion_mech = nn.BCEWithLogitsLoss(pos_weight=pos_weight_mech)
    
    # Define optimizer with weight decay for regularization.
    optimizer = AdamW(classifier.parameters(), lr=2e-5, weight_decay=1e-2)
    # Learning rate scheduler that reduces LR on plateau (based on validation loss).
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    num_epochs = 50  # maximum epochs
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training loop with early stopping.
    for epoch in range(num_epochs):
        train_loss = train_model(classifier, train_loader, optimizer, device, criterion_mort, criterion_los, criterion_mech)
        val_loss = compute_loss(classifier, val_loader, device, criterion_mort, criterion_los, criterion_mech)
        scheduler.step(val_loss)
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping check.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model.
            torch.save(classifier.state_dict(), "best_model.pt")
            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("Training complete.")
    
    # Load the best model for final evaluation.
    classifier.load_state_dict(torch.load("best_model.pt"))
    
    # Evaluate on the test set.
    test_metrics = evaluate_model(classifier, test_loader, device, threshold=0.5)
    print("\nTest Evaluation Metrics:")
    for outcome, m in test_metrics.items():
        print(f"\nOutcome: {outcome.capitalize()}")
        print(f"  AUROC     : {m['aucroc']:.4f}")
        print(f"  AUPRC     : {m['auprc']:.4f}")
        print(f"  F1-Score  : {m['f1']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  TPR       : {m['tpr']:.4f}")
        print(f"  FPR       : {m['fpr']:.4f}")
    
    # Inference and Fairness Evaluation for All Three Tasks.
    probs = get_patient_probabilities(classifier, test_loader, device)
    
    # Build results dataframe by merging predictions with demographics.
    demo_cols = ['subject_id', 'age', 'ethnicity_category', 'insurance_category']
    df_demo = df_unique[demo_cols].copy()
    
    # Use patient_ids from df_unique (which is in the same order as the dataset) for the test set.
    test_patient_ids = df_unique['subject_id'].values[test_indices]
    
    # Debug: Print lengths to verify alignment.
    print("Length of test_patient_ids:", len(test_patient_ids))
    print("Number of probability predictions:", probs.shape[0])
    
    df_probs = pd.DataFrame({
        'subject_id': test_patient_ids,
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
                          df_unique[['subject_id', 'short_term_mortality', 'los_binary', 'mechanical_ventilation']],
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
    
    print("\nCalculating detailed fairness metrics (EDDI):")
    eddi_mort, details_mort = print_detailed_eddi(y_true_mort, y_pred_mort, "Mortality", df_results, age_order, ethnicity_order, insurance_order)
    eddi_los, details_los = print_detailed_eddi(y_true_los, y_pred_los, "LOS", df_results, age_order, ethnicity_order, insurance_order)
    eddi_mech, details_mech = print_detailed_eddi(y_true_mech, y_pred_mech, "Mechanical Ventilation", df_results, age_order, ethnicity_order, insurance_order)
    
    eddi_all = {
        "mortality": {"overall_EDDI": eddi_mort, "subgroup_EDDI": details_mort},
        "los": {"overall_EDDI": eddi_los, "subgroup_EDDI": details_los},
        "mechanical_ventilation": {"overall_EDDI": eddi_mech, "subgroup_EDDI": details_mech}
    }
    print("\nAll EDDI Results:")
    for outcome, stats in eddi_all.items():
        print(f"{outcome.capitalize()} Overall EDDI: {stats['overall_EDDI']:.4f}")
    
    # Optionally, save the results to CSV.
    # df_results.to_csv("unstructured_predictions_and_fairness.csv", index=False)
    # print("Results saved to 'unstructured_predictions_and_fairness.csv'.")

if __name__ == "__main__":
    train_pipeline()
