import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

######################################
# 1. Read the Final Dataset
######################################
# Load a subset of the dataset for this example.
df = pd.read_csv("final_unstructured.csv", low_memory=False)
# (Make sure your CSV contains a column named "subject_id" for unique patient IDs.)
note_chunk_cols = [col for col in df.columns if col.startswith("note_chunk_")]
print("Note chunk columns found:", note_chunk_cols)
print("Dataset shape:", df.shape)

######################################
# Function to compute class weights using the Inverse of Number of Samples (INS)
######################################
def compute_class_weights(df, label_column):
    class_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

# Compute class weights for each outcome.
class_weights_mortality = compute_class_weights(df, 'short_term_mortality')
class_weights_readmission = compute_class_weights(df, 'readmission_within_30_days')

######################################
# 2. Create a PyTorch Dataset
######################################
class PatientDataset(Dataset):
    def __init__(self, df, note_chunk_cols):
        """
        Expects df to have one row per patient with note_chunk columns,
        a "subject_id" column for patient identity, and two target columns:
        'short_term_mortality' and 'readmission_within_30_days'.
        """
        self.df = df.reset_index(drop=True)
        self.note_chunk_cols = note_chunk_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # For each note chunk column, get the text if available.
        chunks = [str(row[col]) for col in self.note_chunk_cols if pd.notnull(row[col])]
        # Only keep non-empty note chunks.
        patient_chunks = [chunk for chunk in chunks if chunk.strip() != ""]
        # Join all note chunks into one string, using a separator.
        # You may choose a different separator (or even a newline) as appropriate.
        joined_text = " [SEP] ".join(patient_chunks) if patient_chunks else ""
        # The targets are expected to be 0 or 1.
        labels = torch.tensor([row['short_term_mortality'], row['readmission_within_30_days']], dtype=torch.float)
        return {"text": joined_text, "labels": labels, "subject_id": row['subject_id']}

def custom_collate(batch):
    """
    Gathers the joined text (one string per patient) and labels.
    """
    texts = [sample["text"] for sample in batch]
    labels = torch.stack([sample["labels"] for sample in batch])
    return {"texts": texts, "labels": labels}

######################################
# 3. Define the Model Using BioClinicalBERT
######################################
class BioClinicalBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BioClinicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # e.g., 768
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Process the batch of tokenized texts through BERT.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get the CLS token embedding for each input.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]
        logits = self.classifier(cls_embeddings)             # shape: [batch_size, num_labels]
        return logits

######################################
# 4. Training and Evaluation Functions
######################################
def train_model(model, tokenizer, train_loader, val_loader, device, num_epochs, lr,
                class_weights_tensor_mortality, class_weights_tensor_readmission):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    
    # Create loss functions for each task.
    loss_fn_mortality = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_mortality[1])
    loss_fn_readmission = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor_readmission[1])
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            texts = batch["texts"]  # List of joined texts, one per patient.
            labels = batch["labels"].to(device)  # shape: [batch_size, 2]
            
            # Tokenize the batch of texts.
            encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            optimizer.zero_grad()
            logits = model(encoding["input_ids"], encoding["attention_mask"])
            
            loss_mortality = loss_fn_mortality(logits[:, 0], labels[:, 0])
            loss_readmission = loss_fn_readmission(logits[:, 1], labels[:, 1])
            loss = loss_mortality + loss_readmission
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase.
        model.eval()
        all_labels = []
        all_logits = []
        with torch.no_grad():
            for batch in val_loader:
                texts = batch["texts"]
                labels = batch["labels"].to(device)
                encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                encoding = {k: v.to(device) for k, v in encoding.items()}
                logits = model(encoding["input_ids"], encoding["attention_mask"])
                all_labels.append(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        all_labels = np.concatenate(all_labels, axis=0)  # shape: [n_samples, 2]
        all_logits = np.concatenate(all_logits, axis=0)
        probs = 1 / (1 + np.exp(-all_logits))
        preds_bin = (probs >= 0.5).astype(int)
        
        metrics = {}
        tasks = ["short_term_mortality", "readmission_within_30_days"]
        for i, task in enumerate(tasks):
            try:
                auroc = roc_auc_score(all_labels[:, i], probs[:, i])
            except Exception:
                auroc = np.nan
            try:
                auprc = average_precision_score(all_labels[:, i], probs[:, i])
            except Exception:
                auprc = np.nan
            f1 = f1_score(all_labels[:, i], preds_bin[:, i])
            recall = recall_score(all_labels[:, i], preds_bin[:, i])
            precision = precision_score(all_labels[:, i], preds_bin[:, i])
            metrics[task] = {"auroc": auroc, "auprc": auprc, "f1": f1, "recall": recall, "precision": precision}
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print("Validation Metrics:")
        for task, m in metrics.items():
            print(f" {task}: AUROC: {m['auroc']:.4f}, AUPRC: {m['auprc']:.4f}, F1: {m['f1']:.4f}, "
                  f"Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")

def evaluate_model(model, tokenizer, data_loader, device):
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in data_loader:
            texts = batch["texts"]
            labels = batch["labels"].to(device)
            encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            logits = model(encoding["input_ids"], encoding["attention_mask"])
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    probs = 1 / (1 + np.exp(-all_logits))
    preds_bin = (probs >= 0.5).astype(int)
    metrics = {}
    tasks = ["short_term_mortality", "readmission_within_30_days"]
    for i, task in enumerate(tasks):
        try:
            auroc = roc_auc_score(all_labels[:, i], probs[:, i])
        except Exception:
            auroc = np.nan
        try:
            auprc = average_precision_score(all_labels[:, i], probs[:, i])
        except Exception:
            auprc = np.nan
        f1 = f1_score(all_labels[:, i], preds_bin[:, i])
        recall = recall_score(all_labels[:, i], preds_bin[:, i])
        precision = precision_score(all_labels[:, i], preds_bin[:, i])
        metrics[task] = {"auroc": auroc, "auprc": auprc, "f1": f1, "recall": recall, "precision": precision}
    return metrics

######################################
# 5. Helper Function: Get Aggregated Patient Embeddings
######################################
def get_patient_embeddings(model, tokenizer, df, note_chunk_cols, device):
    """
    For each unique patient (by subject_id), extracts all non-null notes from the given note columns,
    joins them into one string, tokenizes that text, computes the CLS embedding using the provided model,
    and returns the embedding. Returns an array of shape (num_patients, hidden_size) and a list of patient IDs.
    If the model is wrapped in DataParallel, we access the underlying module.
    """
    model_to_use = model.module if hasattr(model, "module") else model
    model_to_use.eval()
    
    patient_ids = df['subject_id'].unique()
    embeddings = []
    with torch.no_grad():
        for pid in patient_ids:
            patient_df = df[df['subject_id'] == pid]
            chunks = []
            for idx, row in patient_df.iterrows():
                for col in note_chunk_cols:
                    if pd.notnull(row[col]) and str(row[col]).strip() != "":
                        chunks.append(str(row[col]))
            joined_text = " [SEP] ".join(chunks) if chunks else ""
            encoding = tokenizer(joined_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model_to_use.bert(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
            cls_embedding = outputs.last_hidden_state[:, 0, :].mean(dim=0)
            embeddings.append(cls_embedding.cpu().numpy())
    embeddings = np.stack(embeddings, axis=0)
    return embeddings, patient_ids

######################################
# 6. Main Script: Data Splitting, Training, and Testing
######################################
if __name__ == "__main__":
    # Shuffle and split the data into train, validation, and test sets.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(0.7 * n)]
    val_df = df.iloc[int(0.7 * n):int(0.85 * n)]
    test_df = df.iloc[int(0.85 * n):]
    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    
    # Initialize the tokenizer.
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets and dataloaders.
    train_dataset = PatientDataset(train_df, note_chunk_cols)
    val_dataset = PatientDataset(val_df, note_chunk_cols)
    test_dataset = PatientDataset(test_df, note_chunk_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)
    
    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the classifier model.
    model = BioClinicalBERTClassifier(model_name, num_labels=2)
    
    # Optionally wrap the model for multi-GPU training.
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Convert class weights to tensors and move them to the device.
    class_weights_tensor_mortality = torch.tensor(class_weights_mortality.values, dtype=torch.float).to(device)
    class_weights_tensor_readmission = torch.tensor(class_weights_readmission.values, dtype=torch.float).to(device)
    
    # Train the model.
    num_epochs = 3
    train_model(model, tokenizer, train_loader, val_loader, device, num_epochs=num_epochs, lr=2e-5,
                class_weights_tensor_mortality=class_weights_tensor_mortality,
                class_weights_tensor_readmission=class_weights_tensor_readmission)
    
    # Evaluate on the test set.
    test_metrics = evaluate_model(model, tokenizer, test_loader, device)
    print("Test Set Metrics:")
    for task, m in test_metrics.items():
        print(f" {task}: AUROC: {m['auroc']:.4f}, AUPRC: {m['auprc']:.4f}, F1: {m['f1']:.4f}, "
              f"Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}")
    
    # (Optional) Get aggregated patient embeddings.
    aggregated_embeddings, patient_ids = get_patient_embeddings(model, tokenizer, df, note_chunk_cols, device)
    print("Aggregated embeddings shape:", aggregated_embeddings.shape)
