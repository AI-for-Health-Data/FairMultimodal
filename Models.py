import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertModel, BertConfig, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight

# Define the BioClinicalBERT Fine-tuned Model
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, BioBert, BioBertConfig, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = BioBert
        self.device = device
        self.config = BioBertConfig  # Expose the config

    def forward(self, input_ids, attention_mask):
        # Extract CLS embedding from the last hidden state
        output = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

# Define the BEHRT Model
class BEHRTModel(nn.Module):
    def __init__(self, hidden_size=768):
        super(BEHRTModel, self).__init__()
        self.config = BertConfig(
            vocab_size=30522,
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(self.config)

    def forward(self, input_ids, attention_mask):
        # Extract CLS embedding from the last hidden state
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

# Define the Multimodal Transformer model
class MultimodalTransformer(nn.Module):
    def __init__(self, BioBert, BEHRT, device):
        super(MultimodalTransformer, self).__init__()
        self.BioBert = BioBert
        self.BEHRT = BEHRT
        self.device = device

        self.ts_embed_size = BEHRT.config.hidden_size
        self.text_embed_size = BioBert.config.hidden_size

        # Projection layers for embeddings
        self.ts_projector = nn.Sequential(
            nn.Linear(self.ts_embed_size, 256),
            nn.ReLU()
        )
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_embed_size, 256),
            nn.ReLU()
        )

        # Combine both embeddings
        self.combined_embed_size = 512
        self.classifier = nn.Linear(self.combined_embed_size, 1)

    def forward(self, ts_inputs, ts_attention_mask, text_inputs, text_attention_mask):
        # BEHRT
        ts_cls_embedding = self.BEHRT(input_ids=ts_inputs, attention_mask=ts_attention_mask)
        print(f"BEHRT CLS embedding shape: {ts_cls_embedding.shape}")
        print(f"BEHRT CLS embedding sample: {ts_cls_embedding[0].detach().cpu().numpy()[:5]}")  # Print first 5 values

        ts_projected = self.ts_projector(ts_cls_embedding)
        
        # BioBERT
        text_cls_embedding = self.BioBert(input_ids=text_inputs, attention_mask=text_attention_mask)
        print(f"BioBERT CLS embedding shape: {text_cls_embedding.shape}")
        print(f"BioBERT CLS embedding sample: {text_cls_embedding[0].detach().cpu().numpy()[:5]}")  # Print first 5 values

        text_projected = self.text_projector(text_cls_embedding)

        # Combine embeddings
        combined_embeddings = torch.cat((ts_projected, text_projected), dim=1)
        logits = self.classifier(combined_embeddings).squeeze(-1)
        return logits

# Function to calculate evaluation metrics
def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, attention_mask, labels = batch
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(ts_inputs=inputs, ts_attention_mask=attention_mask, text_inputs=inputs, text_attention_mask=attention_mask)
            probs = torch.sigmoid(logits)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_probs > 0.5, average='binary')
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }

# Updated training pipeline
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load structured data
    structured_data = pd.read_csv('structured_first_admissions.csv').sample(100)
    print("Columns in structured_data:", structured_data.columns)

    # Preprocess structured data
    structured_data['age'] = structured_data['age'].astype(int)
    structured_data['ADMISSION_LOCATION'] = structured_data['ADMISSION_LOCATION'].astype('category').cat.codes
    structured_data['DISCHARGE_LOCATION'] = structured_data['DISCHARGE_LOCATION'].astype('category').cat.codes
    structured_data['GENDER'] = structured_data['GENDER'].astype('category').cat.codes
    structured_data['ETHNICITY'] = structured_data['ETHNICITY'].astype('category').cat.codes
    structured_data['INSURANCE'] = structured_data['INSURANCE'].astype('category').cat.codes

    structured_inputs = torch.tensor(structured_data[['age']].values, dtype=torch.long)
    structured_attention_mask = torch.ones_like(structured_inputs)
    labels = torch.tensor(structured_data['short_term_mortality'].values, dtype=torch.float32)

    structured_dataset = TensorDataset(structured_inputs, structured_attention_mask, labels)
    structured_dataloader = DataLoader(structured_dataset, batch_size=32, shuffle=True)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(structured_data['short_term_mortality'].values),
                                         y=structured_data['short_term_mortality'].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize BEHRT
    behrt_model = BEHRTModel()

    # Load unstructured data
    unstructured_data = pd.read_csv('first_notes_unstructured.csv').sample(100)
    print("Columns in unstructured_data:", unstructured_data.columns)

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenized_notes = tokenizer(
        list(unstructured_data['note']),
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    note_dataset = TensorDataset(
        tokenized_notes['input_ids'],
        tokenized_notes['attention_mask'],
        torch.tensor(unstructured_data['short_term_mortality'].values, dtype=torch.float32)
    )
    note_dataloader = DataLoader(note_dataset, batch_size=32, shuffle=True)

    # Initialize BioClinicalBERT
    biobert_model = BioClinicalBERT_FT(BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'), BertConfig(), device=device)

    # Initialize Multimodal Model
    multimodal_model = MultimodalTransformer(BioBert=biobert_model, BEHRT=behrt_model, device=device).to(device)

    # Training setup
    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        multimodal_model.train()
        for (structured_batch, unstructured_batch) in zip(structured_dataloader, note_dataloader):
            structured_inputs, structured_attention_mask, labels = structured_batch
            unstructured_inputs, unstructured_attention_mask, text_labels = unstructured_batch

            structured_inputs = structured_inputs.to(device)
            structured_attention_mask = structured_attention_mask.to(device)
            unstructured_inputs = unstructured_inputs.to(device)
            unstructured_attention_mask = unstructured_attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = multimodal_model(
                ts_inputs=structured_inputs,
                ts_attention_mask=structured_attention_mask,
                text_inputs=unstructured_inputs,
                text_attention_mask=unstructured_attention_mask
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # Evaluation after each epoch
        metrics = evaluate_model(multimodal_model, note_dataloader, device)
        print(f"Epoch {epoch + 1} Metrics: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}, AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")

if __name__ == "__main__":
    train_pipeline()
