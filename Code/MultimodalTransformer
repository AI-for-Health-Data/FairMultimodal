import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class BioClinicalBERT_FT(nn.Module):
    def __init__(self, BioBert):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = BioBert

    def forward(self, input_ids, attention_mask):
        output = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding

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
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding

class MultimodalTransformer(nn.Module):
    def __init__(self, BioBert, BEHRT, device):
        super(MultimodalTransformer, self).__init__()
        self.BioBert = BioBert
        self.BEHRT = BEHRT
        self.device = device

        self.ts_embed_size = BEHRT.config.hidden_size
        self.text_embed_size = BioBert.BioBert.config.hidden_size

        self.ts_projector = nn.Sequential(
            nn.Linear(self.ts_embed_size, 256),
            nn.ReLU()
        )
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_embed_size, 256),
            nn.ReLU()
        )

        self.combined_embed_size = 512

        # Mortality-specific layers
        self.mortality_fc1 = nn.Linear(self.combined_embed_size, 128)
        self.mortality_fc2 = nn.Linear(128, 64)
        self.mortality_classifier = nn.Linear(64, 1)

        # Readmission-specific layers
        self.readmission_fc1 = nn.Linear(self.combined_embed_size, 128)
        self.readmission_fc2 = nn.Linear(128, 64)
        self.readmission_classifier = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, ts_inputs, ts_attention_mask, text_embeddings):
        ts_cls_embedding = self.BEHRT(input_ids=ts_inputs, attention_mask=ts_attention_mask)
        ts_projected = self.ts_projector(ts_cls_embedding)

        text_projected = self.text_projector(text_embeddings)

        combined_embeddings = torch.cat((ts_projected, text_projected), dim=1)

        # Mortality prediction
        mortality_hidden = self.dropout(torch.relu(self.mortality_fc1(combined_embeddings)))
        mortality_hidden = self.dropout(torch.relu(self.mortality_fc2(mortality_hidden)))
        mortality_logits = self.mortality_classifier(mortality_hidden).squeeze(-1)

        # Readmission prediction
        readmission_hidden = self.dropout(torch.relu(self.readmission_fc1(combined_embeddings)))
        readmission_hidden = self.dropout(torch.relu(self.readmission_fc2(readmission_hidden)))
        readmission_logits = self.readmission_classifier(readmission_hidden).squeeze(-1)

        return mortality_logits, readmission_logits

def process_note_chunks(notes, tokenizer, model, device):
    cls_embeddings = []
    for note in notes:
        if note is not None and isinstance(note, str):
            tokenized = tokenizer(note, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            with torch.no_grad():
                cls_embedding = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings.append(cls_embedding)

    return torch.cat(cls_embeddings, dim=0) if len(cls_embeddings) > 0 else torch.zeros((1, model.BioBert.config.hidden_size), device=device)

def apply_bioclinicalbert_on_chunks(data, note_columns, tokenizer, model, device):
    model.eval()
    embeddings = []
    for _, row in data.iterrows():
        note_chunks = [row[col] for col in note_columns if pd.notnull(row[col])]
        patient_embedding = process_note_chunks(note_chunks, tokenizer, model, device)
        embeddings.append(patient_embedding.cpu().numpy())
    return np.vstack(embeddings)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = {'mortality': [], 'readmission': []}
    all_preds = {'mortality': [], 'readmission': []}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ts_inputs, ts_attention_mask, text_embeddings, labels_mortality, labels_readmission = batch
            ts_inputs = ts_inputs.to(device)
            ts_attention_mask = ts_attention_mask.to(device)
            text_embeddings = text_embeddings.to(device)
            labels_mortality = labels_mortality.to(device)
            labels_readmission = labels_readmission.to(device)

            mortality_logits, readmission_logits = model(
                ts_inputs=ts_inputs,
                ts_attention_mask=ts_attention_mask,
                text_embeddings=text_embeddings
            )

            all_labels['mortality'].extend(labels_mortality.cpu().numpy())
            all_labels['readmission'].extend(labels_readmission.cpu().numpy())

            all_preds['mortality'].extend(torch.sigmoid(mortality_logits).cpu().numpy())
            all_preds['readmission'].extend(torch.sigmoid(readmission_logits).cpu().numpy())

    metrics = {}
    for key in ['mortality', 'readmission']:
        labels = np.array(all_labels[key])
        preds = np.array(all_preds[key])
        metrics[key] = {
            'AUROC': roc_auc_score(labels, preds),
            'AUPRC': average_precision_score(labels, preds),
            'F1': f1_score(labels, preds > 0.5),
            'Precision': precision_score(labels, preds > 0.5),
            'Recall': recall_score(labels, preds > 0.5)
        }
    return metrics

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    structured_data = pd.read_csv('structured_first_icu_stays.csv')
    structured_inputs = torch.tensor(structured_data[['age']].values, dtype=torch.long)
    structured_attention_mask = torch.ones_like(structured_inputs)
    structured_labels_mortality = torch.tensor(structured_data['short_term_mortality'].values, dtype=torch.float32)
    structured_labels_readmission = torch.tensor(structured_data['readmission_within_30_days'].values, dtype=torch.float32)

    structured_dataset = TensorDataset(structured_inputs, structured_attention_mask, structured_labels_mortality, structured_labels_readmission)
    structured_dataloader = DataLoader(structured_dataset, batch_size=32, shuffle=True)

    unstructured_data = pd.read_csv('processed_icu_notes.csv')
    note_columns = [col for col in unstructured_data.columns if col.startswith('note_')]

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    biobert_model = BioClinicalBERT_FT(BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")).to(device)

    text_embeddings = apply_bioclinicalbert_on_chunks(unstructured_data, note_columns, tokenizer, biobert_model, device)
    text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(device)

    unstructured_labels_mortality = torch.tensor(unstructured_data['short_term_mortality'].values, dtype=torch.float32).to(device)
    unstructured_labels_readmission = torch.tensor(unstructured_data['readmitted_within_30_days'].values, dtype=torch.float32).to(device)

    unstructured_dataset = TensorDataset(text_embeddings, unstructured_labels_mortality, unstructured_labels_readmission)
    unstructured_dataloader = DataLoader(unstructured_dataset, batch_size=32, shuffle=True)

    behrt_model = BEHRTModel()
    multimodal_model = MultimodalTransformer(BioBert=biobert_model, BEHRT=behrt_model, device=device).to(device)

    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        multimodal_model.train()
        for (structured_batch, unstructured_batch) in zip(structured_dataloader, unstructured_dataloader):
            ts_inputs, ts_attention_mask, labels_mortality, labels_readmission = structured_batch
            text_embeddings, text_labels_mortality, text_labels_readmission = unstructured_batch

            ts_inputs = ts_inputs.to(device)
            ts_attention_mask = ts_attention_mask.to(device)
            text_embeddings = text_embeddings.to(device)
            labels_mortality = labels_mortality.to(device)
            labels_readmission = labels_readmission.to(device)

            optimizer.zero_grad()
            mortality_logits, readmission_logits = multimodal_model(
                ts_inputs=ts_inputs,
                ts_attention_mask=ts_attention_mask,
                text_embeddings=text_embeddings
            )

            loss_mortality = criterion(mortality_logits, labels_mortality)
            loss_readmission = criterion(readmission_logits, labels_readmission)
            loss = loss_mortality + loss_readmission

            loss.backward()
            optimizer.step()

        scheduler.step(loss)

        metrics = evaluate_model(multimodal_model, structured_dataloader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Metrics: {metrics}")

if __name__ == "__main__":
    train_pipeline()
