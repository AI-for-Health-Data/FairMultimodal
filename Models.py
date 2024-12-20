import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertModel, BertConfig, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

# Define the BioClinicalBERT Fine-tuned Model
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, BioBert, BioBertConfig, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = BioBert
        self.device = device
        self.config = BioBertConfig  

    def forward(self, input_ids, attention_mask):
        # Extract CLS embedding from the last hidden state
        output = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  
        return cls_embedding

# Define the BEHRT Model
class BEHRTModel(nn.Module):
    def __init__(self, num_diseases, num_ages, num_segments, num_admission_locs, num_discharge_locs, 
                 num_genders, num_ethnicities, num_insurances, hidden_size=768):
        super(BEHRTModel, self).__init__()
        
        # Store the configuration
        self.config = BertConfig(
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
        self.bert = BertModel(self.config)

    def forward(self, input_ids, attention_mask):
        # Extract CLS embedding from the last hidden state
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :] 
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

        self.sigmoid = nn.Sigmoid()

    def forward(self, ts_inputs, ts_attention_mask, text_inputs, text_attention_mask):
        # BEHRT
        ts_outputs = self.BEHRT(input_ids=ts_inputs, attention_mask=ts_attention_mask)
        ts_cls_embedding = ts_outputs.last_hidden_state[:, 0, :]
        ts_projected = self.ts_projector(ts_cls_embedding)

        # BioBERT
        text_outputs = self.BioBert(input_ids=text_inputs, attention_mask=text_attention_mask)
        text_cls_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_projected = self.text_projector(text_cls_embedding)

        # Combine embeddings
        combined_embeddings = torch.cat((ts_projected, text_projected), dim=1)
        logits = self.classifier(combined_embeddings).squeeze(-1)
        probs = self.sigmoid(logits)
        return logits, probs

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
    structured_dataset = TensorDataset(structured_inputs, structured_attention_mask)
    structured_dataloader = DataLoader(structured_dataset, batch_size=32, shuffle=True)

    # Initialize BEHRT
    behrt_model = BertModel(BertConfig(hidden_size=768))

    # Load unstructured data
    unstructured_data = pd.read_csv('first_notes_unstructured.csv').sample(100)
    print("Columns in unstructured_data:", unstructured_data.columns)

    if 'note' not in unstructured_data.columns:
        raise KeyError("The 'note' column is missing from the unstructured data. Please check the file.")

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
        tokenized_notes['attention_mask']
    )
    note_dataloader = DataLoader(note_dataset, batch_size=32, shuffle=True)

    # Initialize BioClinicalBERT
    biobert_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Initialize Multimodal Model
    multimodal_model = MultimodalTransformer(BioBert=biobert_model, BEHRT=behrt_model, device=device).to(device)

    # Training setup
    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        multimodal_model.train()
        for (structured_batch, unstructured_batch) in zip(structured_dataloader, note_dataloader):
            structured_inputs, structured_attention_mask = structured_batch
            unstructured_inputs, unstructured_attention_mask = unstructured_batch

            structured_inputs = structured_inputs.to(device)
            structured_attention_mask = structured_attention_mask.to(device)
            unstructured_inputs = unstructured_inputs.to(device)
            unstructured_attention_mask = unstructured_attention_mask.to(device)

            optimizer.zero_grad()

            logits, probs = multimodal_model(
                ts_inputs=structured_inputs,
                ts_attention_mask=structured_attention_mask,
                text_inputs=unstructured_inputs,
                text_attention_mask=unstructured_attention_mask
            )

            labels = torch.randint(0, 2, (logits.size(0),), dtype=torch.float32).to(device)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    train_pipeline()
