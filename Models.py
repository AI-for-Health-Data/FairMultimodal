import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

# BioClinicalBERT Model
class BioClinicalBERT_FT(nn.Module):
    def __init__(self, BioBert, BioBertConfig, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = BioBert
        self.device = device

    def forward(self, input_ids, attention_mask):
        # Extract CLS embedding from the last hidden state
        output = self.BioBert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

# BEHRT Model
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

    def forward(self, input_ids, attention_mask):
        # Extract CLS embedding from the last hidden state
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

class MultimodalTransformer(nn.Module):
    def __init__(self, BioBert, BEHRT, device):
        super(MultimodalTransformer, self).__init__()

        # Initialize BioClinicalBERT and BEHRT
        self.BioBert = BioBert
        self.BEHRT = BEHRT
        self.device = device

        # Time series: BEHRT-specific embedding size
        self.ts_embed_size = BEHRT.config.hidden_size

        # Text: BioClinicalBERT-specific embedding size
        self.text_embed_size = BioBert.config.hidden_size

        # Projection layers for aligning embeddings from BEHRT and BioClinicalBERT
        self.ts_projector = nn.Sequential(
            nn.Linear(self.ts_embed_size, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_embed_size, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False)
        )

        # Combine both embeddings
        self.combined_embed_size = 256 * 2  # 256 from time series, 256 from text

        # Transformer encoder for combined embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_embed_size, 
            nhead=4, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        # Final classifier layer
        self.classifier = nn.Linear(self.combined_embed_size, 1)
        
        # Activation
        self.sigmoid = nn.Sigmoid()

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Weight initialization for final layer
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, ts_inputs, ts_attention_mask, text_inputs, text_attention_mask):
        """
        Forward pass for the multimodal transformer.

        Args:
            ts_inputs: Structured data tokenized for BEHRT (batch_size, seq_len)
            ts_attention_mask: Attention mask for BEHRT (batch_size, seq_len)
            text_inputs: Text data tokenized for BioBERT (batch_size, seq_len)
            text_attention_mask: Attention mask for BioBERT (batch_size, seq_len)

        Returns:
            logits: Raw output from the classifier (batch_size,)
            probs: Probability outputs after sigmoid activation (batch_size,)
        """
        # Process structured data through BEHRT
        ts_outputs = self.BEHRT(input_ids=ts_inputs, attention_mask=ts_attention_mask)
        ts_cls_embedding = ts_outputs.last_hidden_state[:, 0, :]  # Extract [CLS] embedding
        ts_projected = self.ts_projector(ts_cls_embedding)  # Project to lower dimension

        # Process text data through BioClinicalBERT
        text_outputs = self.BioBert(input_ids=text_inputs, attention_mask=text_attention_mask)
        text_cls_embedding = text_outputs.last_hidden_state[:, 0, :]  # Extract [CLS] embedding
        text_projected = self.text_projector(text_cls_embedding)  # Project to lower dimension

        # Combine projected embeddings
        combined_embeddings = torch.cat((ts_projected, text_projected), dim=1).unsqueeze(1)

        # Transformer encoder for combined embeddings
        transformer_output = self.transformer_encoder(combined_embeddings)
        combined_cls_embedding = transformer_output[:, 0, :]  # Extract [CLS] embedding

        # Final classifier
        logits = self.classifier(combined_cls_embedding).squeeze(-1)
        probs = self.sigmoid(logits)

        return logits, probs

    def compute_loss(self, logits, labels):
        """
        Compute BCE loss for the given logits and labels.

        Args:
            logits: Raw output from the classifier (batch_size,)
            labels: Ground truth labels (batch_size,)

        Returns:
            loss: Computed loss value
        """
        loss = self.criterion(logits, labels)
        return loss

    def get_l2_regularization(self):
        """
        Compute L2 regularization for the classifier weights.

        Returns:
            l2_reg: L2 regularization value
        """
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.classifier.parameters():
            l2_reg += param.norm(2)
        return l2_reg


# Training pipeline integrating BEHRT and BioClinicalBERT
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load structured data for BEHRT
    structured_data = pd.read_csv('structured_first_admissions.csv')

    # Preprocess structured data (encode categorical variables, etc.)
    structured_data['age'] = structured_data['age'].astype(int)
    structured_data['admission_location'] = structured_data['ADMISSION_LOCATION'].astype('category').cat.codes
    structured_data['discharge_location'] = structured_data['DISCHARGE_LOCATION'].astype('category').cat.codes
    structured_data['gender'] = structured_data['GENDER'].astype('category').cat.codes
    structured_data['ethnicity'] = structured_data['ETHNICITY'].astype('category').cat.codes
    structured_data['insurance'] = structured_data['INSURANCE'].astype('category').cat.codes

    # Define parameters for BEHRT
    num_diseases = len(structured_data['DIAGNOSIS'].unique())
    num_ages = structured_data['age'].nunique()
    num_segments = 2
    num_admission_locs = structured_data['admission_location'].nunique()
    num_discharge_locs = structured_data['discharge_location'].nunique()
    num_genders = structured_data['gender'].nunique()
    num_ethnicities = structured_data['ethnicity'].nunique()
    num_insurances = structured_data['insurance'].nunique()

    # Initialize BEHRT
    behrt_model = BEHRTModel(
        num_diseases=num_diseases,
        num_ages=num_ages,
        num_segments=num_segments,
        num_admission_locs=num_admission_locs,
        num_discharge_locs=num_discharge_locs,
        num_genders=num_genders,
        num_ethnicities=num_ethnicities,
        num_insurances=num_insurances,
    ).to(device)

    # Load unstructured data for BioClinicalBERT
    unstructured_data = pd.read_csv('unstructured_first_admissions.csv')  # Replace with your actual unstructured data file
    notes = unstructured_data[unstructured_data['CATEGORY'] == 'Discharge summary']
    notes['TEXT'] = notes['TEXT'].fillna('')

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenized_notes = tokenizer(
        list(notes['TEXT']),
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
    biobert_ft_model = BioClinicalBERT_FT(biobert_model, biobert_model.config, device).to(device)

    # Initialize multimodal model
    multimodal_model = MultimodalModel(behrt_hidden_size=768, biobert_hidden_size=768).to(device)

    print("BEHRT and BioClinicalBERT models initialized.")
    print("Starting multimodal training pipeline...")

    # Placeholder for combined training logic
    for structured_batch, unstructured_batch in zip(DataLoader(...), note_dataloader):
        # Process structured data through BEHRT
        structured_inputs = ...  # Prepare structured inputs for BEHRT
        behrt_cls = behrt_model(*structured_inputs)  # Extract BEHRT CLS embeddings

        # Process unstructured data through BioClinicalBERT
        input_ids, attention_mask = unstructured_batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        biobert_cls = biobert_ft_model(input_ids, attention_mask)  # Extract BioClinicalBERT CLS embeddings

        # Combine embeddings and predict
        logits, probs = multimodal_model(behrt_cls, biobert_cls)
        # Compute loss and update weights (add loss logic)

if __name__ == "__main__":
    train_pipeline()
