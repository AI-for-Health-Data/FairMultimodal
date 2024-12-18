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
        # Extract CLS embedding from the last layer
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
        # Extract CLS embedding from last layer
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
        self.combined_embed_size = 256 * 2  #can change this if needed

        # Transformer encoder for combined embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_embed_size, 
            nhead=4, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        # Final classifier layers
        self.classifier_mortality = nn.Linear(self.combined_embed_size, 1)  # Short-term mortality
        self.classifier_readmission = nn.Linear(self.combined_embed_size, 1)  # Readmission
        #add mechanical ventilation later
        
        # Activation
        self.sigmoid = nn.Sigmoid()

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, ts_inputs, ts_attention_mask, text_inputs, text_attention_mask):
        """
        Forward pass for the multimodal transformer.

        Args:
            ts_inputs: Structured data tokenized for BEHRT (batch_size, seq_len)
            ts_attention_mask: Attention mask for BEHRT (batch_size, seq_len)
            text_inputs: Text data tokenized for BioBERT (batch_size, seq_len)
            text_attention_mask: Attention mask for BioBERT (batch_size, seq_len)

        Returns:
            logits_mortality: Raw output for short-term mortality (batch_size,)
            logits_readmission: Raw output for readmission (batch_size,)
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

        # Final classifier outputs for the two predictions
        logits_mortality = self.classifier_mortality(combined_cls_embedding).squeeze(-1)
        logits_readmission = self.classifier_readmission(combined_cls_embedding).squeeze(-1)

        return logits_mortality, logits_readmission

    def compute_loss(self, logits_mortality, logits_readmission, labels):
        """
        Compute BCE loss for the given logits and labels.

        Args:
            logits_mortality: Raw output for short-term mortality (batch_size,)
            logits_readmission: Raw output for readmission (batch_size,)
            labels: Ground truth labels for both predictions (batch_size, 2)

        Returns:
            loss: Combined loss value for both predictions
        """
        loss_mortality = self.criterion(logits_mortality, labels[:, 0])
        loss_readmission = self.criterion(logits_readmission, labels[:, 1])
        loss = loss_mortality + loss_readmission
        return loss
