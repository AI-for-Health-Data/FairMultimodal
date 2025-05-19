import os
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel
import torch

def preprocess_notes(text):
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()


def chunk_text(text, chunk_size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


def get_readmission_flag(df_stays):
    # Flag if patient has another ICU stay within 30 days after this stay
    df = df_stays.sort_values(['subject_id','intime'])
    # next ICU admission time
    df['next_intime'] = df.groupby('subject_id')['intime'].shift(-1)
    df['readmit_30d'] = ((df['next_intime'] - df['outtime']) <= pd.Timedelta(days=30)).fillna(False).astype(int)
    return df[['subject_id','hadm_id','stay_id','readmit_30d']]


def embed_chunks(chunks, tokenizer, model, device):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            cls_emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_emb)
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.config.hidden_size)

admissions = pd.read_csv(
    'admissions.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','admittime','dischtime','insurance','race'],
    parse_dates=['admittime','dischtime']
)

patients = pd.read_csv(
    'patients.csv.gz', compression='gzip',
    usecols=['subject_id','gender','anchor_age']
)

icustays = pd.read_csv(
    'icustays.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','stay_id','intime','outtime'],
    parse_dates=['intime','outtime']
)

labevents = pd.read_csv(
    'labevents.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','itemid','charttime','valuenum'],
    parse_dates=['charttime'], low_memory=False
)

notes = pd.read_csv(
    'discharge.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','charttime','text'],
    parse_dates=['charttime'], low_memory=False
)

stays = icustays.merge(patients, on='subject_id', how='left')
stays = stays[stays['anchor_age'] >= 18]
stays['dur_h'] = (stays['outtime'] - stays['intime']).dt.total_seconds()/3600
stays = stays[stays['dur_h'] >= 30]
# Flag readmission on all stays
readmit_flags = get_readmission_flag(stays)
# Select first stay per patient
first_icu = (
    stays.sort_values('intime')
         .drop_duplicates('subject_id', keep='first')
)
cohort = first_icu.merge(
    readmit_flags[['subject_id','hadm_id','stay_id','readmit_30d']],
    on=['subject_id','hadm_id','stay_id'], how='left'
).fillna({'readmit_30d': 0})

data = cohort[['subject_id','hadm_id','stay_id','outtime']]
labs = labevents.merge(data, on=['subject_id','hadm_id'], how='inner')
labs['delta_h'] = (labs['outtime'] - labs['charttime']).dt.total_seconds()/3600
labs = labs[labs['delta_h'].between(0,24)]
labs['hour_bin'] = (labs['delta_h']//2).astype(int)
labs['lab_col'] = labs.apply(lambda r: f"lab_{r['itemid']}_b{int(r['hour_bin'])}", axis=1)
lab_agg = labs.pivot_table(
    index=['subject_id','hadm_id','stay_id'],
    columns='lab_col',
    values='valuenum',
    aggfunc='mean'
).reset_index()
structured = cohort.merge(lab_agg, on=['subject_id','hadm_id','stay_id'], how='left')
structured.to_csv('structured_dataset.csv', index=False)
print("Structured final shape:", structured.shape)
print("Structured readmission positives:", structured['readmit_30d'].sum())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)

notes24 = notes.merge(data, on=['subject_id','hadm_id'], how='inner')
notes24['delta_h'] = (data['outtime'] - notes24['charttime']).dt.total_seconds()/3600
notes24 = notes24[notes24['delta_h'].between(0,24)]
agg_notes = (
    notes24.groupby(['subject_id','hadm_id'])['text']
           .apply(lambda s: ' '.join(s.astype(str)))
           .reset_index()
)
agg_notes['cleaned'] = agg_notes['text'].apply(preprocess_notes)
agg_notes['chunks'] = agg_notes['cleaned'].apply(chunk_text)
# Embed
agg_notes['emb_vector'] = agg_notes['chunks'].apply(lambda ch: embed_chunks(ch, tokenizer, model, device))

unstructured = cohort[['subject_id','hadm_id','stay_id','readmit_30d','anchor_age','gender']].merge(
    agg_notes[['subject_id','hadm_id','emb_vector']],
    on=['subject_id','hadm_id'], how='left'
)
unstructured.to_csv('unstructured_embeddings.csv', index=False)
print("Unstructured final shape:", unstructured.shape)
print("Unstructured readmission positives:", unstructured['readmit_30d'].sum())
