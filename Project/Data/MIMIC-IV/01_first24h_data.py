import os
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel
import torch


def preprocess_notes(text: str) -> str:
    """Remove bracketed text, numeric lists, extra whitespace, lowercase."""
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def chunk_text(text: str, chunk_size: int = 512) -> list[str]:
    """Split on whitespace into fixed-size word chunks."""
    tokens = text.split()
    return [' '.join(tokens[i:i + chunk_size]) 
            for i in range(0, len(tokens), chunk_size)]

def embed_chunks(chunks: list[str],
                 tokenizer: AutoTokenizer,
                 model: AutoModel,
                 device: torch.device) -> np.ndarray:
    """
    Tokenize & embed each chunk with Bio_ClinicalBERT, return mean CLS vector.
    """
    embs = []
    for chunk in chunks:
        inputs = tokenizer(chunk,
                           return_tensors='pt',
                           truncation=True,
                           max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            cls_vec = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embs.append(cls_vec)
    if embs:
        return np.mean(embs, axis=0)
    else:
        # if no chunks, return zero vector
        return np.zeros(model.config.hidden_size)


admissions = pd.read_csv(
    'admissions.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','admittime','dischtime','deathtime','insurance','race'],
    parse_dates=['admittime','dischtime','deathtime']
)
patients = pd.read_csv(
    'patients.csv.gz', compression='gzip',
    usecols=['subject_id','gender','anchor_age','dod'],
    parse_dates=['dod']
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
    'radiology.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','charttime','text'],
    parse_dates=['charttime'], low_memory=False
)
diag = pd.read_csv(
    'diagnoses_icd.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','icd_code']
)


# pick first ICU stay per subject
first_icu = (
    icustays
    .sort_values('intime')
    .drop_duplicates('subject_id', keep='first')
)

first_icu = first_icu.merge(
    patients[['subject_id','anchor_age','gender','dod']],
    on='subject_id', how='left'
)
first_icu = first_icu[first_icu['anchor_age'] >= 18]

# filter stays >= 30 h
first_icu['dur_h'] = (
    first_icu['outtime'] - first_icu['intime']
).dt.total_seconds() / 3600
first_icu = first_icu[first_icu['dur_h'] >= 30]

# merge admissions to get deathtime, insurance, race
first_icu = first_icu.merge(
    admissions[['subject_id','hadm_id','deathtime','insurance','race','admittime']],
    on=['subject_id','hadm_id'], how='left'
)

# compute short-term mortality flag
first_icu['short_term_mortality'] = (
    (first_icu['deathtime'].notnull()) &
    (first_icu['deathtime'] <= first_icu['outtime'])
).astype(int)

# age buckets
first_icu['age_bucket'] = pd.cut(
    first_icu['anchor_age'],
    bins=[18,30,50,70,90,200],
    labels=['18-29','30-49','50-69','70-89','90+'],
    right=False
)

# race & ethnicity grouping
def categorize_race(r):
    r2 = str(r).upper()
    if 'WHITE' in r2:   return 'White'
    if 'BLACK' in r2:   return 'Black'
    if 'HISPANIC' in r2 or 'LATINO' in r2: return 'Hispanic'
    if 'ASIAN' in r2:   return 'Asian'
    return 'Other'

first_icu['race_group'] = first_icu['race'].apply(categorize_race)
first_icu['ethnicity'] = first_icu['race'].str.upper().apply(
    lambda r: 'Hispanic' if 'HISPANIC' in r or 'LATINO' in r else 'Non-Hispanic'
)
first_icu['insurance_group'] = first_icu['insurance'].where(
    first_icu['insurance'].isin(['Medicare','Medicaid','Private']),
    other='Other'
)

# ICD flags for PE & PH
diag['code'] = diag['icd_code'].str.replace('.', '', regex=False).fillna('')
diag['pe'] = diag['code'].str.startswith('415').astype(int)
diag['ph'] = diag['code'].str.startswith('416').astype(int)
flags = diag.groupby(['subject_id','hadm_id'])[['pe','ph']].max().reset_index()

# final cohort
cohort = first_icu.merge(flags, on=['subject_id','hadm_id'], how='left') \
                  .fillna({'pe':0,'ph':0})

labs = labevents.merge(
    cohort[['subject_id','hadm_id','intime']],
    on=['subject_id','hadm_id'], how='inner'
)
labs['delta_h'] = (
    labs['charttime'] - labs['intime']
).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0,24)]
labs['hour_bin'] = (labs['delta_h'] // 2).astype(int)
labs['lab_col'] = labs.apply(
    lambda r: f"lab_{r['itemid']}_b{int(r['hour_bin'])}", axis=1
)

lab_agg = labs.pivot_table(
    index=['subject_id','hadm_id'],
    columns='lab_col',
    values='valuenum',
    aggfunc='mean'
).reset_index()

structured = cohort.merge(lab_agg, on=['subject_id','hadm_id'], how='left')

structured.to_csv('final_structured_dataset.csv', index=False)
print("Structured final shape:", structured.shape)
print("Structured mortality positives:", structured['short_term_mortality'].sum())
print("Structured PE positives:", structured['pe'].sum())
print("Structured PH positives:", structured['ph'].sum())


# filter notes to 0–24h after ICU admission
notes24 = notes.merge(
    cohort[['subject_id','hadm_id','intime']],
    on=['subject_id','hadm_id'], how='inner'
)
notes24['delta_h'] = (
    notes24['charttime'] - notes24['intime']
).dt.total_seconds() / 3600
notes24 = notes24[notes24['delta_h'].between(0,24)]

# concatenate per admission
agg_notes = notes24.groupby(
    ['subject_id','hadm_id']
)['text'].apply(lambda s: ' '.join(s.astype(str))).reset_index(name='full_text')

agg_notes['cleaned'] = agg_notes['full_text'].apply(preprocess_notes)
agg_notes['chunks']  = agg_notes['cleaned'].apply(chunk_text)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
model     = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)

agg_notes['emb_vector'] = agg_notes['chunks'].apply(
    lambda ch: embed_chunks(ch, tokenizer, model, device)
)

emb_df = pd.DataFrame(
    agg_notes['emb_vector'].tolist(),
    index=agg_notes.set_index(['subject_id','hadm_id']).index
)
emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
emb_df = emb_df.reset_index()

unstructured = cohort.merge(
    emb_df, on=['subject_id','hadm_id'], how='left'
)

unstructured.to_csv('final_unstructured_embeddings.csv', index=False)
print("Unstructured final shape:", unstructured.shape)
print("Unstructured mortality positives:", unstructured['short_term_mortality'].sum())
print("Unstructured PE positives:", unstructured['pe'].sum())
print("Unstructured PH positives:", unstructured['ph'].sum())
