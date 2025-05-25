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
    """Split note text into chunks of ~512 words for embedding."""
    tokens = text.split()
    return [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

def embed_chunks(chunks: list[str], tokenizer, model, device) -> np.ndarray:
    embs = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            cls_vec = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embs.append(cls_vec)
    if embs:
        return np.mean(embs, axis=0)
    else:
        return np.zeros(model.config.hidden_size)

def categorize_race(race_val):
    r = str(race_val).upper()
    if 'HISPANIC' in r or 'LATINO' in r or 'SOUTH AMERICAN' in r or 'PORTUGUESE' in r or 'BRAZILIAN' in r:
        return 'Hispanic'
    if 'BLACK' in r or 'AFRICAN' in r or 'CAPE VERDEAN' in r or 'CARIBBEAN ISLAND' in r or 'HAITIAN' in r:
        return 'Black'
    if 'ASIAN' in r or 'KOREAN' in r or 'CHINESE' in r or 'INDIAN' in r or 'SOUTH EAST ASIAN' in r:
        return 'Asian'
    if 'WHITE' in r or 'EASTERN EUROPEAN' in r or 'RUSSIAN' in r or 'OTHER EUROPEAN' in r:
        return 'White'
    if ('AMERICAN INDIAN' in r or 'ALASKA NATIVE' in r or 'NATIVE HAWAIIAN' in r or 
        'PACIFIC ISLANDER' in r or 'MIDDLE EASTERN' in r or 'MULTIPLE' in r or
        r in {'OTHER', 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN', 'UNKNOWN/NOT SPECIFIED'}):
        return 'Other'
    return 'Other'

def categorize_ethnicity(race_val):
    r = str(race_val).upper()
    return 'Hispanic' if ('HISPANIC' in r or 'LATINO' in r) else 'Non-Hispanic'

def categorize_insurance(ins):
    if pd.isna(ins):
        return 'Other'
    i = str(ins).strip().lower()
    if i == 'medicare':
        return 'Medicare'
    if i == 'medicaid':
        return 'Medicaid'
    if i == 'private':
        return 'Private'
    return 'Other'

def categorize_age(age):
    if pd.isna(age):
        return 'Other'
    if 18 <= age <= 29:
        return '18-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    elif 70 <= age <= 90:
        return '70-90'
    elif age > 90:
        return '90+'
    else:
        return 'Other'

admissions = pd.read_csv('admissions.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','admittime','dischtime','deathtime','insurance','race'],
    parse_dates=['admittime','dischtime','deathtime'])
patients = pd.read_csv('patients.csv.gz', compression='gzip',
    usecols=['subject_id','gender','anchor_age','dod'], parse_dates=['dod'])
icustays = pd.read_csv('icustays.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','stay_id','intime','outtime'],
    parse_dates=['intime','outtime'])
labevents = pd.read_csv('labevents.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','itemid','charttime','valuenum'],
    parse_dates=['charttime'], low_memory=False)
notes = pd.read_csv('radiology.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','charttime','text'],
    parse_dates=['charttime'], low_memory=False)
diag = pd.read_csv('diagnoses_icd.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','icd_code'])

first_icu = (
    icustays
    .sort_values('intime')
    .drop_duplicates('subject_id', keep='first')
    .merge(patients[['subject_id','anchor_age','gender','dod']], on='subject_id', how='left')
)
first_icu = first_icu[first_icu['anchor_age'] >= 18]
first_icu['dur_h'] = (first_icu['outtime'] - first_icu['intime']).dt.total_seconds() / 3600
first_icu = first_icu[first_icu['dur_h'] >= 0]

first_icu = first_icu.merge(
    admissions[['subject_id','hadm_id','deathtime','insurance','race','admittime']],
    on=['subject_id','hadm_id'], how='left'
)

first_icu['short_term_mortality'] = (
    (first_icu['deathtime'].notnull()) & (first_icu['deathtime'] <= first_icu['outtime'])
).astype(int)

first_icu['age_bucket'] = first_icu['anchor_age'].apply(categorize_age)
first_icu['race_group'] = first_icu['race'].apply(categorize_race)
first_icu['ethnicity'] = first_icu['race'].apply(categorize_ethnicity)
first_icu['insurance_group'] = first_icu['insurance'].apply(categorize_insurance)

# PE/PH flags from ICD codes
diag['code'] = diag['icd_code'].str.replace('.', '', regex=False).fillna('')
diag['pe'] = diag['code'].str.startswith('415').astype(int)
diag['ph'] = diag['code'].str.startswith('416').astype(int)
flags = diag.groupby(['subject_id','hadm_id'])[['pe','ph']].max().reset_index()
cohort = first_icu.merge(flags, on=['subject_id','hadm_id'], how='left').fillna({'pe':0,'ph':0})

labs = labevents.merge(
    cohort[['subject_id','hadm_id','intime']],
    on=['subject_id','hadm_id'], how='inner'
)
labs['delta_h'] = (labs['charttime'] - labs['intime']).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0,24)]
labs['hour_bin'] = (labs['delta_h'] // 2).astype(int)
labs['lab_col'] = labs.apply(lambda r: f"lab_{r['itemid']}_b{int(r['hour_bin'])}", axis=1)
lab_agg = labs.pivot_table(
    index=['subject_id','hadm_id'],
    columns='lab_col',
    values='valuenum',
    aggfunc='mean'
).reset_index()
structured = cohort.merge(lab_agg, on=['subject_id','hadm_id'], how='left')

notes24 = notes.merge(
    cohort[['subject_id','hadm_id','intime']],
    on=['subject_id','hadm_id'], how='inner'
)
notes24['delta_h'] = (notes24['charttime'] - notes24['intime']).dt.total_seconds() / 3600
notes24 = notes24[notes24['delta_h'].between(0,24)]
agg_notes = notes24.groupby(['subject_id','hadm_id'])['text'].apply(
    lambda s: ' '.join(s.astype(str))).reset_index(name='full_text')
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

demog_cols = [
    'subject_id','hadm_id','short_term_mortality','pe','ph',
    'gender','anchor_age','age_bucket','race_group','ethnicity','insurance_group'
]
unstructured = emb_df.merge(
    cohort[demog_cols], on=['subject_id','hadm_id'], how='left'
)

subj_struct = set(structured['subject_id'].unique())
subj_unstruct = set(unstructured['subject_id'].unique())
common_subjs = subj_struct & subj_unstruct

structured_common = structured[structured['subject_id'].isin(common_subjs)].reset_index(drop=True)
unstructured_common = unstructured[unstructured['subject_id'].isin(common_subjs)].reset_index(drop=True)

structured_common.to_csv('final_structured_dataset.csv', index=False)
unstructured_common.to_csv('final_unstructured_embeddings.csv', index=False)

print("Structured final shape:", structured_common.shape)
print("Structured mortality positives:", structured_common['short_term_mortality'].sum())
print("Structured PE positives:", structured_common['pe'].sum())
print("Structured PH positives:", structured_common['ph'].sum())
print("Unstructured final shape:", unstructured_common.shape)
print("Unstructured mortality positives:", unstructured_common['short_term_mortality'].sum())
print("Unstructured PE positives:", unstructured_common['pe'].sum())
print("Unstructured PH positives:", unstructured_common['ph'].sum())
print("Number of common subjects:", len(common_subjs))
