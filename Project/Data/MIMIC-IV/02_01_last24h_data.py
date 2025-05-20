import os
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from transformers import AutoTokenizer
import torch

def preprocess_notes(text: str) -> str:
    """Basic cleaning: remove bracketed text, numeric lists, extra whitespace."""
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()


def split_and_expand(df: pd.DataFrame,
                     text_col: str,
                     tokenizer: AutoTokenizer,
                     max_length: int = 512,
                     stride: int = 0
                    ) -> tuple[pd.DataFrame, list[str]]:

    chunked = df[text_col].fillna('').apply(
        lambda t: tokenizer.encode(t, add_special_tokens=False)
    ).apply(
        lambda ids: [ids[i:i + max_length] for i in range(0, len(ids), max_length - stride or max_length)]
    )
    max_chunks = chunked.map(len).max()

    note_cols = []
    for i in range(max_chunks):
        col = f'note_{i}'
        df[col] = chunked.map(lambda chunks: tokenizer.decode(chunks[i], clean_up_tokenization_spaces=True) if i < len(chunks) else "")
        note_cols.append(col)

    return df, note_cols


def get_readmission_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['subject_id', 'intime'])
    df['next_intime'] = df.groupby('subject_id')['intime'].shift(-1)
    df['readmit_30d'] = (
        (df['next_intime'] - df['outtime']) <= pd.Timedelta(days=30)
    ).fillna(False).astype(int)
    return df[['subject_id', 'hadm_id', 'stay_id', 'readmit_30d']]

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
stays['dur_h'] = (stays['outtime'] - stays['intime']).dt.total_seconds() / 3600
stays = stays[stays['dur_h'] >= 30]

readmit_flags = get_readmission_flag(stays)
first_icu = (
    stays.sort_values('intime')
         .drop_duplicates('subject_id', keep='first')
)
cohort = first_icu.merge(
    readmit_flags,
    on=['subject_id','hadm_id','stay_id'],
    how='left'
).fillna({'readmit_30d': 0})

data = cohort[['subject_id','hadm_id','stay_id','outtime']]
labs = labevents.merge(data, on=['subject_id','hadm_id'], how='inner')
labs['delta_h'] = (labs['outtime'] - labs['charttime']).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0, 24)]
labs['hour_bin'] = (labs['delta_h'] // 2).astype(int)
labs['lab_col'] = labs.apply(lambda r: f"lab_{r['itemid']}_b{r['hour_bin']}", axis=1)
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

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', use_fast=True)

notes24 = notes.merge(data, on=['subject_id','hadm_id'], how='inner')
notes24['delta_h'] = (notes24['outtime'] - notes24['charttime']).dt.total_seconds() / 3600
notes24 = notes24[notes24['delta_h'].between(0, 24)]

agg_notes = (
    notes24.groupby(['subject_id','hadm_id'])['text']
           .apply(lambda s: ' '.join(s.astype(str)))
           .reset_index()
)
agg_notes['cleaned_text'] = agg_notes['text'].apply(preprocess_notes)

# Merge the cleaned text back into cohort
cohort_notes = cohort.merge(
    agg_notes[['subject_id','hadm_id','cleaned_text']],
    on=['subject_id','hadm_id'],
    how='left'
)

# Tokenize & expand into note_0, note_1, ...
cohort_expanded, note_cols = split_and_expand(
    cohort_notes, 'cleaned_text', tokenizer, max_length=512, stride=50
)

out_cols = ['subject_id','hadm_id','stay_id','readmit_30d','anchor_age','gender'] + note_cols
cohort_expanded[out_cols].to_csv('unstructured_512token_notes.csv', index=False)
print("Unstructured final shape:", cohort_expanded[out_cols].shape)
print(f"Created {len(note_cols)} note chunk columns per patient.")

common_ids = set(structured['subject_id']).intersection(
    cohort_expanded['subject_id']
)
structured_common = structured[structured['subject_id'].isin(common_ids)].copy()
unstructured_common = cohort_expanded[cohort_expanded['subject_id'].isin(common_ids)].copy()

structured_common.to_csv('structured_common_subjects.csv', index=False)
unstructured_common.to_csv('unstructured_common_subjects.csv', index=False)

print("Structured (common IDs) shape:", structured_common.shape)
print("Structured readmission positives:", structured_common['readmit_30d'].sum())
print("Unstructured (common IDs) shape:", unstructured_common.shape)
print("Unstructured readmission positives:", unstructured_common['readmit_30d'].sum())
