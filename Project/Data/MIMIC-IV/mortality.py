#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re

def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def preprocess_notes(text):
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def chunk_text(text, chunk_size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


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
    parse_dates=['charttime']
)

notes = pd.read_csv(
    'radiology.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','charttime','text'],
    parse_dates=['charttime'], low_memory=False
)

diagnoses = pd.read_csv(
    'diagnoses_icd.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','icd_code']
)

first_icu = (
    icustays
    .sort_values('intime')
    .drop_duplicates('subject_id', keep='first')
)

first_icu = (
    first_icu
    .merge(patients, on='subject_id', how='left')
    .merge(
        admissions[['subject_id','hadm_id','deathtime','insurance','race']],
        on=['subject_id','hadm_id'], how='left'
    )
)

first_icu['short_term_mortality'] = (
    first_icu['deathtime'].notnull() &
    (first_icu['deathtime'] <= first_icu['outtime'])
).astype(int)

def age_bucket(a):
    if 15 <= a <= 29: return '15-29'
    if 30 <= a <= 49: return '30-49'
    if 50 <= a <= 69: return '50-69'
    if 70 <= a <= 89: return '70-89'
    return 'Other'
first_icu['age_bucket'] = first_icu['anchor_age'].apply(age_bucket)

def categorize_race(r):
    r2 = str(r).upper()
    if 'WHITE'   in r2: return 'White'
    if 'BLACK'   in r2: return 'Black'
    if 'HISPANIC' in r2 or 'LATINO' in r2: return 'Hispanic'
    if 'ASIAN'   in r2: return 'Asian'
    return 'Other'
first_icu['race_group'] = first_icu['race'].apply(categorize_race)

first_icu['ethnicity'] = first_icu['race'].str.upper().apply(
    lambda r: 'Hispanic' if 'HISPANIC' in r or 'LATINO' in r else 'Non-Hispanic'
)

def bucket_insurance(ins):
    i = str(ins).title()
    if i in ['Medicare','Medicaid','Private']: return i
    return 'Other'
first_icu['insurance_group'] = first_icu['insurance'].apply(bucket_insurance)


first_icu['age_years'] = first_icu['anchor_age']

# PE & PH FLAGS 

# normalize ICD code column and flag
diagnoses['code'] = diagnoses['icd_code'].str.replace('.', '', regex=False).fillna('')
diagnoses['pe']   = diagnoses['code'].str.startswith('415').astype(int)
diagnoses['ph']   = diagnoses['code'].str.startswith('416').astype(int)

# one flag per stay
flags = (
    diagnoses
    .groupby(['subject_id','hadm_id'])[['pe','ph']]
    .max()
    .reset_index()
)

cohort = first_icu.merge(flags, on=['subject_id','hadm_id'], how='left').fillna({'pe':0,'ph':0})

labs = (
    labevents
    # keep only non‑null numeric values; no itemid filter
    .loc[lambda df: df['valuenum'].notnull()]
    .merge(cohort[['subject_id','hadm_id','intime']], on=['subject_id','hadm_id'], how='inner')
)

labs['delta_h'] = (labs['charttime'] - labs['intime']).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0,24)]
labs['hour_bin'] = (labs['delta_h']//2).astype(int)
labs['lab_col'] = labs.apply(lambda r: f"lab_{r['itemid']}_b{int(r['hour_bin'])}", axis=1)

lab_agg = (
    labs
    .pivot_table(
        index=['subject_id','hadm_id'],
        columns='lab_col',
        values='valuenum',
        aggfunc='mean'
    )
    .reset_index()
)

structured = cohort.merge(lab_agg, on=['subject_id','hadm_id'], how='left')

structured.to_csv('structured_dataset.csv', index=False)
print("structured_dataset shape:", structured.shape)
print("short_term_mortality positives:", structured['short_term_mortality'].sum())

notes24 = (
    notes
    .merge(cohort[['subject_id','hadm_id','intime']], on=['subject_id','hadm_id'])
)
notes24['delta_h'] = (notes24['charttime'] - notes24['intime']).dt.total_seconds() / 3600
notes24 = notes24[notes24['delta_h'].between(0,24)]

agg = (
    notes24
    .groupby('subject_id')['text']
    .apply(lambda s: " ".join(s.astype(str)))
    .reset_index()
)
agg['cleaned'] = agg['text'].apply(preprocess_notes)
agg['chunks']  = agg['cleaned'].apply(chunk_text)

notes_expanded = pd.DataFrame(
    agg['chunks'].tolist(),
    index=agg.set_index('subject_id').index
).reset_index()

unstructured = (
    cohort[['subject_id','hadm_id','short_term_mortality','pe','ph']]
    .merge(notes_expanded, on='subject_id', how='left')
)

unstructured.to_csv('unstructured_chunks.csv', index=False)
print("unstructured_chunks shape:", unstructured.shape)
print("short_term_mortality positives:", unstructured['short_term_mortality'].sum())
print("PE positives:             ", unstructured['pe'].sum())
print("PH positives:             ", unstructured['ph'].sum())
