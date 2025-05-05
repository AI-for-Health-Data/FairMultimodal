import pandas as pd
import numpy as np
import re

admissions = pd.read_csv(
    'admissions.csv.gz',
    compression='gzip',
    usecols=[
        'subject_id','hadm_id',
        'admittime','dischtime','deathtime',
        'insurance','race'
    ]
)
patients = pd.read_csv(
    'patients.csv.gz',
    compression='gzip',
    usecols=['subject_id','gender','anchor_age','dod']
)
icustays = pd.read_csv(
    'icustays.csv.gz',
    compression='gzip',
    usecols=['subject_id','hadm_id','stay_id','intime','outtime']
)
labevents = pd.read_csv(
    'labevents.csv.gz',
    compression='gzip',
    usecols=['subject_id','hadm_id','itemid','charttime','valuenum']
)

for df, cols in [
    (admissions, ['admittime','dischtime','deathtime']),
    (patients,   ['dod']),
    (icustays,   ['intime','outtime']),
    (labevents,  ['charttime'])
]:
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')

first_icu = (
    icustays
    .sort_values('intime')
    .groupby('subject_id', as_index=False)
    .first()
)

first_icu = (
    first_icu
    .merge(
        patients[['subject_id','gender','anchor_age','dod']],
        on='subject_id', how='left'
    )
    .merge(
        admissions[['subject_id','hadm_id','deathtime']],
        on=['subject_id','hadm_id'], how='left'
    )
)
first_icu['short_term_mortality'] = first_icu['deathtime'].notnull().astype(int)

def age_bucket(a):
    if 15 <= a <= 29: return '15-29'
    if 30 <= a <= 49: return '30-49'
    if 50 <= a <= 69: return '50-69'
    if 70 <= a <= 89: return '70-89'
    return 'Other'
first_icu['age_bucket'] = first_icu['anchor_age'].apply(age_bucket)

first_icu = first_icu.merge(
    admissions[['subject_id','hadm_id','insurance','race']],
    on=['subject_id','hadm_id'], how='left'
)
def categorize_race(r):
    r2 = str(r).upper()
    if 'WHITE' in r2:    return 'White'
    if 'BLACK' in r2:    return 'Black'
    if 'HISPANIC' in r2 or 'LATINO' in r2: return 'Hispanic'
    if 'ASIAN' in r2:    return 'Asian'
    return 'Other'
first_icu['race_group'] = first_icu['race'].apply(categorize_race)
first_icu['ethnicity'] = first_icu['race'].str.upper().apply(
    lambda r: 'Hispanic' if 'HISPANIC' in r or 'LATINO' in r else 'Non-Hispanic'
)
def bucket_insurance(ins):
    i = str(ins).title()
    if i in ['Medicare','Private','Medicaid']: return i
    return 'Other'
first_icu['insurance_group'] = first_icu['insurance'].apply(bucket_insurance)

lab = labevents.merge(
    first_icu[['subject_id','hadm_id','stay_id','intime']],
    on=['subject_id','hadm_id'], how='inner'
)
lab['hrs'] = (lab['charttime'] - lab['intime']).dt.total_seconds() / 3600
lab = lab[(lab['hrs']>=0) & (lab['hrs']<=24)]
lab['hour_bin'] = (lab['hrs']//2).astype(int)
lab_agg = (
    lab.groupby(['subject_id','stay_id','hour_bin','itemid'], as_index=False)['valuenum']
       .mean()
       .pivot_table(
           index=['subject_id','stay_id'],
           columns='itemid',
           values='valuenum'
       )
       .reset_index()
)
lab_agg.columns = ['subject_id','stay_id'] + [
    f"lab_t{int(c)}" for c in lab_agg.columns
    if c not in ['subject_id','stay_id']
]

structured = first_icu.merge(lab_agg, on=['subject_id','stay_id'], how='left')

structured.to_csv('structured_dataset.csv', index=False)
print("Structured dataset shape:", structured.shape)
print("Short-term mortality count:", structured['short_term_mortality'].sum())

# Build Unstructured (Radiology)
cohort = structured[[
    'subject_id','hadm_id','intime','outtime',
    'gender','age_bucket','race_group','ethnicity',
    'insurance_group','short_term_mortality'
]].drop_duplicates('subject_id')

notes = pd.read_csv(
    'radiology.csv.gz',
    compression='gzip',
    usecols=['subject_id','hadm_id','charttime','text'],
    parse_dates=['charttime'], low_memory=False
)

j = notes.merge(
    cohort[['subject_id','hadm_id','intime','outtime']],
    on=['subject_id','hadm_id'], how='inner'
)
j['hrs'] = (j['charttime'] - j['intime']).dt.total_seconds() / 3600
first24 = j[(j['hrs']>=0) & (j['hrs']<=24)]

agg = first24.groupby('subject_id')['text'] \
             .apply(lambda ts: " ".join(ts)) \
             .reset_index()

def preprocess1(x):
    y = re.sub(r'\[(.*?)\]', '', x)
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'dr\.', 'doctor', y, flags=re.IGNORECASE)
    y = re.sub(r'm\.d\.', 'md', y, flags=re.IGNORECASE)
    y = re.sub(r'admission date:', '', y, flags=re.IGNORECASE)
    y = re.sub(r'discharge date:', '', y, flags=re.IGNORECASE)
    return re.sub(r'--|__|==', '', y)

def clean_text(t):
    t = t.replace('\n',' ').replace('\r',' ').strip().lower()
    return preprocess1(t)

def split_chunks(text, size=512):
    tokens = text.split()
    return pd.Series({
        f'note_chunk_{i+1}': " ".join(tokens[i:i+size])
        for i in range(0, len(tokens), size)
    })

agg['cleaned'] = agg['text'].fillna(' ').apply(clean_text)
chunks = agg['cleaned'].apply(split_chunks)
chunks['subject_id'] = agg['subject_id']

unstructured = cohort.merge(chunks, on='subject_id', how='left')

unstructured.to_csv('unstructured_chunks.csv', index=False)
print("Unstructured dataset shape:", unstructured.shape)
print("Short-term mortality count:", unstructured['short_term_mortality'].sum())
