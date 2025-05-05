import pandas as pd
import numpy as np
import re

def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def extract_hispanic_flag(eth):
    eth = str(eth).upper()
    return 1 if 'HISPANIC' in eth else 0

def extract_race(eth):
    eth = str(eth).upper()
    if 'WHITE' in eth:
        return 'White'
    if 'BLACK' in eth:
        return 'Black'
    if 'ASIAN' in eth:
        return 'Asian'
    return 'Other'

def preprocess_notes(text):
    """Clean clinical note text."""
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def chunk_text(text, chunk_size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

icustays = pd.read_csv(
    'ICUSTAYS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','INTIME'],
    parse_dates=['INTIME']
)
icustays.rename(columns={
    'SUBJECT_ID':'subject_id',
    'HADM_ID':'hadm_id',
    'INTIME':'intime'
}, inplace=True)

admissions = pd.read_csv(
    'ADMISSIONS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DEATHTIME','ETHNICITY','INSURANCE'],
    parse_dates=['ADMITTIME','DEATHTIME']
)
admissions.rename(columns={
    'SUBJECT_ID':'subject_id',
    'HADM_ID':'hadm_id',
    'ADMITTIME':'admit_time',
    'DEATHTIME':'death_time'
}, inplace=True)
admissions['is_hispanic'] = admissions['ETHNICITY'].apply(extract_hispanic_flag)
admissions['race']        = admissions['ETHNICITY'].apply(extract_race)
admissions['ethnicity']   = admissions['ETHNICITY']

patients = pd.read_csv(
    'PATIENTS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','DOB','GENDER'],
    parse_dates=['DOB']
)
patients.rename(columns={'SUBJECT_ID':'subject_id'}, inplace=True)


df = (
    icustays
    .merge(admissions, on=['subject_id','hadm_id'], how='left')
    .merge(patients,   on='subject_id',        how='left')
)
assert 'intime' in df.columns, "intime missing!"
df = df.sort_values('intime').groupby('subject_id', as_index=False).first()


df['age']       = df.apply(lambda r: calculate_age(r['DOB'], r['intime']), axis=1)
df['gender']    = df['GENDER'].str.strip().str.lower().map({'m':'male','f':'female'}).fillna(df['GENDER'])
df['mortality'] = df['death_time'].notnull().astype(int)

structured = df[[
    'subject_id','hadm_id','age','gender',
    'race','is_hispanic','ethnicity','INSURANCE','mortality'
]].rename(columns={'INSURANCE':'insurance'})

lab_itemids = [51221,51480,51265,50811,51222,51249,51248,51250,51279,51277,
               50902,50868,50912,50809,50931,51478,50960,50893,50970,51237,
               51274,51275,51375,51427,51446,51116,51244,51355,51379,51120,
               51254,51256,51367,51387,51442,51112,51146,51345,51347,51368,
               51419,51444,51114,51200,51474,50820,50831,51094,51491,50802,
               50804,50818,51498,50813,50861,50878,50863,50862,490,1165,50819]

labs = pd.read_csv(
    'LABEVENTS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','CHARTTIME','ITEMID','VALUENUM'],
    parse_dates=['CHARTTIME']
)
labs.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)
labs = labs[labs['VALUENUM'].notnull() & labs['ITEMID'].isin(lab_itemids)]

labs = labs.merge(df[['subject_id','hadm_id','intime']], on=['subject_id','hadm_id'], how='inner')
labs['delta_h'] = (labs['CHARTTIME'] - labs['intime']).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0, 24)]

labs['lab_col'] = labs.apply(
    lambda r: f"lab_{r['ITEMID']}_bin{int(r['delta_h'] // 2)}", axis=1
)

lab_wide = labs.pivot_table(
    index=['subject_id','hadm_id'],
    columns='lab_col',
    values='VALUENUM',
    aggfunc='mean'
).reset_index()

structured = structured.merge(lab_wide, on=['subject_id','hadm_id'], how='left')

notes = pd.read_csv(
    'NOTEEVENTS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','CHARTTIME','TEXT'],
    parse_dates=['CHARTTIME']
)
notes.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)

notes = notes.merge(df[['subject_id','hadm_id','intime']], on=['subject_id','hadm_id'])
notes['delta_h'] = (notes['CHARTTIME'] - notes['intime']).dt.total_seconds()/3600
notes = notes[notes['delta_h'].between(0,24)]

agg_txt = notes.groupby(['subject_id','hadm_id'])['TEXT'] \
               .apply(lambda s: " ".join(s.astype(str))).reset_index()
agg_txt['clean'] = agg_txt['TEXT'].apply(preprocess_notes)
chunks = agg_txt['clean'].apply(chunk_text)

notes_expanded = pd.DataFrame(
    chunks.tolist(),
    index=agg_txt.set_index(['subject_id','hadm_id']).index
).reset_index()

unstructured = notes_expanded.merge(
    structured[['subject_id','hadm_id','age','gender',
                'race','is_hispanic','ethnicity','insurance','mortality']],
    on=['subject_id','hadm_id'], how='left'
)

structured.to_csv('structured_24h_mortality_demogs_labs.csv', index=False)
unstructured.to_csv('unstructured_24h_notes_with_demogs.csv', index=False)

print("Done:", structured.shape, unstructured.shape)

struct = pd.read_csv('structured_24h_mortality_demogs_labs.csv')
unstruct = pd.read_csv('unstructured_24h_notes_with_demogs.csv')

common_ids = set(struct['subject_id']).intersection(unstruct['subject_id'])

struct_common   = struct[ struct['subject_id'].isin(common_ids) ].copy()
unstruct_common = unstruct[ unstruct['subject_id'].isin(common_ids) ].copy()

print(f"Structured (common IDs): {struct_common.shape}")
print(f"  → Positive mortality count: {struct_common['mortality'].sum()}")

print(f"Unstructured (common IDs): {unstruct_common.shape}")
print(f"  → Positive mortality count: {unstruct_common['mortality'].sum()}")
