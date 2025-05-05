import pandas as pd
import numpy as np
import re
from datetime import timedelta

def get_readmission_flag(df_stays):
    df = df_stays.sort_values(['subject_id','intime'])
    df['prev_out'] = df.groupby('subject_id')['outtime'].shift(1)
    df['readmit_30d'] = ((df['intime'] - df['prev_out']) <= pd.Timedelta(days=30)).astype(int)
    df['readmit_30d'] = df['readmit_30d'].fillna(0).astype(int)
    return df[['subject_id','hadm_id','icustay_id','readmit_30d']]

def preprocess_notes(text):
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def chunk_text(text, chunk_size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i+chunk_size])
            for i in range(0, len(tokens), chunk_size)]

icustays = pd.read_csv(
    'ICUSTAYS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','INTIME','OUTTIME'],
    parse_dates=['INTIME','OUTTIME']
).rename(columns={
    'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id',
    'ICUSTAY_ID':'icustay_id','INTIME':'intime','OUTTIME':'outtime'
})

admissions = pd.read_csv(
    'ADMISSIONS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','INSURANCE'],
    parse_dates=['ADMITTIME','DISCHTIME']
).rename(columns={
    'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id',
    'ADMITTIME':'admit_time','DISCHTIME':'discharge_time',
    'INSURANCE':'insurance'
})

patients = pd.read_csv(
    'PATIENTS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','DOB','GENDER'],
    parse_dates=['DOB']
).rename(columns={'SUBJECT_ID':'subject_id'})

readmit = get_readmission_flag(icustays)
df = icustays.merge(
    readmit,
    on=['subject_id','hadm_id','icustay_id'], how='left'
)


df = df.merge(
    admissions[['subject_id','hadm_id','insurance']],
    on=['subject_id','hadm_id'], how='left'
).merge(
    patients, on='subject_id', how='left'
)

df['age'] = (
    df['intime'].dt.year - df['DOB'].dt.year
    - ((df['intime'].dt.month < df['DOB'].dt.month) |
       ((df['intime'].dt.month == df['DOB'].dt.month) &
        (df['intime'].dt.day < df['DOB'].dt.day)))
)
df['gender'] = df['GENDER'].str.lower().map({'m':'male','f':'female'}).fillna(df['GENDER'])

structured = df[[
    'subject_id','hadm_id','icustay_id',
    'age','gender','insurance','readmit_30d'
]].rename(columns={'readmit_30d':'readmission'})

lab_itemids = [
    51221,51480,51265,50811,51222,51249,51248,51250,51279,51277,
    50902,50868,50912,50809,50931,51478,50960,50893,50970,51237,
    51274,51275,51375,51427,51446,51116,51244,51355,51379,51120,
    51254,51256,51367,51387,51442,51112,51146,51345,51347,51368,
    51419,51444,51114,51200,51474,50820,50831,51094,51491,50802,
    50804,50818,51498,50813,50861,50878,50863,50862,490,1165,50819
]

labs = pd.read_csv(
    'LABEVENTS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','CHARTTIME','ITEMID','VALUENUM'],
    parse_dates=['CHARTTIME']
).rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'})

labs = labs[labs['VALUENUM'].notnull() & labs['ITEMID'].isin(lab_itemids)]
labs = labs.merge(
    df[['subject_id','hadm_id','icustay_id','outtime']],
    on=['subject_id','hadm_id'], how='inner'
)

labs['delta_h'] = (labs['outtime'] - labs['CHARTTIME']).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0,24)]
labs['hour_bin'] = (labs['delta_h']//2).astype(int)
labs['lab_col'] = (
    'lab_' + labs['ITEMID'].astype(str) + '_bin' + labs['hour_bin'].astype(str)
)

lab_wide = labs.pivot_table(
    index=['subject_id','hadm_id','icustay_id'],
    columns='lab_col', values='VALUENUM', aggfunc='mean'
).reset_index()

structured = structured.merge(
    lab_wide,
    on=['subject_id','hadm_id','icustay_id'], how='left'
)


notes = pd.read_csv(
    'NOTEEVENTS.csv.gz', compression='gzip',
    usecols=['SUBJECT_ID','HADM_ID','CHARTTIME','TEXT'],
    parse_dates=['CHARTTIME']
).rename(columns={
    'SUBJECT_ID':'subject_id',
    'HADM_ID':'hadm_id',
    'CHARTTIME':'charttime'
})

notes = notes.merge(
    df[['subject_id','hadm_id','icustay_id','outtime']],
    on=['subject_id','hadm_id'], how='inner'
)

notes['delta_h'] = (notes['outtime'] - notes['charttime']).dt.total_seconds() / 3600
notes = notes[notes['delta_h'].between(0,24)]

agg_txt = notes.groupby(['subject_id','hadm_id','icustay_id'])['TEXT'] \
               .apply(lambda s: " ".join(s.astype(str))).reset_index()
agg_txt['clean'] = agg_txt['TEXT'].apply(preprocess_notes)
agg_txt['chunks'] = agg_txt['clean'].apply(chunk_text)

notes_expanded = pd.DataFrame(
    agg_txt['chunks'].tolist(),
    index=agg_txt.set_index(['subject_id','hadm_id','icustay_id']).index
).reset_index()


unstructured = structured.merge(
    notes_expanded,
    on=['subject_id','hadm_id','icustay_id'],
    how='left'
)

assert structured.shape[0] == unstructured.shape[0]

structured.to_csv('structured_mimiciii_readmit_labs.csv', index=False)
unstructured.to_csv('unstructured_mimiciii_readmit_notes.csv', index=False)

print("Structured shape:", structured.shape)
print("Unstructured shape:", unstructured.shape)
print("Positive readmissions in structured:", int(structured['readmission'].sum()))
print("Positive readmissions in unstructured:", int(unstructured['readmission'].sum()))
