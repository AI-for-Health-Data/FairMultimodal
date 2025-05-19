import os
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from transformers import AutoTokenizer

def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def categorize_age(age):
    if 18 <= age <= 29:
        return '18-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    elif 70 <= age <= 90:
        return '70-90'
    else:
        return 'Other'

def categorize_ethnicity(eth):
    eth = str(eth).strip().upper()
    return 'Hispanic' if 'HISPANIC' in eth else 'Non-Hispanic'

def categorize_race(race):
    r = str(race).upper()
    if 'WHITE' in r:
        return 'White'
    if 'BLACK' in r:
        return 'Black'
    if 'ASIAN' in r:
        return 'Asian'
    if 'HISPANIC' in r:
        return 'Hispanic'
    return 'Other'

def categorize_insurance(ins):
    i = str(ins).upper()
    if 'MEDICARE' in i:
        return 'Medicare'
    if 'MEDICAID' in i:
        return 'Medicaid'
    if 'PRIVATE' in i:
        return 'Private'
    return 'Other'

adm = pd.read_csv('ADMISSIONS.csv.gz', compression='gzip', usecols=[
    'SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DEATHTIME','ETHNICITY','INSURANCE'
])
adm[['ADMITTIME','DISCHTIME','DEATHTIME']] = adm[['ADMITTIME','DISCHTIME','DEATHTIME']].apply(pd.to_datetime)

pat = pd.read_csv('PATIENTS.csv.gz', compression='gzip', usecols=['SUBJECT_ID','DOB','GENDER'])
pat['DOB'] = pd.to_datetime(pat['DOB'], errors='coerce')

stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', usecols=[
    'SUBJECT_ID','HADM_ID','ICUSTAY_ID','INTIME','OUTTIME'
])
stays[['INTIME','OUTTIME']] = stays[['INTIME','OUTTIME']].apply(pd.to_datetime)

adm.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)
pat.rename(columns={'SUBJECT_ID':'subject_id'}, inplace=True)
stays.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)

df = stays.merge(adm, on=['subject_id','hadm_id']).merge(pat, on='subject_id')
df['age'] = df.apply(lambda r: calculate_age(r['DOB'], r['INTIME']), axis=1)
df = df[df['age'].between(18,90)]
# first ICU stay only
df = df.sort_values('INTIME').groupby('subject_id').first().reset_index()

df['age_bucket'] = df['age'].apply(categorize_age)
df['ethnicity']  = df['ETHNICITY'].apply(categorize_ethnicity)
df['race']       = df['ETHNICITY'].apply(categorize_race)
df['insurance']  = df['INSURANCE'].apply(categorize_insurance)
df['gender']     = df['GENDER'].str.lower().map(lambda x: 'female' if x=='f' else ('male' if x=='m' else 'Other'))
df['mortality'] = df['DEATHTIME'].notnull().astype(int)

diag = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip', usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE'])
diag['code'] = diag['ICD9_CODE'].str.split('.').str[0]
# map flags
diag['PE'] = diag['code'].isin(['4151','41511','41512','41513']).astype(int)
diag['PH'] = diag['code'].isin(['416','4160','4161','4162','4168','4169']).astype(int)
label_pe = diag.groupby(['SUBJECT_ID','HADM_ID'])['PE'].max().reset_index().rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'})
label_ph = diag.groupby(['SUBJECT_ID','HADM_ID'])['PH'].max().reset_index().rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'})
labels = df[['subject_id','hadm_id']].merge(label_pe, on=['subject_id','hadm_id'], how='left').merge(label_ph, on=['subject_id','hadm_id'], how='left').fillna(0)
df = df.merge(labels, on=['subject_id','hadm_id'], how='left')

feature_set = {
    'chartevents': [220051, 220052, 618, 220210, 224641, 220292, 535, 224695, 506, 220339, 448, 224687, 224685, 220293, 444, 224697, 220074, 224688, 223834, 50815, 225664, 220059, 683, 224684, 220060, 226253, 224161, 642, 225185, 226758, 226757, 226756, 220050, 211, 220045, 223761, 223835, 226873, 226871, 8364, 8555, 8368, 53, 646, 1529, 50809, 50931, 51478, 224639, 763, 224639, 226707],
    'labevents': [51221, 51480, 51265, 50811, 51222, 51249, 51248, 51250, 51279, 51277, 50902, 50868, 50912, 50809, 50931, 51478, 50960, 50893, 50970, 51237, 51274, 51275, 51375, 51427, 51446, 51116, 51244, 51355, 51379, 51120, 51254, 51256, 51367, 51387, 51442, 51112, 51146, 51345, 51347, 51368, 51419, 51444, 51114, 51200, 51474, 50820, 50831, 51094, 51491, 50802, 50804, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 490, 1165, 50902, 50819],
    'inputevents': [30008, 220864, 30005, 220970, 221385, 30023, 221456, 221668, 221749, 221794, 221828, 221906, 30027, 222011, 222056, 223258, 30126, 225154, 30297, 225166, 225168, 30144, 225799, 225823, 44367, 225828, 225943, 30065, 225944, 226089, 226364, 30056, 226452, 30059, 226453, 227522, 227523, 30044, 221289, 30051, 222315, 30043, 221662, 30124, 30118, 221744, 30131, 222168],
    'outputevents': [226573, 40054, 40085, 44890, 43703, 226580, 226588, 226589, 226599, 226626, 226633, 227510],
    'prescriptions': ['Docusate Sodium', 'Aspirin', 'Bisacodyl', 'Humulin-R Insulin', 'Metoprolol', 'Pantoprazole Sodium', 'Pantoprazole']
}

def aggregate_features(table_key, filename, val_col):
    if not os.path.exists(filename):
        return pd.DataFrame()
    d = pd.read_csv(filename, compression='gzip', low_memory=False)
    d.columns = d.columns.str.upper()
    d = d[d['ITEMID'].isin(feature_set[table_key])]
    d = d.merge(df[['subject_id','hadm_id','INTIME']].rename(columns={'subject_id':'SUBJECT_ID','hadm_id':'HADM_ID'}), on=['SUBJECT_ID','HADM_ID'], how='inner')
    d['CHARTTIME'] = pd.to_datetime(d['CHARTTIME'], errors='coerce')
    d['hrs'] = (d['CHARTTIME'] - d['INTIME']).dt.total_seconds() / 3600
    d = d[(d['hrs']>=0)&(d['hrs']<24)]
    d['bin'] = (d['hrs']//2).astype(int)
    pt = pd.pivot_table(d, index=['SUBJECT_ID','HADM_ID'], columns=['bin','ITEMID'], values=val_col, aggfunc='mean')
    cols = [f"{table_key}_t{int(b*2)}h_item{int(i)}" for b,i in pt.columns]
    pt.columns = cols
    return pt.reset_index().rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'})

chartevents = aggregate_features('chartevents','CHARTEVENTS.csv.gz','VALUENUM')
labevents  = aggregate_features('labevents', 'LABEVENTS.csv.gz',  'VALUENUM')

df_struct = df[['subject_id','hadm_id','age_bucket','ethnicity','race','insurance','gender','mortality','PE','PH']]
for feat in [chartevents, labevents]:
    if not feat.empty:
        df_struct = df_struct.merge(feat, on=['subject_id','hadm_id'], how='left')

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
notes = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip', usecols=['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT'])
notes.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)
notes = notes.merge(df[['subject_id','hadm_id','INTIME']], on=['subject_id','hadm_id'], how='inner')
notes['CHARTDATE']=pd.to_datetime(notes['CHARTDATE'], errors='coerce')
notes['hrs'] = (notes['CHARTDATE']-notes['INTIME']).dt.total_seconds()/3600
notes = notes[(notes['hrs']>=0)&(notes['hrs']<24)]
agg_text = notes.groupby(['subject_id','hadm_id'])['TEXT'].apply(lambda x: ' '.join(x)).reset_index()
def bert_chunks(text):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+512] for i in range(0, len(tokens), 512)]
    return [' '.join(chunk) for chunk in chunks]
chunks_df = agg_text['TEXT'].apply(lambda x: pd.Series(bert_chunks(x))).add_prefix('note_chunk_')
df_unstruct = pd.concat([agg_text[['subject_id','hadm_id']], chunks_df], axis=1)
# merge demographics/outcomes
df_unstruct = df_unstruct.merge(df_struct[['subject_id','age_bucket','ethnicity','race','insurance','gender','mortality','PE','PH']], on='subject_id', how='left')

common = set(df_struct['subject_id']).intersection(df_unstruct['subject_id'])
final_struct   = df_struct[df_struct['subject_id'].isin(common)].reset_index(drop=True)
final_unstruct = df_unstruct[df_unstruct['subject_id'].isin(common)].reset_index(drop=True)

final_struct.to_csv('final_structured.csv', index=False)
final_unstruct.to_csv('final_unstructured.csv', index=False)

print('Structured shape:', final_struct.shape)
print('Unstructured shape:', final_unstruct.shape)
for c in ['mortality','PE','PH']:
    print(f"{c} positives: {int(final_struct[c].sum())}")
print('Done.')
