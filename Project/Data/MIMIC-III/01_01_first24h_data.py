import os
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from transformers import AutoTokenizer

def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def categorize_age(age):
    if 18 <= age <= 29: return '18-29'
    elif 30 <= age <= 49: return '30-49'
    elif 50 <= age <= 69: return '50-69'
    elif 70 <= age <= 90: return '70-90'
    else: return 'Other'

def categorize_ethnicity(eth):
    eth = str(eth).upper()
    return 'Hispanic' if 'HISPANIC' in eth or 'LATINO' in eth else 'Non-Hispanic'

def categorize_race(eth):
    eth = str(eth).upper()
    if 'WHITE' in eth: return 'White'
    if 'BLACK' in eth or 'CARIBBEAN' in eth or 'HAITIAN' in eth or 'CAPE VERDEAN' in eth: return 'Black'
    if 'ASIAN' in eth: return 'Asian'
    if 'HISPANIC' in eth or 'LATINO' in eth or 'SOUTH AMERICAN' in eth or 'PORTUGUESE' in eth or 'BRAZILIAN' in eth: return 'Hispanic'
    if 'AMERICAN INDIAN' in eth or 'NATIVE HAWAIIAN' in eth or 'PACIFIC ISLANDER' in eth: return 'Other'
    if 'MIDDLE EASTERN' in eth or 'MULTI RACE' in eth: return 'Other'
    if eth in {'OTHER', 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN/NOT SPECIFIED'}: return 'Other'
    return 'Other'

def categorize_insurance(ins):
    i = str(ins).upper()
    if 'MEDICARE' in i: return 'Medicare'
    if 'MEDICAID' in i: return 'Medicaid'
    if 'PRIVATE' in i:  return 'Private'
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
df = df.sort_values('INTIME').groupby('subject_id').first().reset_index()

df['age_bucket'] = df['age'].apply(categorize_age)
df['ethnicity']  = df['ETHNICITY'].apply(categorize_ethnicity)
df['race']       = df['ETHNICITY'].apply(categorize_race)
df['insurance']  = df['INSURANCE'].apply(categorize_insurance)
df['gender']     = df['GENDER'].str.lower().map(lambda x: 'female' if x=='f' else ('male' if x=='m' else 'Other'))
df['mortality']  = df['DEATHTIME'].notnull().astype(int)

diag = pd.read_csv('DIAGNOSES_ICD.csv.gz', compression='gzip', usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE'])
diag['code'] = diag['ICD9_CODE'].str.split('.').str[0]
diag['PE'] = diag['code'].isin(['4151','41511','41512','41513']).astype(int)
diag['PH'] = diag['code'].isin(['416','4160','4161','4162','4168','4169']).astype(int)
label_pe = diag.groupby(['SUBJECT_ID','HADM_ID'])['PE'].max().reset_index()
label_ph = diag.groupby(['SUBJECT_ID','HADM_ID'])['PH'].max().reset_index()
label_pe.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)
label_ph.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)

labels = df[['subject_id','hadm_id']].merge(label_pe, on=['subject_id','hadm_id'], how='left').merge(label_ph, on=['subject_id','hadm_id'], how='left').fillna(0)
labels['PE'] = labels['PE'].astype(int)
labels['PH'] = labels['PH'].astype(int)
df = df.merge(labels, on=['subject_id','hadm_id'], how='left')

feature_set_C_items = {
    'chartevents': [220051, 220052, 618, 220210, 224641, 220292, 535, 224695, 506, 220339, 448, 224687, 224685, 220293, 444, 224697, 220074, 224688, 223834, 50815, 225664, 220059, 683, 224684, 220060, 226253, 224161, 642, 225185, 226758, 226757, 226756, 220050, 211, 220045, 223761, 223835, 226873, 226871, 8364, 8555, 8368, 53, 646, 1529, 50809, 50931, 51478, 224639, 763, 224639, 226707],
    'labevents': [51221, 51480, 51265, 50811, 51222, 51249, 51248, 51250, 51279, 51277, 50902, 50868, 50912, 50809, 50931, 51478, 50960, 50893, 50970, 51237, 51274, 51275, 51375, 51427, 51446, 51116, 51244, 51355, 51379, 51120, 51254, 51256, 51367, 51387, 51442, 51112, 51146, 51345, 51347, 51368, 51419, 51444, 51114, 51200, 51474, 50820, 50831, 51094, 51491, 50802, 50804, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 490, 1165, 50902, 50819],
    'inputevents': [30008, 220864, 30005, 220970, 221385, 30023, 221456, 221668, 221749, 221794, 221828, 221906, 30027, 222011, 222056, 223258, 30126, 225154, 30297, 225166, 225168, 30144, 225799, 225823, 44367, 225828, 225943, 30065, 225944, 226089, 226364, 30056, 226452, 30059, 226453, 227522, 227523, 30044, 221289, 30051, 222315, 30043, 221662, 30124, 30118, 221744, 30131, 222168],
    'outputevents': [226573, 40054, 40085, 44890, 43703, 226580, 226588, 226589, 226599, 226626, 226633, 227510],
    'prescriptions': ['Docusate Sodium', 'Aspirin', 'Bisacodyl', 'Humulin-R Insulin', 'Metoprolol', 'Pantoprazole Sodium', 'Pantoprazole']
}

input_files = {
    'chartevents':   'CHARTEVENTS.csv.gz',
    'labevents':     'LABEVENTS.csv.gz',
    'inputevents':   ['inputevents_cv.csv.gz','inputevents_mv.csv.gz'],
    'outputevents':  'OUTPUTEVENTS.csv.gz',
    'prescriptions': 'PRESCRIPTIONS.csv.gz'
}

def aggregate_event_features(table, items, files, cohort_df, intime_col, bin_size=2, drug_mode=False):
    """Aggregate features for specified items/drugs over 2h bins in 0-24h after ICU admission."""
    if isinstance(files, list):
        df_all = pd.concat([pd.read_csv(f, compression='gzip', low_memory=False) for f in files], ignore_index=True)
    else:
        df_all = pd.read_csv(files, compression='gzip', low_memory=False)

    df_all.columns = df_all.columns.str.lower()
    cohort_keys = cohort_df[['subject_id','hadm_id',intime_col]]

    # Merge for only relevant admissions/intimes
    df_all = df_all.merge(
        cohort_keys.rename(columns={intime_col:'intime'}),
        on=['subject_id','hadm_id'], how='inner'
    )

    if drug_mode:
        df_all = df_all[df_all['drug'].str.lower().isin([d.lower() for d in items])]
    elif 'itemid' in df_all.columns:
        df_all = df_all[df_all['itemid'].isin(items)]

    event_col = None
    for col in ['charttime','starttime','eventtime','storetime']:
        if col in df_all.columns:
            event_col = col
            break
    if event_col is None:
        return pd.DataFrame(columns=['subject_id','hadm_id'])
    df_all[event_col] = pd.to_datetime(df_all[event_col], errors='coerce')

    df_all['hrs'] = (df_all[event_col] - df_all['intime']).dt.total_seconds()/3600
    df_all = df_all[(df_all['hrs'] >= 0) & (df_all['hrs'] < 24)]
    df_all['bin'] = (df_all['hrs']//bin_size).astype(int)

    value_cols = [c for c in ['valuenum','value','amount','rate'] if c in df_all.columns]
    if not value_cols:
        return pd.DataFrame(columns=['subject_id','hadm_id'])
    val_col = value_cols[0]
    df_all[val_col] = pd.to_numeric(df_all[val_col], errors='coerce')

    if drug_mode:
        idx_cols = ['subject_id','hadm_id','bin','drug']
    else:
        idx_cols = ['subject_id','hadm_id','bin','itemid']
    agg = df_all.groupby(idx_cols)[val_col].mean().reset_index()

    if drug_mode:
        agg['colname'] = agg.apply(lambda r: f'{table}_t{r.bin*bin_size}h_drug_{r.drug.lower().replace(" ","_")}', axis=1)
    else:
        agg['colname'] = agg.apply(lambda r: f'{table}_t{r.bin*bin_size}h_item{r.itemid}', axis=1)
    feat = agg.pivot(index=['subject_id','hadm_id'], columns='colname', values=val_col)
    feat = feat.reset_index()
    return feat

all_feats = df[['subject_id','hadm_id']].copy()
for tab in ['chartevents','labevents','inputevents','outputevents','prescriptions']:
    print(f"Processing {tab}...")
    items = feature_set_C_items[tab]
    file = input_files[tab]
    drug_mode = (tab == 'prescriptions')
    tab_feats = aggregate_event_features(tab, items, file, df, 'INTIME', bin_size=2, drug_mode=drug_mode)
    all_feats = all_feats.merge(tab_feats, on=['subject_id','hadm_id'], how='left')

all_feats.to_csv('structured_features_first24h_2h_bins.csv', index=False)

df_struct = df[['subject_id','hadm_id','age_bucket','ethnicity','race','insurance','gender','mortality','PE','PH']]
df_struct = df_struct.merge(all_feats, on=['subject_id','hadm_id'], how='left')

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
notes = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip', usecols=['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT'])
notes.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id'}, inplace=True)
notes = notes.merge(df[['subject_id','hadm_id','INTIME']], on=['subject_id','hadm_id'], how='inner')
notes['CHARTDATE'] = pd.to_datetime(notes['CHARTDATE'], errors='coerce')
notes['hrs'] = (notes['CHARTDATE']-notes['INTIME']).dt.total_seconds()/3600
notes = notes[(notes['hrs']>=0)&(notes['hrs']<24)]
agg_text = notes.groupby(['subject_id','hadm_id'])['TEXT'].apply(lambda x: ' '.join(x)).reset_index()

def bert_chunks(text):
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+512] for i in range(0, len(tokens), 512)]
    return [' '.join(chunk) for chunk in chunks]

chunks_df = agg_text['TEXT'].apply(lambda x: pd.Series(bert_chunks(x))).add_prefix('note_chunk_')
df_unstruct = pd.concat([agg_text[['subject_id','hadm_id']], chunks_df], axis=1)
df_unstruct = df_unstruct.merge(
    df_struct[['subject_id','hadm_id','age_bucket','ethnicity','race','insurance','gender','mortality','PE','PH']],
    on=['subject_id','hadm_id'], how='left'
)

common = set(zip(df_struct['subject_id'], df_struct['hadm_id'])).intersection(
    set(zip(df_unstruct['subject_id'], df_unstruct['hadm_id']))
)
common = pd.DataFrame(list(common), columns=['subject_id','hadm_id'])

final_struct   = df_struct.merge(common, on=['subject_id','hadm_id'], how='inner').reset_index(drop=True)
final_unstruct = df_unstruct.merge(common, on=['subject_id','hadm_id'], how='inner').reset_index(drop=True)

print(">>> Structured columns:", final_struct.columns.tolist())
print(">>> Unstructured columns:", final_unstruct.columns.tolist())

final_struct.to_csv('final_structured.csv', index=False)
final_unstruct.to_csv('final_unstructured.csv', index=False)

print("Structured data:", final_struct.shape)
print("Unstructured data:", final_unstruct.shape)

for df_name, df_ in [("Structured", final_struct), ("Unstructured", final_unstruct)]:
    print(f"\n{df_name} data positive counts:")
    print("  mortality:",    df_['mortality'].sum())
    print("  PE:",           df_['PE'].sum())
    print("  PH:",           df_['PH'].sum())
