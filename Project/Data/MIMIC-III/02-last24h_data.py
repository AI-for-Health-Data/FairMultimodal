import pandas as pd
import numpy as np
import re
from datetime import timedelta

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

def get_readmission_flag_index(df_stays):
    df = df_stays.sort_values(['subject_id','intime']).copy()
    df['next_intime'] = df.groupby('subject_id')['intime'].shift(-1)
    df['readmit_30d'] = ((df['next_intime'] - df['outtime']) <= pd.Timedelta(days=30)).fillna(False).astype(int)
    return df[['subject_id','hadm_id','icustay_id','readmit_30d']]

feature_set_C_items = {
    'chartevents': [220051,220052,618,220210,224641,220292,535,224695,506,220339,448,224687,224685,220293,444,224697,220074,224688,223834,50815,225664,220059,683,224684,220060,226253,224161,642,225185,226758,226757,226756,220050,211,220045,223761,223835,226873,226871,8364,8555,8368,53,646,1529,50809,50931,51478,224639,763,224639,226707],
    'labevents': [51221,51480,51265,50811,51222,51249,51248,51250,51279,51277,50902,50868,50912,50809,50931,51478,50960,50893,50970,51237,51274,51275,51375,51427,51446,51116,51244,51355,51379,51120,51254,51256,51367,51387,51442,51112,51146,51345,51347,51368,51419,51444,51114,51200,51474,50820,50831,51094,51491,50802,50804,50818,51498,50813,50861,50878,50863,50862,490,1165,50902,50819],
    'inputevents': [30008,220864,30005,220970,221385,30023,221456,221668,221749,221794,221828,221906,30027,222011,222056,223258,30126,225154,30297,225166,225168,30144,225799,225823,44367,225828,225943,30065,225944,226089,226364,30056,226452,30059,226453,227522,227523,30044,221289,30051,222315,30043,221662,30124,30118,221744,30131,222168],
    'outputevents': [226573,40054,40085,44890,43703,226580,226588,226589,226599,226626,226633,227510],
    'prescriptions': ['Docusate Sodium','Aspirin','Bisacodyl','Humulin-R Insulin','Metoprolol','Pantoprazole Sodium','Pantoprazole']
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
    df_all = df_all.merge(cohort_keys.rename(columns={intime_col:'intime'}), on=['subject_id','hadm_id'], how='inner')

    # Select only target itemids/drugs
    if drug_mode:
        df_all = df_all[df_all['drug'].str.lower().isin([d.lower() for d in items])]
    elif 'itemid' in df_all.columns:
        df_all = df_all[df_all['itemid'].isin(items)]

    event_col = None
    for col in ['charttime','starttime','eventtime','storetime']:
        if col in df_all.columns:
            event_col = col
            break
    if event_col is None: return pd.DataFrame(columns=['subject_id','hadm_id'])
    df_all[event_col] = pd.to_datetime(df_all[event_col], errors='coerce')
    df_all['hrs'] = (df_all[event_col] - df_all['intime']).dt.total_seconds()/3600
    df_all = df_all[(df_all['hrs'] >= 0) & (df_all['hrs'] < 24)]
    df_all['bin'] = (df_all['hrs']//bin_size).astype(int)

    value_cols = [c for c in ['valuenum','value','amount','rate'] if c in df_all.columns]
    if not value_cols: return pd.DataFrame(columns=['subject_id','hadm_id'])
    val_col = value_cols[0]
    df_all[val_col] = pd.to_numeric(df_all[val_col], errors='coerce')

    idx_cols = ['subject_id','hadm_id','bin','drug'] if drug_mode else ['subject_id','hadm_id','bin','itemid']
    agg = df_all.groupby(idx_cols)[val_col].mean().reset_index()
    agg['colname'] = (
        agg.apply(lambda r: f"{table}_t{r.bin*bin_size}h_drug_{r.drug.lower().replace(' ','_')}" if drug_mode
        else f"{table}_t{r.bin*bin_size}h_item{r.itemid}", axis=1)
    )
    feat = agg.pivot(index=['subject_id','hadm_id'], columns='colname', values=val_col).reset_index()
    return feat

admissions = pd.read_csv('ADMISSIONS.csv.gz', compression='gzip', low_memory=False).rename(columns=str.lower)
patients = pd.read_csv('PATIENTS.csv.gz', compression='gzip', low_memory=False).rename(columns=str.lower)
patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', low_memory=False).rename(columns=str.lower)
icu_stays['intime']  = pd.to_datetime(icu_stays['intime'],  errors='coerce')
icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'], errors='coerce')

admissions['ethnicity_flag'] = admissions['ethnicity'].apply(categorize_ethnicity)
admissions['race']           = admissions['ethnicity'].apply(categorize_race)
admissions['insurance_cat']  = admissions['insurance'].apply(categorize_insurance)
df = (
    icu_stays
      .merge(admissions[['subject_id','hadm_id','race','ethnicity_flag','insurance_cat']],
             on=['subject_id','hadm_id'], how='left')
      .merge(patients[['subject_id','gender','dob']], on='subject_id', how='left')
)
df['age']        = df.apply(lambda r: calculate_age(r['dob'], r['intime']), axis=1)
df = df[(df['age'] >= 18) & (df['age'] <= 90)]
df['age_bucket'] = df['age'].apply(categorize_age)
df['gender']     = df['gender'].str.lower().apply(lambda x: 'male' if x.startswith('m') else ('female' if x.startswith('f') else x))

flags = get_readmission_flag_index(icu_stays)
df = df.merge(flags[['subject_id','icustay_id','readmit_30d']], on=['subject_id','icustay_id'], how='left')
df['readmit_30d'] = df['readmit_30d'].fillna(0).astype(int)
df = df.sort_values(['subject_id','intime']).groupby('subject_id', as_index=False).first()

all_feats = df[['subject_id','hadm_id']].copy()
for tab in feature_set_C_items:
    print(f"Processing {tab}...")
    items = feature_set_C_items[tab]
    file = input_files[tab]
    drug_mode = (tab == 'prescriptions')
    tab_feats = aggregate_event_features(tab, items, file, df, 'intime', bin_size=2, drug_mode=drug_mode)
    all_feats = all_feats.merge(tab_feats, on=['subject_id','hadm_id'], how='left')

df_struct = df[['subject_id','hadm_id','age_bucket','ethnicity_flag','race','insurance_cat','gender','readmit_30d']].merge(all_feats, on=['subject_id','hadm_id'], how='left')

def preprocess_text(x):
    y = re.sub(r'\[(.*?)\]', '', str(x))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def chunk_text(text, size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i+size]) for i in range(0, len(tokens), size)]

notes = (
    pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip', low_memory=False)
      .rename(columns=str.lower)
      [['subject_id','hadm_id','chartdate','text']]
)
notes = notes.merge(df[['subject_id', 'hadm_id']], on=['subject_id', 'hadm_id'], how='inner')
notes['chartdate'] = pd.to_datetime(notes['chartdate'], errors='coerce')
notes = notes.merge(df[['subject_id','hadm_id','outtime']], on=['subject_id','hadm_id'], how='left')
notes['hours_to_end'] = (notes['outtime'] - notes['chartdate']).dt.total_seconds() / 3600
notes = notes[(notes['hours_to_end'] >= 0) & (notes['hours_to_end'] <= 24)]
notes['text'] = notes['text'].fillna('')

notes_agg = notes.groupby(['subject_id','hadm_id'], as_index=False)['text'].apply(lambda txt: ' '.join(txt))
notes_agg['clean'] = notes_agg['text'].apply(preprocess_text)
chunked = notes_agg['clean'].apply(lambda x: pd.Series(chunk_text(x)))
chunked.columns = [f'note_{i}' for i in chunked.columns]
notes_final = pd.concat([notes_agg[['subject_id','hadm_id']], chunked], axis=1)

merge_cols = [
    'subject_id','hadm_id','readmit_30d','age_bucket','gender','race','ethnicity_flag','insurance_cat'
]
notes_final = notes_final.merge(
    df[merge_cols],
    on=['subject_id','hadm_id'], how='left'
)
meta_cols = ['subject_id','hadm_id','readmit_30d','age_bucket','gender','race','ethnicity_flag','insurance_cat']
note_cols = [c for c in notes_final.columns if c.startswith('note_')]
notes_final = notes_final[meta_cols + note_cols]

common_pairs = pd.merge(df_struct[['subject_id','hadm_id']], notes_final[['subject_id','hadm_id']], on=['subject_id','hadm_id']).drop_duplicates()
df_common   = df_struct.merge(common_pairs,   on=['subject_id','hadm_id'], how='inner')
notes_common = notes_final.merge(common_pairs, on=['subject_id','hadm_id'], how='inner')

df_common.to_csv('cohort_structured_common_subjects.csv', index=False)
notes_common.to_csv('cohort_unstructured_common_subjects.csv', index=False)

print(f"All structured: {df_struct.shape}, positives: {df_struct['readmit_30d'].sum()}")
print(f"All unstructured: {notes_final.shape}, positives: {notes_final['readmit_30d'].sum()}")
print(f"Common structured: {df_common.shape}, positives: {df_common['readmit_30d'].sum()}")
print(f"Common unstructured: {notes_common.shape}, positives: {notes_common['readmit_30d'].sum()}")

for col in ['age_bucket', 'ethnicity_flag', 'race', 'insurance_cat', 'gender']:
    print(f"\nStructured {col} value counts:")
    print(df_common[col].value_counts())
    print(f"Unstructured {col} value counts:")
    print(notes_common[col].value_counts())
