import pandas as pd
import numpy as np
import re
from datetime import timedelta

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
    if 'WHITE' in r:    return 'White'
    if 'BLACK' in r:    return 'Black'
    if 'ASIAN' in r:    return 'Asian'
    if 'HISPANIC' in r: return 'Hispanic'
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
    df['readmit_30d'] = (
        (df['next_intime'] - df['outtime']) <= pd.Timedelta(days=30)
    ).fillna(False).astype(int)
    return df[['subject_id','hadm_id','icustay_id','readmit_30d']]

def filter_last_24h(df, time_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.merge(
        icu_stays[['subject_id','hadm_id','outtime']],
        on=['subject_id','hadm_id'], how='inner'
    )
    df['hours_to_end'] = (df['outtime'] - df[time_col]).dt.total_seconds() / 3600
    return df[(df['hours_to_end'] >= 0) & (df['hours_to_end'] <= 24)].copy()

def agg_lab(path, bin_size=2):
    le = pd.read_csv(path, compression='gzip', low_memory=False)
    le.columns = le.columns.str.lower()
    le = le[le['valuenum'].notnull()]
    le = filter_last_24h(le, 'charttime')
    le['hour_bin'] = (le['hours_to_end'] // bin_size).astype(int)

    agg = (
        le.groupby(['subject_id','hadm_id','hour_bin','itemid'])['valuenum']
          .mean()
          .unstack(fill_value=np.nan)
          .reset_index()
    )
    agg.drop(columns=['hour_bin'], inplace=True)

    cols = ['subject_id','hadm_id'] + [
        f'lab_t{int(c)}' for c in agg.columns
        if c not in ['subject_id','hadm_id']
    ]
    agg.columns = cols
    return agg

def agg_feature(table, items, files, bin_size=2, agg_func='mean'):
    if isinstance(files, list):
        fe = pd.concat([pd.read_csv(f, compression='gzip', low_memory=False) for f in files],
                       ignore_index=True)
    else:
        fe = pd.read_csv(files, compression='gzip', low_memory=False)

    fe.columns = fe.columns.str.lower()
    time_cols = [c for c in ['charttime','starttime','eventtime','storetime'] if c in fe.columns]
    if not time_cols:
        return pd.DataFrame(columns=['subject_id','hadm_id'])

    fe = filter_last_24h(fe, time_cols[0])
    if fe.empty:
        return pd.DataFrame(columns=['subject_id','hadm_id'])

    if 'itemid' in fe.columns:
        fe = fe[fe['itemid'].isin(items)]
    if table == 'prescriptions' and 'drug' in fe.columns:
        fe = fe[fe['drug'].str.lower().isin([i.lower() for i in items])]
    if fe.empty:
        return pd.DataFrame(columns=['subject_id','hadm_id'])

    num_cols = [c for c in ['valuenum','value','amount'] if c in fe.columns]
    val_col = num_cols[0]
    fe[val_col] = pd.to_numeric(fe[val_col], errors='coerce')
    fe['hour_bin'] = (fe['hours_to_end'] // bin_size).astype(int)

    grp = fe.groupby(['subject_id','hadm_id','hour_bin'])[val_col]
    out = grp.sum() if agg_func == 'sum' else grp.mean()
    agg_df = out.unstack(fill_value=np.nan).reset_index()
    if 'hour_bin' in agg_df.columns:
        agg_df.drop(columns=['hour_bin'], inplace=True)

    new_cols = ['subject_id','hadm_id'] + [
        f'{val_col}_t{int(c)}' for c in agg_df.columns
        if c not in ['subject_id','hadm_id']
    ]
    agg_df.columns = new_cols
    return agg_df

admissions = (
    pd.read_csv('ADMISSIONS.csv.gz', compression='gzip', low_memory=False)
      .rename(columns=str.lower)
      [['subject_id','hadm_id','ethnicity','insurance']]
)

patients = (
    pd.read_csv('PATIENTS.csv.gz', compression='gzip', low_memory=False)
      .rename(columns=str.lower)
      [['subject_id','gender','dob']]
)
patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')

icu_stays = (
    pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', low_memory=False)
      .rename(columns=str.lower)
      [['subject_id','hadm_id','icustay_id','intime','outtime']]
)
icu_stays['intime']  = pd.to_datetime(icu_stays['intime'],  errors='coerce')
icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'], errors='coerce')

admissions['ethnicity_flag'] = admissions['ethnicity'].apply(categorize_ethnicity)
admissions['race']            = admissions['ethnicity'].apply(categorize_race)
admissions['insurance_cat']   = admissions['insurance'].apply(categorize_insurance)

df = (
    icu_stays
      .merge(admissions[['subject_id','hadm_id','race','ethnicity_flag','insurance_cat']],
             on=['subject_id','hadm_id'], how='left')
      .merge(patients[['subject_id','gender','dob']], on='subject_id', how='left')
)

df['age']        = df.apply(lambda r: calculate_age(r['dob'], r['intime']), axis=1)
df = df[(df['age'] >= 18) & (df['age'] <= 90)]
df['age_bucket'] = df['age'].apply(categorize_age)
df['gender']     = df['gender'].str.lower().apply(
                      lambda x: 'male' if x.startswith('m')
                                else ('female' if x.startswith('f') else x)
                  )

flags = get_readmission_flag_index(icu_stays)
df = df.merge(
    flags[['subject_id','icustay_id','readmit_30d']],
    on=['subject_id','icustay_id'], how='left'
)
df['readmit_30d'] = df['readmit_30d'].fillna(0).astype(int)

df = df.sort_values(['subject_id','intime']).groupby('subject_id', as_index=False).first()

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
    'prescriptions':'PRESCRIPTIONS.csv.gz'
}

lab_feats = agg_lab(input_files['labevents'])
df = df.merge(lab_feats, on=['subject_id','hadm_id'], how='left')

time_cols = [c for c in df.columns if re.match(r'.*_t\d+$', c)]
df['hours_data_present'] = df[time_cols].notnull().sum(axis=1) * 2
df = df[df['hours_data_present'] >= 30]

df.to_csv('cohort_structured_first24h_temp.csv', index=False)

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

notes_last24 = filter_last_24h(notes, 'chartdate')
notes_last24['text'] = notes_last24['text'].fillna('')

notes_agg = (
    notes_last24
      .groupby(['subject_id','hadm_id'], as_index=False)['text']
      .apply(lambda txt: ' '.join(txt))
)
notes_agg['clean'] = notes_agg['text'].apply(preprocess_text)

chunks = notes_agg['clean'].apply(lambda x: pd.Series(chunk_text(x)))
notes_final = pd.concat([notes_agg[['subject_id','hadm_id']], chunks], axis=1)

notes_final = notes_final.merge(
    df[['subject_id','hadm_id','readmit_30d']],
    on=['subject_id','hadm_id'], how='left'
)

common_pairs = (
    pd.merge(
        df[['subject_id','hadm_id']],
        notes_final[['subject_id','hadm_id']],
        on=['subject_id','hadm_id']
    )
    .drop_duplicates()
)

df_common   = df.merge(common_pairs,   on=['subject_id','hadm_id'], how='inner')
notes_common = notes_final.merge(common_pairs, on=['subject_id','hadm_id'], how='inner')

assert df_common.shape[0] == notes_common.shape[0], (
    f"Structured has {df_common.shape[0]} rows, unstructured has {notes_common.shape[0]}"
)

df_common.to_csv('cohort_structured_common_subjects.csv', index=False)
notes_common.to_csv('cohort_unstructured_common_subjects.csv', index=False)

print(f"Common stays: {df_common.shape[0]}")
print(f"Structured readmissions:   {df_common['readmit_30d'].sum()}")
print(f"Unstructured readmissions: {notes_common['readmit_30d'].sum()}")

print("Structured common-subjects shape:", df_common.shape)
print("Unstructured common-subjects shape:", notes_common.shape)

df_common.to_csv('cohort_structured_common_subjects.csv', index=False)
notes_common.to_csv('cohort_unstructured_common_subjects.csv', index=False)
