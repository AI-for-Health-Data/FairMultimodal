import pandas as pd
import numpy as np
import re

feature_set_C_labs = {
    'chartevents':    [220051,220052,618,220210,224641,220292,535,224695,506,220339,
                       448,224687,224685,220293,444,224697,220074,224688,223834,50815,
                       225664,220059,683,224684,220060,226253,224161,642,225185,226758,
                       226757,226756,220050,211,220045,223761,223835,226873,226871,8364,
                       8555,8368,53,646,1529,50809,50931,51478,224639,763,226707],
    'labevents':      [51221,51480,51265,50811,51222,51249,51248,51250,51279,51277,
                       50902,50868,50912,50809,50931,51478,50960,50893,50970,51237,
                       51274,51275,51375,51427,51446,51116,51244,51355,51379,51120,
                       51254,51256,51367,51387,51442,51112,51146,51345,51347,51368,
                       51419,51444,51114,51200,51474,50820,50831,51094,51491,50802,
                       50804,50818,51498,50813,50861,50878,50863,50862,490,1165,50819],
    'inputevents':    [30008,220864,30005,220970,221385,30023,221456,221668,221749,
                       221794,221828,221906,30027,222011,222056,223258,30126,225154,
                       30297,225166,225168,30144,225799,225823,44367,225828,225943,
                       30065,225944,226089,226364,30056,226452,30059,226453,227522,
                       227523,30044,221289,30051,222315,30043,221662,30124,30118,
                       221744,30131,222168],
    'outputevents':   [226573,40054,40085,44890,43703,226580,226588,226589,226599,
                       226626,226633,227510],
    'prescriptions':  ['Docusate Sodium','Aspirin','Bisacodyl','Humulin-R Insulin',
                       'Metoprolol','Pantoprazole Sodium','Pantoprazole']
}

input_files = {
    'chartevents':    'CHARTEVENTS.csv.gz',
    'labevents':      'LABEVENTS.csv.gz',
    'inputevents':   ['inputevents_cv.csv.gz','inputevents_mv.csv.gz'],
    'outputevents':   'OUTPUTEVENTS.csv.gz',
    'prescriptions':  'PRESCRIPTIONS.csv.gz'
}

def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def categorize_ethnicity(e):
    e = str(e).upper()
    if 'HISPANIC' in e:     return 'Hispanic'
    if 'NON-HISPANIC' in e: return 'Non-Hispanic'
    return 'Unknown'

def categorize_race_from_eth(e):
    r = str(e).upper()
    if 'WHITE' in r:   return 'White'
    if 'BLACK' in r:   return 'Black'
    if 'ASIAN' in r:   return 'Asian'
    if 'NATIVE' in r:  return 'Native'
    if 'OTHER' in r:   return 'Other'
    return 'Unknown'

def categorize_insurance(i):
    i = str(i).upper()
    if 'MEDICARE' in i: return 'Medicare'
    if 'MEDICAID' in i: return 'Medicaid'
    if 'PRIVATE' in i:  return 'Private'
    if 'SELF PAY' in i: return 'Self Pay'
    return 'Government'

def calculate_short_term_mortality(adm_df):
    adm_df['deathtime'] = pd.to_datetime(adm_df['deathtime'], errors='coerce')
    return adm_df['deathtime'].notnull().astype(int)

def preprocess_notes(text):
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def chunk_text(text, size=512):
    tokens = text.split()
    return [' '.join(tokens[i:i+size]) for i in range(0, len(tokens), size)]

icustays = pd.read_csv(
    'ICUSTAYS.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID','HADM_ID','INTIME','OUTTIME'],
    parse_dates=['INTIME','OUTTIME']
)
icustays.columns = icustays.columns.str.lower()

icustays['los_hours'] = (icustays['outtime'] - icustays['intime']).dt.total_seconds() / 3600
icustays = icustays[icustays['los_hours'] >= 30]

admissions = pd.read_csv(
    'ADMISSIONS.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID','HADM_ID','DEATHTIME','ETHNICITY','INSURANCE']
)
admissions.columns = admissions.columns.str.lower()
admissions['short_term_mortality'] = calculate_short_term_mortality(admissions)
admissions['ethnicity_cat'] = admissions['ethnicity'].apply(categorize_ethnicity)
admissions['race_cat']      = admissions['ethnicity'].apply(categorize_race_from_eth)
admissions['insurance_cat'] = admissions['insurance'].apply(categorize_insurance)

patients = pd.read_csv(
    'PATIENTS.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID','GENDER','DOB'], parse_dates=['DOB']
)
patients.columns = patients.columns.str.lower()

diag = pd.read_csv(
    'DIAGNOSES_ICD.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE']
)
diag.columns = diag.columns.str.lower()
diag['icd9'] = diag['icd9_code'].str.replace('.', '', regex=False).fillna('')

pe_flag = (diag[diag['icd9'].str.startswith('415')]
           .groupby(['subject_id','hadm_id']).size()
           .reset_index(name='count'))
pe_flag['pe'] = 1
pe_flag = pe_flag[['subject_id','hadm_id','pe']]

ph_flag = (diag[diag['icd9'].str.startswith('416')]
           .groupby(['subject_id','hadm_id']).size()
           .reset_index(name='count'))
ph_flag['ph'] = 1
ph_flag = ph_flag[['subject_id','hadm_id','ph']]

df = (
    icustays
    .merge(admissions, on=['subject_id','hadm_id'], how='left')
    .merge(patients,   on='subject_id',        how='left')
    .merge(pe_flag,    on=['subject_id','hadm_id'], how='left')
    .merge(ph_flag,    on=['subject_id','hadm_id'], how='left')
)
df['pe'] = df['pe'].fillna(0).astype(int)
df['ph'] = df['ph'].fillna(0).astype(int)

df['age']    = df.apply(lambda r: calculate_age(r['dob'], r['intime']), axis=1)
df['gender'] = df['gender'].str.lower().map({'m':'male','f':'female'}).fillna('other')
df = df[(df['age'] >= 18) & (df['age'] <= 90)].copy()

df = (
    df
    .sort_values('intime')
    .groupby('subject_id', as_index=False)
    .first()
)

structured = df[[
    'subject_id','hadm_id','age','gender',
    'race_cat','ethnicity_cat','insurance_cat',
    'short_term_mortality','pe','ph'
]]

patients_kept = set(structured['subject_id'])

def aggregate_first24h_labs(lab_path, stays_df):
    labs = pd.read_csv(lab_path, compression='gzip', low_memory=False,
                       usecols=['SUBJECT_ID','HADM_ID','CHARTTIME','ITEMID','VALUENUM'])
    labs.columns = labs.columns.str.lower()
    labs = labs[labs['itemid'].isin(feature_set_C_labs)].dropna(subset=['valuenum'])
    labs['charttime'] = pd.to_datetime(labs['charttime'], errors='coerce')
    stays = stays_df[['subject_id','hadm_id','intime']].copy()
    labs = labs.merge(stays, on=['subject_id','hadm_id'], how='inner')
    labs['hrs'] = (labs['charttime'] - labs['intime']).dt.total_seconds()/3600
    labs = labs[labs['hrs'].between(0,24)]
    labs['bin'] = (labs['hrs']//2).astype(int)
    agg = (
        labs
        .groupby(['subject_id','hadm_id','bin','itemid'])['valuenum']
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )

    new_cols = []
    for c in agg.columns:
        if c in ('subject_id','hadm_id'):
            new_cols.append(c)
        elif isinstance(c, tuple) and len(c) == 2:
            b, item = c
            new_cols.append(f"lab_t{b}_{item}")
        else:
            new_cols.append(str(c))
    agg.columns = new_cols

def clean_and_chunk_notes(notes_path, stays_df):
    notes = pd.read_csv(notes_path, compression='gzip', low_memory=False,
                        usecols=['SUBJECT_ID','HADM_ID','CHARTTIME','TEXT'])
    notes.columns = notes.columns.str.lower()
    notes['charttime'] = pd.to_datetime(notes['charttime'], errors='coerce')
    stays = stays_df[['subject_id','hadm_id','intime','outtime']].copy()
    merged = notes.merge(stays, on=['subject_id','hadm_id'], how='inner')
    window = merged[(merged['charttime']>=merged['intime']) & (merged['charttime']<=merged['outtime'])]
    agg = (window.groupby(['subject_id','hadm_id'])['text']
               .agg(lambda ts: " ".join(ts.dropna())).reset_index())
    agg['clean'] = agg['text'].apply(preprocess_notes)
    rows = []
    for _, r in agg.iterrows():
        data = {'subject_id': r.subject_id, 'hadm_id': r.hadm_id}
        for idx, chunk in enumerate(chunk_text(r.clean), start=1):
            data[f'note_chunk_{idx}'] = chunk
        rows.append(data)
    return pd.DataFrame(rows)

notes_df = clean_and_chunk_notes('NOTEEVENTS.csv.gz', df)
unstructured = notes_df.merge(
    structured, on=['subject_id','hadm_id'], how='right'
)

structured.to_csv('final_structured_one_row_per_patient.csv', index=False)
unstructured.to_csv('final_unstructured_one_row_per_patient.csv', index=False)

print("Structured dataset:", structured.shape)
print("Unstructured dataset:", unstructured.shape)
print("Patients retained:", len(patients_kept))
print("Short-term mortality count:", structured['short_term_mortality'].sum())
print("PE count:", structured['pe'].sum())
print("PH count:", structured['ph'].sum())
