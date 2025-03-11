import pandas as pd
import numpy as np
import re
from datetime import timedelta

# Calculate age at ICU admission given date of birth and admission time.
def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

# Categorize age into predefined bins.
def categorize_age(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    elif 70 <= age <= 89:
        return '70-89'
    else:
        return 'Other'

# Simplify ethnicity descriptions.
def categorize_ethnicity(ethnicity):
    eth = str(ethnicity).upper()
    if eth in ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN']:
        return 'White'
    elif eth in ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN', 'BLACK/AFRICAN', 'CARIBBEAN ISLAND']:
        return 'Black'
    elif eth in ['HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - MEXICAN']:
        return 'Hispanic'
    elif eth in ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - INDIAN']:
        return 'Asian'
    else:
        return 'Other'

# Categorize insurance based on keyword matching.
def categorize_insurance(insurance):
    ins = str(insurance).upper()
    if 'MEDICARE' in ins:
        return 'Medicare'
    elif 'PRIVATE' in ins:
        return 'Private'
    elif 'MEDICAID' in ins:
        return 'Medicaid'
    elif 'SELF PAY' in ins:
        return 'Self Pay'
    else:
        return 'Government'

# Calculate short-term mortality using the presence of DEATHTIME.
def calculate_short_term_mortality(df):
    df['short_term_mortality'] = df['DEATHTIME'].notnull().astype(int)
    return df

# Determine mechanical ventilation based on signals from CHARTEVENTS and PROCEDUREEVENTS_MV.
def calculate_mechanical_ventilation():
    # Load relevant CHARTEVENTS
    chartevents = pd.read_csv(
        'CHARTEVENTS.csv.gz', compression='gzip', low_memory=False,
        usecols=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'ERROR']
    )
    chartevents.columns = chartevents.columns.str.lower()
    chartevents = chartevents[chartevents['value'].notnull()]
    chartevents = chartevents[(chartevents['error'] != 1) | (chartevents['error'].isnull())]

    # Define a list of ITEMIDs related to ventilation.
    vent_itemids = [
        720, 223848, 223849, 467,
        445, 448, 449, 450, 1340, 1486, 1600, 224687,
        639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
        218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747,
        221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
        543, 5865, 5866, 224707, 224709, 224705, 224706,
        60, 437, 505, 506, 686, 220339, 224700,
        3459,
        501, 502, 503, 224702,
        223, 667, 668, 669, 670, 671, 672,
        224701,
        # Oxygen device related
        468, 469, 470, 471, 227287, 226732, 223834
    ]
    chartevents = chartevents[chartevents['itemid'].isin(vent_itemids)]

    # Define a function to set ventilation flags based on itemid and value.
    def determine_flags(row):
        mechvent = 0
        oxygen = 0
        extubated = 0
        self_extubated = 0
        iv = row['itemid']
        val = row['value']
        # Conditions for mechanical ventilation.
        if iv == 720 and val != 'Other/Remarks':
            mechvent = 1
        if iv == 223848 and val != 'Other':
            mechvent = 1
        if iv == 223849:
            mechvent = 1
        if iv == 467 and val == 'Ventilator':
            mechvent = 1
        if iv in [445, 448, 449, 450, 1340, 1486, 1600, 224687,
                  639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
                  218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747,
                  221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
                  543, 5865, 5866, 224707, 224709, 224705, 224706,
                  60, 437, 505, 506, 686, 220339, 224700,
                  3459, 501, 502, 503, 224702,
                  223, 667, 668, 669, 670, 671, 672, 224701]:
            mechvent = 1
        # Conditions for oxygen therapy.
        if iv == 226732 and val in ['Nasal cannula', 'Face tent', 'Aerosol-cool', 'Trach mask ',
                                    'High flow neb', 'Non-rebreather', 'Venti mask ', 'Medium conc mask ',
                                    'T-piece', 'High flow nasal cannula', 'Ultrasonic neb', 'Vapomist']:
            oxygen = 1
        if iv == 467 and val in ['Cannula', 'Nasal Cannula', 'Face Tent', 'Aerosol-Cool', 'Trach Mask',
                                  'Hi Flow Neb', 'Non-Rebreather', 'Venti Mask', 'Medium Conc Mask',
                                  'Vapotherm', 'T-Piece', 'Hood', 'Hut', 'TranstrachealCat',
                                  'Heated Neb', 'Ultrasonic Neb']:
            oxygen = 1
        # Conditions for extubation.
        if iv == 640 and val in ['Extubated', 'Self Extubation']:
            extubated = 1
        if iv == 640 and val == 'Self Extubation':
            self_extubated = 1
        return pd.Series({
            'mechvent': mechvent,
            'oxygentherapy': oxygen,
            'extubated': extubated,
            'selfextubated': self_extubated
        })

    vent_flags = chartevents.apply(determine_flags, axis=1)
    chartevents = pd.concat([chartevents, vent_flags], axis=1)
    vent_chartevents = chartevents.groupby(['icustay_id', 'charttime'], as_index=False).agg({
        'mechvent': 'max',
        'oxygentherapy': 'max',
        'extubated': 'max',
        'selfextubated': 'max'
    })

    # Process PROCEDUREEVENTS_MV to capture extubation events.
    proc_events = pd.read_csv(
        'PROCEDUREEVENTS_MV.csv.gz', compression='gzip', low_memory=False,
        usecols=['ICUSTAY_ID', 'STARTTIME', 'ITEMID']
    )
    proc_events.columns = proc_events.columns.str.lower()
    proc_events = proc_events[proc_events['itemid'].isin([227194, 225468, 225477])]
    proc_events.rename(columns={'starttime': 'charttime'}, inplace=True)
    proc_events['mechvent'] = 0
    proc_events['oxygentherapy'] = 0
    proc_events['extubated'] = 1
    proc_events['selfextubated'] = proc_events['itemid'].apply(lambda x: 1 if x == 225468 else 0)
    vent_proc = proc_events[['icustay_id', 'charttime', 'mechvent', 'oxygentherapy', 'extubated', 'selfextubated']].drop_duplicates()

    # Combine the two sets of ventilation signals and map them to subjects.
    ventilation_flags = pd.concat([vent_chartevents, vent_proc], ignore_index=True).drop_duplicates(subset=['icustay_id', 'charttime'])
    icu_stays_temp = pd.read_csv(
        'ICUSTAYS.csv.gz', compression='gzip',
        usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']
    )
    icu_stays_temp.columns = icu_stays_temp.columns.str.lower()
    ventilation_flags = ventilation_flags.merge(icu_stays_temp[['icustay_id', 'subject_id', 'hadm_id']],
                                                on='icustay_id', how='left')
    ventilation_flags_agg = ventilation_flags.groupby(['subject_id', 'hadm_id'], as_index=False).agg({
        'mechvent': 'max',
        'oxygentherapy': 'max',
        'extubated': 'max',
        'selfextubated': 'max'
    })
    ventilation_flags_agg['mechanical_ventilation'] = ventilation_flags_agg[
        ['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']
    ].max(axis=1)
    return ventilation_flags_agg[['subject_id', 'hadm_id', 'mechanical_ventilation']]

# Load and aggregate LABEVENTS into 2-hour bins for the first 24 hours.
def load_and_aggregate_lab_data(file_path, bin_size=2):
    df = pd.read_csv(file_path, compression='gzip', low_memory=False)
    df.columns = df.columns.str.lower()
    if 'valuenum' not in df.columns:
        print("LABEVENTS missing 'valuenum' column.")
        return None
    df = df[df['valuenum'].notnull()]
    # Load ICU stay admission times.
    icu_stays_lab = pd.read_csv(
        'ICUSTAYS.csv.gz', compression='gzip',
        usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME']
    )
    icu_stays_lab.columns = icu_stays_lab.columns.str.lower()
    icu_stays_lab['intime'] = pd.to_datetime(icu_stays_lab['intime'])
    df = df.merge(icu_stays_lab, on=['subject_id', 'hadm_id'], how='inner')
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    df.dropna(subset=['charttime'], inplace=True)
    df['hours_since_admission'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600
    df = df[df['hours_since_admission'].between(0, 24)]
    df['hour_bin'] = (df['hours_since_admission'] // bin_size).astype(int)
    aggregated_df = df.groupby(['subject_id', 'hadm_id', 'hour_bin', 'itemid'])['valuenum'].mean().unstack().reset_index()
    aggregated_df = aggregated_df.drop(columns=['hour_bin'])
    new_cols = ['subject_id', 'hadm_id'] + [f"lab_t{int(col)}" for col in aggregated_df.columns if col not in ['subject_id', 'hadm_id']]
    aggregated_df.columns = new_cols
    return aggregated_df

# Load and aggregate additional feature data into 2-hour bins.
def load_and_aggregate_feature_data(file_paths, table_name):
    print(f"\nProcessing {table_name} from {file_paths}...")
    # Load one or multiple files.
    if isinstance(file_paths, list):
        df_list = []
        for f in file_paths:
            df_chunk = pd.read_csv(f, compression='gzip', low_memory=False)
            df_list.append(df_chunk)
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(file_paths, compression='gzip', low_memory=False)
    
    df.columns = df.columns.str.lower()  

    if 'subject_id' not in df.columns:
        print(f"{table_name} is missing 'subject_id'. Skipping...")
        return None
    df = df[df['subject_id'].isin(filtered_subjects)]
    print(f"{table_name}: After filtering by subject_id - Shape: {df.shape}")
    
    possible_time_cols = ['charttime', 'starttime', 'storetime', 'eventtime', 'endtime']
    timestamp_col = next((col for col in possible_time_cols if col in df.columns), None)
    if not timestamp_col:
        print(f"{table_name} has no valid timestamp column. Skipping...")
        return None

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df.dropna(subset=[timestamp_col], inplace=True)
    
    # Merge with ICU stays to restrict to first 24 hours.
    df = df.merge(icu_stays[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='inner')
    df['hours_since_admission'] = (df[timestamp_col] - df['intime']).dt.total_seconds() / 3600
    df = df[df['hours_since_admission'].between(0, 24)]
    
    if table_name != 'prescriptions' and 'itemid' in df.columns:
        df = df[df['itemid'].isin(feature_set_C_items.get(table_name, []))]
    
    print(f"{table_name}: After filtering features and time window - Shape: {df.shape}")
    
    # Identify the numeric column.
    numeric_col = next((col for col in ['value', 'amount', 'valuenum'] if col in df.columns), None)
    if not numeric_col:
        print(f"{table_name} has no numeric column. Skipping...")
        return None
    df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
    
    # Bin data into 2-hour intervals.
    df['hour_bin'] = (df['hours_since_admission'] // 2).astype(int)
    agg_func = 'sum' if table_name in ['inputevents', 'outputevents'] else 'mean'
    aggregated_df = df.groupby(['subject_id', 'hadm_id', 'hour_bin', 'itemid'])[numeric_col].agg(agg_func).unstack().reset_index()
    
    if 'hour_bin' in aggregated_df.columns:
        aggregated_df.drop(columns=['hour_bin'], inplace=True)
    
    aggregated_df.columns = ['subject_id', 'hadm_id'] + [f"{table_name}_t{int(col)}" for col in aggregated_df.columns[2:]]
    print(f"{table_name}: Final aggregated shape: {aggregated_df.shape}")
    return aggregated_df


# Build the Base Structured Dataset
# Read in structured tables.
admissions = pd.read_csv(
    'ADMISSIONS.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'INSURANCE']
)
patients = pd.read_csv(
    'PATIENTS.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID', 'GENDER', 'DOB']
)
icu_stays = pd.read_csv(
    'ICUSTAYS.csv.gz', compression='gzip', low_memory=False,
    usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']
)

# Convert date/time columns.
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])
icu_stays['INTIME'] = pd.to_datetime(icu_stays['INTIME'])
icu_stays['OUTTIME'] = pd.to_datetime(icu_stays['OUTTIME'])

# Rename columns for consistency.
admissions.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
icu_stays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)

# Merge ICU stays with Admissions and Patients.
df_struct = pd.merge(icu_stays, admissions, on=['subject_id', 'hadm_id'], how='left')
df_struct = pd.merge(df_struct, patients, on='subject_id', how='left')

# Compute age and assign age bucket.
df_struct['DOB'] = pd.to_datetime(df_struct['DOB'], errors='coerce')
df_struct['age'] = df_struct.apply(lambda row: calculate_age(row['DOB'], row['INTIME'])
                                   if pd.notnull(row['DOB']) and pd.notnull(row['INTIME']) else np.nan, axis=1)
df_struct = df_struct[(df_struct['age'] >= 15) & (df_struct['age'] <= 90)]
df_struct['age_bucket'] = df_struct['age'].apply(categorize_age)

# Standardize ethnicity, insurance, and gender.
df_struct['ethnicity_category'] = df_struct['ETHNICITY'].apply(categorize_ethnicity)
df_struct['insurance_category'] = df_struct['INSURANCE'].apply(categorize_insurance)
df_struct['gender'] = df_struct['GENDER'].str.lower().apply(lambda x: 'male' if 'm' in x else ('female' if 'f' in x else x))

# Calculate short-term mortality.
df_struct = calculate_short_term_mortality(df_struct)

# Compute continuous ICU LOS (in hours)
df_struct['icu_los'] = (df_struct['OUTTIME'] - df_struct['INTIME']).dt.total_seconds() / 3600

# Create los_binary column 
# For LOS prediction (> 3 days), use a threshold of 72 hours.
df_struct['los_binary'] = (df_struct['icu_los'] > 72).astype(int)

# Compute mechanical ventilation flag.
vent_flags = calculate_mechanical_ventilation()
df_struct = pd.merge(df_struct, vent_flags, on=['subject_id', 'hadm_id'], how='left')
df_struct['mechanical_ventilation'] = df_struct['mechanical_ventilation'].fillna(0).astype(int)

# Aggregate LABEVENTS features into 2-hour bins over the first 24 hours.
lab_aggregated = load_and_aggregate_lab_data('LABEVENTS.csv.gz', bin_size=2)
if lab_aggregated is not None:
    df_struct = pd.merge(df_struct, lab_aggregated, on=['subject_id', 'hadm_id'], how='left')

# Sort by admission time and take the first ICU stay per subject.
df_struct = df_struct.sort_values(by='INTIME').groupby('subject_id').first().reset_index()

# Save the base structured dataset.
df_struct.to_csv('final_structured_dataset.csv', index=False)
print("Base structured dataset saved as 'final_structured_dataset.csv'.")

# Merge Feature Set C into the Structured Dataset
# Load the dataset that includes icu_los and los_binary.
structured_df = pd.read_csv('final_structured_dataset.csv')
filtered_subjects = set(structured_df['subject_id'].unique())
print(f"Filtered structured dataset shape: {structured_df.shape}")

# Load ICU stays data (used for feature extraction) and filter for stays longer than 30 hours.
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME'])
icu_stays.columns = icu_stays.columns.str.lower()
icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'])
icu_stays['icu_los'] = (icu_stays['outtime'] - icu_stays['intime']).dt.total_seconds() / 3600
icu_stays = icu_stays[icu_stays['subject_id'].isin(filtered_subjects)]
icu_stays = icu_stays[icu_stays['icu_los'] >= 30]
print(f"ICU stays shape after filtering by subject_id and LOS>=30h: {icu_stays.shape}")

# Define the set of features to extract for each table.
feature_set_C_items = {
    'chartevents': [220051, 220052, 618, 220210, 224641, 220292, 535, 224695, 506, 220339, 448, 224687, 224685, 220293, 444, 224697, 220074, 224688, 223834, 50815, 225664, 220059, 683, 224684, 220060, 226253, 224161, 642, 225185, 226758, 226757, 226756, 220050, 211, 220045, 223761, 223835, 226873, 226871, 8364, 8555, 8368, 53, 646, 1529, 50809, 50931, 51478, 224639, 763, 224639, 226707],
    'labevents': [51221, 51480, 51265, 50811, 51222, 51249, 51248, 51250, 51279, 51277, 50902, 50868, 50912, 50809, 50931, 51478, 50960, 50893, 50970, 51237, 51274, 51275, 51375, 51427, 51446, 51116, 51244, 51355, 51379, 51120, 51254, 51256, 51367, 51387, 51442, 51112, 51146, 51345, 51347, 51368, 51419, 51444, 51114, 51200, 51474, 50820, 50831, 51094, 51491, 50802, 50804, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 490, 1165, 50902, 50819],
    'inputevents': [30008, 220864, 30005, 220970, 221385, 30023, 221456, 221668, 221749, 221794, 221828, 221906, 30027, 222011, 222056, 223258, 30126, 225154, 30297, 225166, 225168, 30144, 225799, 225823, 44367, 225828, 225943, 30065, 225944, 226089, 226364, 30056, 226452, 30059, 226453, 227522, 227523, 30044, 221289, 30051, 222315, 30043, 221662, 30124, 30118, 221744, 30131, 222168],
    'outputevents': [226573, 40054, 40085, 44890, 43703, 226580, 226588, 226589, 226599, 226626, 226633, 227510],
    'prescriptions': ['Docusate Sodium', 'Aspirin', 'Bisacodyl', 'Humulin-R Insulin', 'Metoprolol', 'Pantoprazole Sodium', 'Pantoprazole']
}

# Define input files mapping for each table.
input_files = {
    'chartevents': 'CHARTEVENTS.csv.gz',
    'labevents': 'LABEVENTS.csv.gz',
    'inputevents': ['inputevents_cv.csv.gz', 'inputevents_mv.csv.gz'],
    'outputevents': 'OUTPUTEVENTS.csv.gz',
    'prescriptions': 'PRESCRIPTIONS.csv.gz'
}

aggregated_features = {}
for table, file in input_files.items():
    aggregated_features[table] = load_and_aggregate_feature_data(file, table)

# Merge these aggregated features with the structured dataset.
merged_features = structured_df.copy()
for table_name, feature_df in aggregated_features.items():
    if feature_df is not None:
        merged_features = merged_features.merge(feature_df, on=['subject_id', 'hadm_id'], how='left')

# If icu_los is still missing, merge it from structured_df.
if 'icu_los' not in merged_features.columns:
    if 'icu_los' in structured_df.columns:
        merged_features = merged_features.merge(structured_df[['subject_id', 'icu_los']], on='subject_id', how='left')

# Group by subject_id: average numeric columns and take the first value for categoricals.
numeric_cols = merged_features.select_dtypes(include=[np.number]).columns
categorical_cols = merged_features.select_dtypes(exclude=[np.number]).columns
merged_features_numeric = merged_features.groupby('subject_id', as_index=False)[numeric_cols].mean()
merged_features_categorical = merged_features.groupby('subject_id', as_index=False)[categorical_cols].first()
merged_features = merged_features_numeric.merge(merged_features_categorical, on='subject_id', how='left')

output_file = 'final_structured_with_feature_set_C_24h_2h_bins.csv'
merged_features.to_csv(output_file, index=False)
print(f"\nFinal dataset saved as {output_file}")

print(f"\nFinal Dataset Shape: {merged_features.shape}")
print(f"Short-Term Mortality Count: {merged_features['short_term_mortality'].sum()}")
print(f"Average ICU LOS: {merged_features['icu_los'].mean()}")
print(f"Mechanical Ventilation Count: {merged_features['mechanical_ventilation'].sum()}")

# Unstructured Notes Processing
def preprocess1(x):
    """
    Remove extra characters, numeric bullet points, and standardize abbreviations.
    """
    y = re.sub(r'\[(.*?)\]', '', x)
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'dr\.', 'doctor', y)
    y = re.sub(r'm\.d\.', 'md', y)
    y = re.sub(r'admission date:', '', y)
    y = re.sub(r'discharge date:', '', y)
    y = re.sub(r'--|__|==', '', y)
    return y

def preprocessing(df):
    """
    Preprocess the 'TEXT' column of a dataframe:
    remove newlines, extra whitespace, convert to lower case, and apply cleanup.
    """
    df = df.copy()
    df['TEXT'] = df['TEXT'].fillna(' ')
    df['TEXT'] = df['TEXT'].str.replace('\n', ' ', regex=False)
    df['TEXT'] = df['TEXT'].str.replace('\r', ' ', regex=False)
    df['TEXT'] = df['TEXT'].apply(str.strip)
    df['TEXT'] = df['TEXT'].str.lower()
    df['TEXT'] = df['TEXT'].apply(lambda x: preprocess1(x))
    return df

def split_text_to_chunks(text, chunk_size=512):
    """
    Split a text into chunks of a given token size.
    Tokens are defined by whitespace.
    """
    tokens = text.split()
    chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def split_into_512_token_columns(text, chunk_size=512):
    """
    Given a text, return a Series with one column per chunk.
    """
    chunks = split_text_to_chunks(text, chunk_size)
    chunk_dict = {}
    for i, chunk in enumerate(chunks):
        chunk_dict[f"note_chunk_{i+1}"] = chunk
    return pd.Series(chunk_dict)

# File paths for unstructured data and structured outcomes/demographics.
notes_path = 'NOTEEVENTS.csv.gz'
icustays_path = 'ICUSTAYS.csv.gz'
structured_file = 'final_structured_dataset.csv'  

# Read NOTEEVENTS and ICUSTAYS.
df_notes = pd.read_csv(notes_path, compression='gzip', low_memory=False,
                       usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT'])
df_icustays = pd.read_csv(icustays_path, compression='gzip', low_memory=False,
                          usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME'])

# Convert datetime columns.
df_notes['CHARTDATE'] = pd.to_datetime(df_notes['CHARTDATE'], format='%Y-%m-%d', errors='coerce')
df_icustays['INTIME'] = pd.to_datetime(df_icustays['INTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_icustays['OUTTIME'] = pd.to_datetime(df_icustays['OUTTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Rename columns for consistency.
df_notes.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
df_icustays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)

# Extract the first ICU stay per patient by sorting by INTIME.
df_first_icu = df_icustays.sort_values(by='INTIME').groupby('subject_id').first().reset_index()

# Select notes corresponding to the first ICU stay based on hadm_id.
first_icu_notes = df_notes[df_notes['hadm_id'].isin(df_first_icu['hadm_id'])]

# Merge notes with the ICU admission and discharge times from the first ICU stay.
first_icu_admission = df_first_icu[['subject_id', 'hadm_id', 'INTIME', 'OUTTIME']].copy()
first_icu_admission.rename(columns={'INTIME': 'admission_time', 'OUTTIME': 'discharge_time'}, inplace=True)
notes_merged = pd.merge(first_icu_notes, first_icu_admission, on=['subject_id', 'hadm_id'], how='inner')

# Retain only notes recorded during the ICU stay (between admission_time and discharge_time).
notes_filtered = notes_merged[(notes_merged['CHARTDATE'] >= notes_merged['admission_time']) & 
                              (notes_merged['CHARTDATE'] <= notes_merged['discharge_time'])].copy()

# Aggregate notes by subject and hadm_id by concatenating all TEXT entries.
notes_agg = notes_filtered.groupby(['subject_id', 'hadm_id']).agg({
    'TEXT': lambda texts: " ".join(texts)
}).reset_index()

# Clean the aggregated text.
notes_agg = preprocessing(notes_agg)

# Split the aggregated text into 512-token chunks.
df_note_chunks = notes_agg['TEXT'].apply(split_into_512_token_columns)
notes_agg = pd.concat([notes_agg, df_note_chunks], axis=1)

# Merge with Structured Data
structured_df = pd.read_csv(structured_file)
# If 'los_binary' is not present, compute it (using 72 hours as threshold).
if 'los_binary' not in structured_df.columns:
    structured_df['los_binary'] = (structured_df['icu_los'] > 72).astype(int)

unstructured_merged = pd.merge(
    notes_agg,
    structured_df[['subject_id', 'short_term_mortality', 'icu_los', 'los_binary', 'mechanical_ventilation', 
                     'age', 'age_bucket', 'ethnicity_category', 'insurance_category', 'gender']],
    on='subject_id', how='left'
)

unstructured_merged.to_csv('unstructured_with_demographics.csv', index=False)
print("Unstructured dataset with demographics, outcomes, and notes saved as 'unstructured_with_demographics.csv'.")

# Load final structured dataset 
structured_file = 'final_structured_with_feature_set_C_24h_2h_bins.csv'
structured_df = pd.read_csv(structured_file)
print("Final Structured Dataset:")
print("Shape:", structured_df.shape)
print("Columns:", structured_df.columns.tolist())
print("Short-Term Mortality (positive count):", structured_df['short_term_mortality'].sum())
print("Binary LOS (positive count):", structured_df['los_binary'].sum())
print("Mechanical Ventilation (positive count):", structured_df['mechanical_ventilation'].sum())
print("\n")

# Load final unstructured dataset 
unstructured_file = 'unstructured_with_demographics.csv'
unstructured_df = pd.read_csv(unstructured_file)
print("Final Unstructured Dataset:")
print("Shape:", unstructured_df.shape)
print("Columns:", unstructured_df.columns.tolist())
print("Short-Term Mortality (positive count):", unstructured_df['short_term_mortality'].sum())
print("Binary LOS (positive count):", unstructured_df['los_binary'].sum())
print("Mechanical Ventilation (positive count):", unstructured_df['mechanical_ventilation'].sum())
print("\n")

# Identify common subject IDs between the two datasets.
common_ids = set(structured_df['subject_id'].unique()).intersection(set(unstructured_df['subject_id'].unique()))
print("Number of common subject IDs:", len(common_ids))

# Filter both datasets to include only rows with common subject IDs.
structured_common = structured_df[structured_df['subject_id'].isin(common_ids)].copy()
unstructured_common = unstructured_df[unstructured_df['subject_id'].isin(common_ids)].copy()

# Save the final datasets with common subject IDs.
structured_common.to_csv('final_structured_common.csv', index=False)
unstructured_common.to_csv('final_unstructured_common.csv', index=False)

print("Final Structured (Common IDs) Shape:", structured_common.shape)
print("Final Unstructured (Common IDs) Shape:", unstructured_common.shape)
print("Structured - Short-Term Mortality Count:", structured_common['short_term_mortality'].sum())
print("Structured - Binary LOS Count:", structured_common['los_binary'].sum())
print("Structured - Mechanical Ventilation Count:", structured_common['mechanical_ventilation'].sum())
print("Unstructured - Short-Term Mortality Count:", unstructured_common['short_term_mortality'].sum())
print("Unstructured - Binary LOS Count:", unstructured_common['los_binary'].sum())
print("Unstructured - Mechanical Ventilation Count:", unstructured_common['mechanical_ventilation'].sum())
