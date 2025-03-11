import pandas as pd
import numpy as np
import re
from datetime import timedelta

def calculate_age(dob, intime):
    """Calculate age at ICU admission."""
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def categorize_age(age):
    """Categorize age into bins."""
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

def categorize_ethnicity(ethnicity):
    """Simplify ethnicity descriptions."""
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

def categorize_insurance(insurance):
    """Categorize insurance based on keyword matching."""
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

def calculate_short_term_mortality(df):
    """
    Create a binary column 'short_term_mortality' based on presence of DEATHTIME.
    """
    df['short_term_mortality'] = df['DEATHTIME'].notnull().astype(int)
    return df

def calculate_mechanical_ventilation():
    """
    Load ventilation-related signals from CHARTEVENTS and PROCEDUREEVENTS_MV,
    and create a single 'mechanical_ventilation' indicator.
    """
    # --- CHARTEVENTS ---
    chartevents = pd.read_csv(
        'CHARTEVENTS.csv.gz', compression='gzip', low_memory=False,
        usecols=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'ERROR']
    )
    chartevents.columns = chartevents.columns.str.lower()
    chartevents = chartevents[chartevents['value'].notnull()]
    chartevents = chartevents[(chartevents['error'] != 1) | (chartevents['error'].isnull())]

    # Define ventilation-related ITEMIDs.
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

    def determine_flags(row):
        mechvent = 0
        oxygen = 0
        extubated = 0
        self_extubated = 0
        iv = row['itemid']
        val = row['value']
        # Mechanical Ventilation Conditions:
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
        # Oxygen Therapy Conditions:
        if iv == 226732 and val in ['Nasal cannula', 'Face tent', 'Aerosol-cool', 'Trach mask ',
                                    'High flow neb', 'Non-rebreather', 'Venti mask ', 'Medium conc mask ',
                                    'T-piece', 'High flow nasal cannula', 'Ultrasonic neb', 'Vapomist']:
            oxygen = 1
        if iv == 467 and val in ['Cannula', 'Nasal Cannula', 'Face Tent', 'Aerosol-Cool', 'Trach Mask',
                                  'Hi Flow Neb', 'Non-Rebreather', 'Venti Mask', 'Medium Conc Mask',
                                  'Vapotherm', 'T-Piece', 'Hood', 'Hut', 'TranstrachealCat',
                                  'Heated Neb', 'Ultrasonic Neb']:
            oxygen = 1
        # Extubation Conditions:
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

    # --- PROCEDUREEVENTS_MV for extubation ---
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

    # Combine events and map to subjects via ICUSTAYS.
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

def load_and_aggregate_lab_data(file_path, bin_size=2):
    """
    Load LABEVENTS, filter to patients in our cohort, restrict to the first 24 hours
    of ICU stay, and aggregate numeric lab features into bins (in hours) using mean aggregation.
    """
    df = pd.read_csv(file_path, compression='gzip', low_memory=False)
    df.columns = df.columns.str.lower()
    if 'valuenum' not in df.columns:
        print("LABEVENTS missing 'valuenum' column.")
        return None
    df = df[df['valuenum'].notnull()]
    # Read only needed columns from ICUSTAYS.
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

def load_and_aggregate_feature_data(file_paths, table_name):
    """
    Load table(s), filter features to the first 24h (observation window), and aggregate in 2-hour bins.
    """
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
    
    df.columns = df.columns.str.lower()  # Standardize column names.

    # Ensure 'subject_id' exists and filter for our cohort.
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
    
    df = df.merge(icu_stays[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='inner')
    df['hours_since_admission'] = (df[timestamp_col] - df['intime']).dt.total_seconds() / 3600
    df = df[df['hours_since_admission'].between(0, 24)]
    
    if table_name != 'prescriptions' and 'itemid' in df.columns:
        df = df[df['itemid'].isin(feature_set_C_items.get(table_name, []))]
    
    print(f"{table_name}: After filtering features and time window - Shape: {df.shape}")
    
    # Identify the numeric column to aggregate.
    numeric_col = next((col for col in ['value', 'amount', 'valuenum'] if col in df.columns), None)
    if not numeric_col:
        print(f"{table_name} has no numeric column. Skipping...")
        return None
    
    df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
    
    # Bin the data into 2-hour intervals (0-2, 2-4, ..., 22-24).
    df['hour_bin'] = (df['hours_since_admission'] // 2).astype(int)
    
    # Choose aggregation function: use 'sum' for events and 'mean' for others.
    agg_func = 'sum' if table_name in ['inputevents', 'outputevents'] else 'mean'
    aggregated_df = df.groupby(['subject_id', 'hadm_id', 'hour_bin', 'itemid'])[numeric_col].agg(agg_func).unstack().reset_index()
    
    # Drop the hour_bin column.
    if 'hour_bin' in aggregated_df.columns:
        aggregated_df.drop(columns=['hour_bin'], inplace=True)
    
    # Flatten column names: prefix each feature column with the table name.
    aggregated_df.columns = ['subject_id', 'hadm_id'] + [f"{table_name}_t{int(col)}" for col in aggregated_df.columns[2:]]
    
    print(f"{table_name}: Final aggregated shape: {aggregated_df.shape}")
    return aggregated_df

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

# Convert datetime columns.
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

# Process DOB and compute age at ICU admission.
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

# Calculate ICU length of stay (in hours) and restrict to stays < 72 hours.
df_struct['icu_los'] = (df_struct['OUTTIME'] - df_struct['INTIME']).dt.total_seconds() / 3600
df_struct = df_struct[df_struct['icu_los'] < 72]

# Compute mechanical ventilation flag.
vent_flags = calculate_mechanical_ventilation()
df_struct = pd.merge(df_struct, vent_flags, on=['subject_id', 'hadm_id'], how='left')
df_struct['mechanical_ventilation'] = df_struct['mechanical_ventilation'].fillna(0).astype(int)

# Aggregate lab features (from LABEVENTS) in 2-hour bins over the first 24 hours.
lab_aggregated = load_and_aggregate_lab_data('LABEVENTS.csv.gz', bin_size=2)
if lab_aggregated is not None:
    df_struct = pd.merge(df_struct, lab_aggregated, on=['subject_id', 'hadm_id'], how='left')

# Keep only the first ICU stay per subject.
df_struct = df_struct.sort_values(by='INTIME').groupby('subject_id').first().reset_index()

# Save the base structured dataset.
df_struct.to_csv('final_structured_dataset.csv', index=False)
print("Base structured dataset saved as 'final_structured_dataset.csv'.")

# Merge Feature Set C (2-hour bins) into the Structured Dataset

# For feature extraction, we use a filtered version of the structured dataset.
# Here we assume 'filtered_structured_first_icu_stays.csv' contains your cohort.
structured_df = pd.read_csv('filtered_structured_first_icu_stays.csv')
filtered_subjects = set(structured_df['subject_id'].unique()) 
print(f"Filtered structured dataset shape: {structured_df.shape}")

# Load ICU stays data (for admission times) used for feature extraction.
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME'])
icu_stays.columns = icu_stays.columns.str.lower()
icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'])
icu_stays['icu_los'] = (icu_stays['outtime'] - icu_stays['intime']).dt.total_seconds() / 3600
icu_stays = icu_stays[icu_stays['subject_id'].isin(filtered_subjects)]
icu_stays = icu_stays[icu_stays['icu_los'] >= 30]
print(f"ICU stays shape after filtering by subject_id and LOS>=30h: {icu_stays.shape}")

# Define Feature Set C items.
feature_set_C_items = {
    'chartevents': [220051, 220052, 618, 220210, 224641, 220292, 535, 224695, 506, 220339, 448, 224687, 224685, 220293, 444, 224697, 220074, 224688, 223834, 50815, 225664, 220059, 683, 224684, 220060, 226253, 224161, 642, 225185, 226758, 226757, 226756, 220050, 211, 220045, 223761, 223835, 226873, 226871, 8364, 8555, 8368, 53, 646, 1529, 50809, 50931, 51478, 224639, 763, 224639, 226707],
    'labevents': [51221, 51480, 51265, 50811, 51222, 51249, 51248, 51250, 51279, 51277, 50902, 50868, 50912, 50809, 50931, 51478, 50960, 50893, 50970, 51237, 51274, 51275, 51375, 51427, 51446, 51116, 51244, 51355, 51379, 51120, 51254, 51256, 51367, 51387, 51442, 51112, 51146, 51345, 51347, 51368, 51419, 51444, 51114, 51200, 51474, 50820, 50831, 51094, 51491, 50802, 50804, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 490, 1165, 50902, 50819],
    'inputevents': [30008, 220864, 30005, 220970, 221385, 30023, 221456, 221668, 221749, 221794, 221828, 221906, 30027, 222011, 222056, 223258, 30126, 225154, 30297, 225166, 225168, 30144, 225799, 225823, 44367, 225828, 225943, 30065, 225944, 226089, 226364, 30056, 226452, 30059, 226453, 227522, 227523, 30044, 221289, 30051, 222315, 30043, 221662, 30124, 30118, 221744, 30131, 222168],
    'outputevents': [226573, 40054, 40085, 44890, 43703, 226580, 226588, 226589, 226599, 226626, 226633, 227510],
    'prescriptions': ['Docusate Sodium', 'Aspirin', 'Bisacodyl', 'Humulin-R Insulin', 'Metoprolol', 'Pantoprazole Sodium', 'Pantoprazole']
}

# Define input files mapping.
input_files = {
    'chartevents': 'CHARTEVENTS.csv.gz',
    'labevents': 'LABEVENTS.csv.gz',
    'inputevents': ['inputevents_cv.csv.gz', 'inputevents_mv.csv.gz'],
    'outputevents': 'OUTPUTEVENTS.csv.gz',
    'prescriptions': 'PRESCRIPTIONS.csv.gz'
}

# Process each table from Feature Set C (with 2-hour bins).
aggregated_features = {}
for table, file in input_files.items():
    aggregated_features[table] = load_and_aggregate_feature_data(file, table)

# Merge aggregated features with the filtered structured dataset.
merged_features = structured_df.copy()
for table_name, feature_df in aggregated_features.items():
    if feature_df is not None:
        merged_features = merged_features.merge(feature_df, on=['subject_id', 'hadm_id'], how='left')

# If multiple rows per subject remain, aggregate so that there is one row per subject.
numeric_cols = merged_features.select_dtypes(include=[np.number]).columns
categorical_cols = merged_features.select_dtypes(exclude=[np.number]).columns
merged_features_numeric = merged_features.groupby('subject_id', as_index=False)[numeric_cols].mean()
merged_features_categorical = merged_features.groupby('subject_id', as_index=False)[categorical_cols].first()
merged_features = merged_features_numeric.merge(merged_features_categorical, on='subject_id', how='left')

# Save the final structured dataset with Feature Set C.
output_file = 'final_structured_with_feature_set_C_24h_2h_bins.csv'
merged_features.to_csv(output_file, index=False)
print(f"\nFinal dataset saved as {output_file}")

print(f"\nFinal Dataset Shape: {merged_features.shape}")
print(f"Short-Term Mortality Count: {merged_features['short_term_mortality'].sum()}")
if 'readmission_within_30_days' in merged_features.columns:
    print(f"Readmission Count: {merged_features['readmission_within_30_days'].sum()}")

unstructured_df = pd.read_csv('filtered_unstructured.csv', low_memory=False)
print(f"Unstructured dataset shape: {unstructured_df.shape}")
common_subjects = set(merged_features['subject_id']).intersection(set(unstructured_df['subject_id']))
print(f"Number of common subject IDs: {len(common_subjects)}")
filtered_structured = merged_features[merged_features['subject_id'].isin(common_subjects)]
filtered_structured.to_csv('filtered_structured_output.csv', index=False)
print("Filtered structured dataset saved as 'filtered_structured_output.csv'.")
