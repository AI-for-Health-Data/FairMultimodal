import pandas as pd
import numpy as np
import re
from tqdm import tqdm

########################################
# Functions for Demographics & Text Processing
########################################

def calculate_age(dob, intime):
    """Calculate age at time of ICU stay given date of birth and ICU intime."""
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def categorize_age(age):
    """Categorize age into one of four bins."""
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    else:
        return '70-89'

def categorize_ethnicity(ethnicity):
    """Simplify ethnicity descriptions."""
    ethnicity = ethnicity.upper() if isinstance(ethnicity, str) else ''
    if ethnicity in ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN']:
        return 'White'
    elif ethnicity in ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN', 'BLACK/AFRICAN', 'CARIBBEAN ISLAND']:
        return 'Black'
    elif ethnicity in ['HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - MEXICAN']:
        return 'Hispanic'
    elif ethnicity in ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - INDIAN']:
        return 'Asian'
    else:
        return 'Other'

def categorize_insurance(insurance):
    """Categorize insurance based on keyword matching."""
    ins = insurance.upper() if isinstance(insurance, str) else ''
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
    Preprocess the 'TEXT' column of a dataframe: remove newlines, extra whitespace,
    convert to lower case, and apply cleanup.
    """
    df = df.copy()
    df['TEXT'] = df['TEXT'].fillna(' ')
    df['TEXT'] = df['TEXT'].str.replace('\n', ' ', regex=False)
    df['TEXT'] = df['TEXT'].str.replace('\r', ' ', regex=False)
    df['TEXT'] = df['TEXT'].str.strip()
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

########################################
# Functions for Outcome Calculations
########################################

def calculate_short_term_mortality(df):
    """
    Create a binary column 'short_term_mortality' based on whether DEATHTIME is present.
    """
    df['short_term_mortality'] = df['DEATHTIME'].notnull().astype(int)
    return df

def calculate_los_outcome(df):
    """
    Calculate ICU length-of-stay (LOS) in hours and days and create a binary column 'los_gt_3'
    which is 1 if LOS (in days) > 3, else 0.
    """
    df['icu_los_hours'] = (df['OUTTIME'] - df['INTIME']).dt.total_seconds() / 3600.0
    df['icu_los_days'] = df['icu_los_hours'] / 24.0
    df['los_gt_3'] = (df['icu_los_days'] > 3).astype(int)
    return df

########################################
# 1. Structured Dataset Creation
########################################

# File paths for structured data
admissions_path = 'ADMISSIONS.csv.gz'
icustays_path   = 'ICUSTAYS.csv.gz'
patients_path   = 'PATIENTS.csv.gz'

# Read Admissions, ICU stays, and Patients with selected columns
df_adm = pd.read_csv(admissions_path, compression='gzip', low_memory=False,
                     usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'INSURANCE'])
df_icustays = pd.read_csv(icustays_path, compression='gzip', low_memory=False,
                          usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME'])
df_patients = pd.read_csv(patients_path, compression='gzip', low_memory=False,
                          usecols=['SUBJECT_ID', 'DOB', 'GENDER'])

# Convert datetime columns (Admissions)
df_adm['ADMITTIME'] = pd.to_datetime(df_adm['ADMITTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm['DISCHTIME'] = pd.to_datetime(df_adm['DISCHTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm['DEATHTIME'] = pd.to_datetime(df_adm['DEATHTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Convert datetime columns (ICU stays)
df_icustays['INTIME'] = pd.to_datetime(df_icustays['INTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_icustays['OUTTIME'] = pd.to_datetime(df_icustays['OUTTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Convert DOB in Patients
df_patients['DOB'] = pd.to_datetime(df_patients['DOB'], format='%Y-%m-%d', errors='coerce')

# Rename columns for consistency
df_adm.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
df_icustays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
df_patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)

# Merge Admissions with ICU stays and add patient demographics.
df_icu = pd.merge(df_adm, df_icustays, on=['subject_id', 'hadm_id'], how='inner')
df_icu = pd.merge(df_icu, df_patients[['subject_id', 'DOB', 'GENDER']], on='subject_id', how='left')

# Compute age at ICU admission and categorize.
df_icu['age'] = df_icu.apply(lambda row: calculate_age(row['DOB'], row['INTIME'])
                             if pd.notnull(row['DOB']) and pd.notnull(row['INTIME']) else np.nan, axis=1)
df_icu['age_category'] = df_icu['age'].apply(lambda x: categorize_age(x) if pd.notnull(x) else 'Unknown')

# Categorize ethnicity and insurance.
df_icu['ethnicity_category'] = df_icu['ETHNICITY'].apply(lambda x: categorize_ethnicity(x) if pd.notnull(x) else 'Other')
df_icu['insurance_category'] = df_icu['INSURANCE'].apply(lambda x: categorize_insurance(x) if pd.notnull(x) else 'Other')

# Standardize gender to lowercase and map to 'male' or 'female'.
df_icu['gender'] = df_icu['GENDER'].str.lower().apply(lambda x: 'male' if 'm' in x else ('female' if 'f' in x else x))

# Compute outcomes: short-term mortality and LOS > 3 days.
df_icu = calculate_short_term_mortality(df_icu)
df_icu = calculate_los_outcome(df_icu)

# Select only the first ICU stay per patient (sorted by INTIME).
df_first_icu = df_icu.sort_values(by='INTIME').groupby('subject_id').first().reset_index()

# Save the structured dataset.
df_first_icu.to_csv('final_first_icu_dataset.csv', index=False)
print("Structured dataset (first ICU stay) saved as 'final_first_icu_dataset.csv'.")

########################################
# 2. Unstructured Dataset Creation (Notes)
########################################

# File path for Notes.
notes_path = 'NOTEEVENTS.csv.gz'
df_notes = pd.read_csv(notes_path, compression='gzip', low_memory=False,
                       usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT'])
df_notes['CHARTDATE'] = pd.to_datetime(df_notes['CHARTDATE'], format='%Y-%m-%d', errors='coerce')

# Rename columns for consistency.
df_notes.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)

# Select notes corresponding to the first ICU stay.
first_icu_notes = df_notes[df_notes['hadm_id'].isin(df_first_icu['hadm_id'])]
print("Notes shape after filtering by hadm_id:", first_icu_notes.shape)

# Prepare a dataframe with admission and outcome information.
first_icu_admission = df_first_icu[['subject_id', 'hadm_id', 'INTIME', 'DISCHTIME', 
                                    'short_term_mortality', 'los_gt_3']].copy()
first_icu_admission.rename(columns={'INTIME': 'admission_time', 'DISCHTIME': 'discharge_time'}, inplace=True)

# Merge notes with admissions/outcomes.
notes_merged = pd.merge(first_icu_notes, first_icu_admission, on=['subject_id', 'hadm_id'], how='inner')

# Compute hours since admission.
notes_merged['admission_time'] = pd.to_datetime(notes_merged['admission_time'], errors='coerce')
notes_merged['hours_since_admission'] = (notes_merged['CHARTDATE'] - notes_merged['admission_time']).dt.total_seconds() / 3600

# Keep only notes recorded during the ICU stay period.
notes_filtered = notes_merged[(notes_merged['CHARTDATE'] >= notes_merged['admission_time']) & 
                              (notes_merged['CHARTDATE'] <= notes_merged['discharge_time'])].copy()

# Aggregate notes by subject and hadm_id.
notes_agg = notes_filtered.groupby(['subject_id', 'hadm_id']).agg({
    'TEXT': lambda texts: " ".join(texts),
    'short_term_mortality': 'first',
    'los_gt_3': 'first'
}).reset_index()

# Clean the aggregated text.
notes_agg = preprocessing(notes_agg)

# Split the aggregated text into 512-token chunks.
df_note_chunks = notes_agg['TEXT'].apply(split_into_512_token_columns)
notes_agg = pd.concat([notes_agg, df_note_chunks], axis=1)

# Save the unstructured notes dataset.
notes_agg.to_csv('final_unstructured_all_notes.csv', index=False)
print("Unstructured notes dataset saved as 'final_unstructured_all_notes.csv'.")

########################################
# 3. Filtering Both Datasets to Common Subject IDs
########################################

# Load the structured and unstructured datasets.
structured_df = pd.read_csv('final_first_icu_dataset.csv')
unstructured_df = pd.read_csv('final_unstructured_all_notes.csv', engine='python', on_bad_lines='skip')

print("Structured dataset shape:", structured_df.shape)
print("Unstructured dataset shape:", unstructured_df.shape)

# Identify common subject IDs.
common_ids = set(structured_df['subject_id'].unique()).intersection(set(unstructured_df['subject_id'].unique()))
print(f"Number of common subject IDs: {len(common_ids)}")

# Filter each dataset to only include common subject IDs.
filtered_structured = structured_df[structured_df['subject_id'].isin(common_ids)].copy()
filtered_unstructured = unstructured_df[unstructured_df['subject_id'].isin(common_ids)].copy()

# Save the final filtered datasets.
filtered_structured.to_csv('filtered_structured_dataset.csv', index=False)
filtered_unstructured.to_csv('filtered_unstructured_dataset.csv', index=False)
print("Filtered structured dataset saved as 'filtered_structured_dataset.csv'.")
print("Filtered unstructured dataset saved as 'filtered_unstructured_dataset.csv'.")

########################################
# 4. Add Feature Set C: Aggregation in 2-Hour Bins over First 24 Hours
########################################

# Load the filtered structured dataset.
structured_df = pd.read_csv('filtered_structured_first_icu_stays.csv')
filtered_subjects = set(structured_df['subject_id'].unique())
print(f"Filtered structured dataset shape: {structured_df.shape}")

# Load ICU stays data.
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME'])
icu_stays.columns = icu_stays.columns.str.lower()
icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'])

# Compute ICU length-of-stay in hours.
icu_stays['icu_los'] = (icu_stays['outtime'] - icu_stays['intime']).dt.total_seconds() / 3600

# Filter ICU stays for our cohort and for stays >= 30 hours.
icu_stays = icu_stays[icu_stays['subject_id'].isin(filtered_subjects)]
icu_stays = icu_stays[icu_stays['icu_los'] >= 30]
print(f"ICU stays data shape after filtering: {icu_stays.shape}")

# Define Feature Set C items.
feature_set_C_items = {
    'chartevents': [220051, 220052, 618, 220210, 224641, 220292, 535, 224695, 506, 220339, 448, 224687, 224685, 220293, 444, 224697, 220074, 224688, 223834, 50815, 225664, 220059, 683, 224684, 220060, 226253, 224161, 642, 225185, 226758, 226757, 226756, 220050, 211, 220045, 223761, 223835, 226873, 226871, 8364, 8555, 8368, 53, 646, 1529, 50809, 50931, 51478, 224639, 763, 224639, 226707],
    'labevents': [51221, 51480, 51265, 50811, 51222, 51249, 51248, 51250, 51279, 51277, 50902, 50868, 50912, 50809, 50931, 51478, 50960, 50893, 50970, 51237, 51274, 51275, 51375, 51427, 51446, 51116, 51244, 51355, 51379, 51120, 51254, 51256, 51367, 51387, 51442, 51112, 51146, 51345, 51347, 51368, 51419, 51444, 51114, 51200, 51474, 50820, 50831, 51094, 51491, 50802, 50804, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 490, 1165, 50902, 50819],
    'inputevents': [30008, 220864, 30005, 220970, 221385, 30023, 221456, 221668, 221749, 221794, 221828, 221906, 30027, 222011, 222056, 223258, 30126, 225154, 30297, 225166, 225168, 30144, 225799, 225823, 44367, 225828, 225943, 30065, 225944, 226089, 226364, 30056, 226452, 30059, 226453, 227522, 227523, 30044, 221289, 30051, 222315, 30043, 221662, 30124, 30118, 221744, 30131, 222168],
    'outputevents': [226573, 40054, 40085, 44890, 43703, 226580, 226588, 226589, 226599, 226626, 226633, 227510],
    'prescriptions': ['Docusate Sodium', 'Aspirin', 'Bisacodyl', 'Humulin-R Insulin', 'Metoprolol', 'Pantoprazole Sodium', 'Pantoprazole']
}

input_files = {
    'chartevents': 'CHARTEVENTS.csv.gz',
    'labevents': 'LABEVENTS.csv.gz',
    'inputevents': ['inputevents_cv.csv.gz', 'inputevents_mv.csv.gz'],
    'outputevents': 'OUTPUTEVENTS.csv.gz',
    'prescriptions': 'PRESCRIPTIONS.csv.gz'
}

def load_and_aggregate_feature_data(file_paths, table_name):
    """
    Load table(s), filter to first 24 hours (observation window), and aggregate in 2-hour bins.
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
    
    numeric_col = next((col for col in ['value', 'amount', 'valuenum'] if col in df.columns), None)
    if not numeric_col:
        print(f"{table_name} has no numeric column. Skipping...")
        return None
    
    df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
    
    # Bin into 2-hour intervals.
    df['hour_bin'] = (df['hours_since_admission'] // 2).astype(int)
    
    agg_func = 'sum' if table_name in ['inputevents', 'outputevents'] else 'mean'
    aggregated_df = df.groupby(['subject_id', 'hadm_id', 'hour_bin', 'itemid'])[numeric_col].agg(agg_func).unstack().reset_index()
    
    if 'hour_bin' in aggregated_df.columns:
        aggregated_df.drop(columns=['hour_bin'], inplace=True)
    
    aggregated_df.columns = ['subject_id', 'hadm_id'] + [f"{table_name}_t{int(col)}" for col in aggregated_df.columns[2:]]
    
    print(f"{table_name}: Final aggregated shape: {aggregated_df.shape}")
    return aggregated_df

aggregated_features = {}
for table, file in input_files.items():
    aggregated_features[table] = load_and_aggregate_feature_data(file, table)

merged_features = structured_df.copy()
for table_name, feature_df in aggregated_features.items():
    if feature_df is not None:
        merged_features = merged_features.merge(feature_df, on=['subject_id', 'hadm_id'], how='left')

numeric_cols = merged_features.select_dtypes(include=[np.number]).columns
categorical_cols = merged_features.select_dtypes(exclude=[np.number]).columns

merged_features_numeric = merged_features.groupby('subject_id', as_index=False)[numeric_cols].mean()
merged_features_categorical = merged_features.groupby('subject_id', as_index=False)[categorical_cols].first()

merged_features = merged_features_numeric.merge(merged_features_categorical, on='subject_id', how='left')

output_file = 'final_structured_with_feature_set_C_24h_2h_bins.csv'
merged_features.to_csv(output_file, index=False)
print(f"\nFinal dataset saved as {output_file}")

########################################
# 5. Final Summary and Printing
########################################

print(f"\nFinal Dataset Shape: {merged_features.shape}")
print("Columns in Final Dataset:")
print(merged_features.columns.tolist())
print("Short-Term Mortality Count:", merged_features['short_term_mortality'].sum())
# Check for LOS column and print count; if missing, print a message.
if 'los_gt_3' in merged_features.columns:
    print("LOS (>3 days) Count:", merged_features['los_gt_3'].sum())
else:
    print("Column 'los_gt_3' not found in final dataset.")

# (Optional) Load unstructured data and filter to common subjects.
unstructured_df = pd.read_csv('filtered_unstructured.csv', low_memory=False)
print(f"\nUnstructured dataset shape: {unstructured_df.shape}")

common_subjects = set(merged_features['subject_id']).intersection(set(unstructured_df['subject_id']))
print(f"Number of common subject IDs: {len(common_subjects)}")

filtered_structured = merged_features[merged_features['subject_id'].isin(common_subjects)]
filtered_structured.to_csv('filtered_structured_output.csv', index=False)
print("Filtered structured dataset saved as 'filtered_structured_output.csv'.")

print(f"\nFiltered Structured Dataset Shape: {filtered_structured.shape}")
print("Columns in Filtered Structured Dataset:")
print(filtered_structured.columns.tolist())
print("Short-Term Mortality Count:", filtered_structured['short_term_mortality'].sum())
if 'los_gt_3' in filtered_structured.columns:
    print("LOS (>3 days) Count:", filtered_structured['los_gt_3'].sum())
else:
    print("Column 'los_gt_3' not found in filtered structured dataset.")
