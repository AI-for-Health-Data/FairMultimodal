import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

def calculate_age(dob, intime):
    """Calculate age at time of ICU stay given DOB and ICU intime."""
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

def categorize_age(age):
    """Categorize age into bins."""
    # Note: here we use a simple binning. You may adjust if you have extreme values.
    if age < 15:
        return '0-14'
    elif 15 <= age <= 29:
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
    """Categorize insurance based on keywords."""
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
    """Clean up note text."""
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
    Preprocess the 'TEXT' column: remove newlines, extra whitespace,
    lower-case, and clean up.
    """
    df = df.copy()
    df['TEXT'] = df['TEXT'].fillna(' ')
    df['TEXT'] = df['TEXT'].str.replace('\n', ' ', regex=False)
    df['TEXT'] = df['TEXT'].str.replace('\r', ' ', regex=False)
    df['TEXT'] = df['TEXT'].str.strip().str.lower()
    df['TEXT'] = df['TEXT'].apply(preprocess1)
    return df

def split_text_to_chunks(text, chunk_size=512):
    """Split text into chunks defined by whitespace tokens."""
    tokens = text.split()
    chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def split_into_512_token_columns(text, chunk_size=512):
    """Return a Series with one column per 512-token chunk."""
    chunks = split_text_to_chunks(text, chunk_size)
    chunk_dict = {f"note_chunk_{i+1}": chunk for i, chunk in enumerate(chunks)}
    return pd.Series(chunk_dict)

def calculate_short_term_mortality(df):
    """Create binary 'short_term_mortality' (1 if DEATHTIME exists)."""
    df['short_term_mortality'] = df['DEATHTIME'].notnull().astype(int)
    return df

def calculate_los_outcome(df):
    """
    Calculate ICU length-of-stay in hours and days.
    Create binary column 'los_gt_3' which is 1 if ICU LOS (days) > 3.
    """
    df['icu_los_hours'] = (df['OUTTIME'] - df['INTIME']).dt.total_seconds() / 3600.0
    df['icu_los_days'] = df['icu_los_hours'] / 24.0
    df['los_gt_3'] = (df['icu_los_days'] > 3).astype(int)
    return df

# File paths for structured data (adjust paths as needed)
admissions_path = 'ADMISSIONS.csv.gz'
icustays_path   = 'ICUSTAYS.csv.gz'
patients_path   = 'PATIENTS.csv.gz'

# Read selected columns
df_adm = pd.read_csv(admissions_path, compression='gzip', low_memory=False,
                     usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'INSURANCE'])
df_icustays = pd.read_csv(icustays_path, compression='gzip', low_memory=False,
                          usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME'])
df_patients = pd.read_csv(patients_path, compression='gzip', low_memory=False,
                          usecols=['SUBJECT_ID', 'DOB', 'GENDER'])

# Convert datetime columns
df_adm['ADMITTIME'] = pd.to_datetime(df_adm['ADMITTIME'], errors='coerce')
df_adm['DISCHTIME'] = pd.to_datetime(df_adm['DISCHTIME'], errors='coerce')
df_adm['DEATHTIME'] = pd.to_datetime(df_adm['DEATHTIME'], errors='coerce')
df_icustays['INTIME'] = pd.to_datetime(df_icustays['INTIME'], errors='coerce')
df_icustays['OUTTIME'] = pd.to_datetime(df_icustays['OUTTIME'], errors='coerce')
df_patients['DOB'] = pd.to_datetime(df_patients['DOB'], errors='coerce')

# Rename columns for consistency
df_adm.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
df_icustays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
df_patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)

# Merge Admissions with ICU stays, then merge with Patients
df_icu = pd.merge(df_adm, df_icustays, on=['subject_id', 'hadm_id'], how='inner')
df_icu = pd.merge(df_icu, df_patients[['subject_id', 'DOB', 'GENDER']], on='subject_id', how='left')

# Compute age and categorize
df_icu['age'] = df_icu.apply(lambda row: calculate_age(row['DOB'], row['INTIME']) 
                             if pd.notnull(row['DOB']) and pd.notnull(row['INTIME']) else np.nan, axis=1)
df_icu['age_category'] = df_icu['age'].apply(lambda x: categorize_age(x) if pd.notnull(x) else 'Unknown')

# Categorize ethnicity and insurance
df_icu['ethnicity_category'] = df_icu['ETHNICITY'].apply(lambda x: categorize_ethnicity(x) if pd.notnull(x) else 'Other')
df_icu['insurance_category'] = df_icu['INSURANCE'].apply(lambda x: categorize_insurance(x) if pd.notnull(x) else 'Other')

# Standardize gender to lowercase and map to 'male' or 'female'
df_icu['gender'] = df_icu['GENDER'].str.lower().apply(lambda x: 'male' if 'm' in x else ('female' if 'f' in x else x))

# Compute outcomes: short-term mortality and LOS outcomes
df_icu = calculate_short_term_mortality(df_icu)
df_icu = calculate_los_outcome(df_icu)

# Select only the first ICU stay per patient (sorted by INTIME)
df_first_icu = df_icu.sort_values(by='INTIME').groupby('subject_id').first().reset_index()

# Save intermediate structured dataset (first ICU stay)
df_first_icu.to_csv('final_first_icu_dataset.csv', index=False)
print("Structured dataset (first ICU stay) saved as 'final_first_icu_dataset.csv'.")

notes_path = 'NOTEEVENTS.csv.gz'
df_notes = pd.read_csv(notes_path, compression='gzip', low_memory=False,
                       usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT'])
df_notes['CHARTDATE'] = pd.to_datetime(df_notes['CHARTDATE'], errors='coerce')
df_notes.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)

# Filter notes to those with hadm_id in our first ICU stays
first_icu_hadm_ids = df_first_icu['hadm_id'].unique()
first_icu_notes = df_notes[df_notes['hadm_id'].isin(first_icu_hadm_ids)]
print("Notes shape after filtering by hadm_id:", first_icu_notes.shape)

# Prepare a dataframe with admission and outcome information (we use INTIME, DISCHTIME, etc.)
first_icu_admission = df_first_icu[['subject_id', 'hadm_id', 'INTIME', 'DISCHTIME', 
                                    'short_term_mortality', 'icu_los_hours', 'icu_los_days', 'los_gt_3',
                                    'age', 'age_category', 'gender', 'ethnicity_category', 'insurance_category']].copy()
first_icu_admission.rename(columns={'INTIME': 'admission_time', 'DISCHTIME': 'discharge_time'}, inplace=True)

# Merge notes with admission/outcome/demographic info
notes_merged = pd.merge(first_icu_notes, first_icu_admission, on=['subject_id', 'hadm_id'], how='inner')
notes_merged['admission_time'] = pd.to_datetime(notes_merged['admission_time'], errors='coerce')
# Only keep notes within the ICU stay window
notes_merged = notes_merged[(notes_merged['CHARTDATE'] >= notes_merged['admission_time']) & 
                            (notes_merged['CHARTDATE'] <= notes_merged['discharge_time'])].copy()

# Aggregate notes per patient and ICU stay
notes_agg = notes_merged.groupby(['subject_id', 'hadm_id']).agg({
    'TEXT': lambda texts: " ".join(texts),
    'short_term_mortality': 'first',
    'icu_los_hours': 'first',
    'icu_los_days': 'first',
    'los_gt_3': 'first',
    'age': 'first',
    'age_category': 'first',
    'gender': 'first',
    'ethnicity_category': 'first',
    'insurance_category': 'first'
}).reset_index()

# Preprocess aggregated text and split into 512-token chunks
notes_agg = preprocessing(notes_agg)
note_chunks = notes_agg['TEXT'].apply(split_into_512_token_columns)
notes_agg = pd.concat([notes_agg, note_chunks], axis=1)

# Save intermediate unstructured dataset
notes_agg.to_csv('final_unstructured_all_notes.csv', index=False)
print("Unstructured notes dataset saved as 'final_unstructured_all_notes.csv'.")

def load_and_aggregate_feature_data(file_paths, table_name, filtered_subjects, icu_stays, feature_ids=None):
    """
    Load one or multiple files, filter by subject_id and time window (first 24 hours), and aggregate in 2-hour bins.
    If feature_ids is provided and the table has an 'itemid' column, filter to those features.
    """
    print(f"\nProcessing {table_name} from {file_paths}...")
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
        print(f"{table_name} missing subject_id. Skipping...")
        return None
    df = df[df['subject_id'].isin(filtered_subjects)]
    print(f"{table_name}: After filtering by subject_id - Shape: {df.shape}")
    
    # Identify a timestamp column
    possible_time_cols = ['charttime', 'starttime', 'storetime', 'eventtime', 'endtime']
    timestamp_col = next((col for col in possible_time_cols if col in df.columns), None)
    if not timestamp_col:
        print(f"{table_name} has no valid timestamp column. Skipping...")
        return None
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df.dropna(subset=[timestamp_col], inplace=True)
    
    # Merge with ICU stays to get admission time (use subject_id and hadm_id)
    df = df.merge(icu_stays[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='inner')
    df['hours_since_admission'] = (df[timestamp_col] - df['intime']).dt.total_seconds() / 3600
    df = df[df['hours_since_admission'].between(0, 24)]
    
    if feature_ids is not None and 'itemid' in df.columns:
        df = df[df['itemid'].isin(feature_ids)]
    
    # Convert numeric column
    numeric_col = next((col for col in ['value', 'amount', 'valuenum'] if col in df.columns), None)
    if not numeric_col:
        print(f"{table_name} has no numeric column. Skipping...")
        return None
    df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
    
    # Bin into 2-hour intervals.
    df['hour_bin'] = (df['hours_since_admission'] // 2).astype(int)
    
    # Choose aggregation function: sum for inputs/outputs, mean for labs.
    agg_func = 'sum' if table_name in ['inputevents', 'outputevents'] else 'mean'
    aggregated_df = df.groupby(['subject_id', 'hadm_id', 'hour_bin', 'itemid'])[numeric_col].agg(agg_func).unstack()
    aggregated_df = aggregated_df.reset_index()
    if 'hour_bin' in aggregated_df.columns:
        aggregated_df.drop(columns=['hour_bin'], inplace=True)
    aggregated_df.columns = ['subject_id', 'hadm_id'] + [f"{table_name}_t{int(col)}" for col in aggregated_df.columns[2:]]
    
    print(f"{table_name}: Final aggregated shape: {aggregated_df.shape}")
    return aggregated_df

# Define feature IDs for each table if needed (adjust these lists as required)
feature_set_C_items = {
    'chartevents': [220051, 220052, 618, 220210, 224641, 220292, 535, 224695, 506, 220339, 448, 224687, 224685, 220293, 444, 224697, 220074, 224688, 223834, 50815, 225664, 220059, 683, 224684, 220060, 226253, 224161, 642, 225185, 226758, 226757, 226756, 220050, 211, 220045, 223761, 223835, 226873, 226871, 8364, 8555, 8368, 53, 646, 1529, 50809, 50931, 51478, 224639, 763, 224639, 226707],  
    'labevents': [51221, 51480, 51265, 50811, 51222, 51249, 51248, 51250, 51279, 51277, 50902, 50868, 50912, 50809, 50931, 51478, 50960, 50893, 50970, 51237, 51274, 51275, 51375, 51427, 51446, 51116, 51244, 51355, 51379, 51120, 51254, 51256, 51367, 51387, 51442, 51112, 51146, 51345, 51347, 51368, 51419, 51444, 51114, 51200, 51474, 50820, 50831, 51094, 51491, 50802, 50804, 50818, 51498, 50813, 50861, 50878, 50863, 50862, 490, 1165, 50902, 50819],  
    'inputevents': [30008, 220864, 30005, 220970, 221385, 30023, 221456, 221668, 221749, 221794, 221828, 221906, 30027, 222011, 222056, 223258, 30126, 225154, 30297, 225166, 225168, 30144, 225799, 225823, 44367, 225828, 225943, 30065, 225944, 226089, 226364, 30056, 226452, 30059, 226453, 227522, 227523, 30044, 221289, 30051, 222315, 30043, 221662, 30124, 30118, 221744, 30131, 222168],  
    'outputevents': [226573, 40054, 40085, 44890, 43703, 226580, 226588, 226589, 226599, 226626, 226633, 227510],  # Urine output
    'prescriptions': ['Docusate Sodium', 'Aspirin', 'Bisacodyl', 'Humulin-R Insulin', 'Metoprolol', 'Pantoprazole Sodium', 'Pantoprazole']
}

# Get filtered subjects and ICU stays from our first ICU dataset.
filtered_subjects = set(df_first_icu['subject_id'].unique())
icu_stays = pd.read_csv(icustays_path, compression='gzip', low_memory=False,
                        usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME'])
icu_stays.columns = icu_stays.columns.str.lower()
icu_stays.rename(columns={'subject_id': 'subject_id', 'hadm_id': 'hadm_id'}, inplace=True)
icu_stays['intime'] = pd.to_datetime(icu_stays['intime'], errors='coerce')

# Aggregate features from different tables.
aggregated_features = {}

input_files = {
    'chartevents': 'CHARTEVENTS.csv.gz',
    'labevents': 'LABEVENTS.csv.gz',
    'inputevents': ['inputevents_cv.csv.gz', 'inputevents_mv.csv.gz'],
    'outputevents': 'OUTPUTEVENTS.csv.gz',
    'prescriptions': 'PRESCRIPTIONS.csv.gz'
}

for table, file in input_files.items():
    aggregated_features[table] = load_and_aggregate_feature_data(file, table, filtered_subjects, icu_stays,
                                                                   feature_ids=feature_set_C_items.get(table))

# Merge aggregated features with our structured dataset.
merged_structured = df_first_icu.copy()
for table_name, feature_df in aggregated_features.items():
    if feature_df is not None:
        merged_structured = pd.merge(merged_structured, feature_df, on=['subject_id', 'hadm_id'], how='left')

# Save final structured dataset with aggregated features.
merged_structured.to_csv('final_structured_with_feature_set_C_24h_2h_bins.csv', index=False)
print("\nFinal structured dataset with feature set saved as 'final_structured_with_feature_set_C_24h_2h_bins.csv'.")

# Load final structured and unstructured datasets.
structured_final = pd.read_csv('final_structured_with_feature_set_C_24h_2h_bins.csv')
unstructured_final = pd.read_csv('final_unstructured_all_notes.csv', engine='python', on_bad_lines='skip')

# Identify common subject IDs.
common_ids = set(structured_final['subject_id'].unique()).intersection(set(unstructured_final['subject_id'].unique()))
print(f"\nNumber of common subject IDs: {len(common_ids)}")

# Filter both datasets.
structured_final = structured_final[structured_final['subject_id'].isin(common_ids)].copy()
unstructured_final = unstructured_final[unstructured_final['subject_id'].isin(common_ids)].copy()

# Save final filtered datasets.
structured_final.to_csv('filtered_structured_output.csv', index=False)
unstructured_final.to_csv('filtered_unstructured_dataset.csv', index=False)
print("Final filtered structured dataset saved as 'filtered_structured_output.csv'.")
print("Final filtered unstructured dataset saved as 'filtered_unstructured_dataset.csv'.")

print("\n--- Final Structured Dataset ---")
print("Shape:", structured_final.shape)
print("Columns:", structured_final.columns.tolist())
print("Short-Term Mortality Count:", structured_final['short_term_mortality'].sum())
if 'los_gt_3' in structured_final.columns:
    print("LOS (>3 days) Count:", structured_final['los_gt_3'].sum())
else:
    print("Column 'los_gt_3' not found.")

print("\n--- Final Unstructured Dataset ---")
print("Shape:", unstructured_final.shape)
print("Columns:", unstructured_final.columns.tolist())
# For outcomes, assume the unstructured dataset has the merged outcome columns from the notes aggregation.
if 'short_term_mortality' in unstructured_final.columns:
    print("Short-Term Mortality Count:", unstructured_final['short_term_mortality'].sum())
else:
    print("Column 'short_term_mortality' not found in unstructured data.")
if 'los_gt_3' in unstructured_final.columns:
    print("LOS (>3 days) Count:", unstructured_final['los_gt_3'].sum())
else:
    print("Column 'los_gt_3' not found in unstructured data.")
