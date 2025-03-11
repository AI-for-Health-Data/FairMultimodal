import pandas as pd
import numpy as np
import re
from datetime import timedelta

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


# Unstructured Notes Processing
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

# Load the structured dataset (which includes outcomes and demographics).
structured_df = pd.read_csv(structured_file)
# Expected columns in structured_df include: 
# subject_id, short_term_mortality, icu_los, los_binary, mechanical_ventilation, 
# age, age_bucket, ethnicity_category, insurance_category, gender

# Merge the structured outcomes/demographics with the aggregated notes on subject_id.
unstructured_merged = pd.merge(
    notes_agg,
    structured_df[['subject_id', 'short_term_mortality', 'icu_los', 'los_binary', 'mechanical_ventilation', 
                   'age', 'age_bucket', 'ethnicity_category', 'insurance_category', 'gender']],
    on='subject_id', how='left'
)

# Save the final unstructured dataset with outcomes, demographics, and notes.
unstructured_merged.to_csv('unstructured_with_demographics.csv', index=False)
print("Unstructured dataset with demographics, outcomes, and notes saved as 'unstructured_with_demographics.csv'.")
