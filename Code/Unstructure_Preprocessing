import pandas as pd
import spacy
import os

# Load SpaCy English tokenizer
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000  

# Function to calculate age
def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

# Function to categorize age
def categorize_age(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    else:
        return '70-89'

# Function to categorize ethnicity
def categorize_ethnicity(ethnicity):
    ethnicity = ethnicity.upper()
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

# Function to categorize insurance
def categorize_insurance(insurance):
    if 'MEDICARE' in insurance.upper():
        return 'Medicare'
    elif 'PRIVATE' in insurance.upper():
        return 'Private'
    elif 'MEDICAID' in insurance.upper():
        return 'Medicaid'
    elif 'SELF PAY' in insurance.upper():
        return 'Self Pay'
    else:
        return 'Government'

# Function to calculate short-term mortality
def calculate_short_term_mortality(icu_stays):
    icu_stays['short_term_mortality'] = icu_stays['DEATHTIME'].notnull().astype(int)
    return icu_stays

# Function to calculate readmission within 30 days
def calculate_readmission(icu_stays):
    if 'DISCHTIME' not in icu_stays.columns or 'INTIME' not in icu_stays.columns or 'hadm_id' not in icu_stays.columns:
        raise KeyError("Required columns are missing in the input data.")
    
    # Sort by subject_id, admission time, and ICU intime
    icu_stays = icu_stays.sort_values(by=['subject_id', 'ADMITTIME', 'INTIME'])
    
    # Extract DISCHTIME of the current admission
    icu_stays['current_admission_dischtime'] = icu_stays.groupby(['subject_id', 'hadm_id'])['DISCHTIME'].transform('first')
    
    # Identify the INTIME of the first ICU stay in the next admission
    icu_stays['next_admission_icu_intime'] = icu_stays.groupby('subject_id')['INTIME'].shift(-1)
    icu_stays['next_hadm_id'] = icu_stays.groupby('subject_id')['hadm_id'].shift(-1)
    
    # Calculate time difference between DISCHTIME of current admission and INTIME of next ICU stay
    icu_stays['readmission_within_30_days'] = (
        (icu_stays['next_admission_icu_intime'] - icu_stays['current_admission_dischtime']).dt.days <= 30
    ).astype(int)
    
    icu_stays['readmission_within_30_days'] = icu_stays['readmission_within_30_days'].fillna(0).astype(int)
    return icu_stays

# Function to split long text into manageable chunks
def split_large_text(text, max_chunk_size=1000000):
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

# Function to split notes into chunks of max_tokens
def split_notes(note, max_tokens=500):
    if len(note) > nlp.max_length:
        chunks = split_large_text(note)
    else:
        chunks = [note]
    token_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens = [token.text for token in doc]
        token_chunks.extend([' '.join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)])
    return token_chunks

# Function to process notes and split long notes
def process_notes(data, note_column='note', max_tokens=500):
    data['note_chunks'] = data[note_column].apply(lambda x: split_notes(x, max_tokens))
    max_chunks = data['note_chunks'].apply(len).max()

    # Create chunked columns using pd.concat
    chunked_data = pd.concat(
        [data] + [pd.DataFrame(data['note_chunks'].tolist(), columns=[f'note_{i + 1}' for i in range(max_chunks)])],
        axis=1
    )
    return chunked_data.drop(columns=[note_column, 'note_chunks'])

# Preprocessing function for ICU data
def preprocess_icu_data(noteevents_file, icustays_file, patients_file, admissions_file, output_file, max_tokens=500):
    if not os.path.exists(noteevents_file):
        raise FileNotFoundError(f"NOTEEVENTS file {noteevents_file} not found.")
    if not os.path.exists(icustays_file):
        raise FileNotFoundError(f"ICUSTAYS file {icustays_file} not found.")
    if not os.path.exists(patients_file):
        raise FileNotFoundError(f"PATIENTS file {patients_file} not found.")
    if not os.path.exists(admissions_file):
        raise FileNotFoundError(f"ADMISSIONS file {admissions_file} not found.")

    # Load NOTEEVENTS data
    print(f"Loading data from {noteevents_file}...")
    noteevents = pd.read_csv(noteevents_file, compression='gzip', low_memory=False)
    noteevents.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    noteevents['TEXT'] = noteevents['TEXT'].fillna('')

    # Load ICUSTAYS data
    print(f"Loading data from {icustays_file}...")
    icu_stays = pd.read_csv(icustays_file, compression='gzip', low_memory=False)
    icu_stays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    icu_stays['INTIME'] = pd.to_datetime(icu_stays['INTIME'])
    icu_stays['OUTTIME'] = pd.to_datetime(icu_stays['OUTTIME'])

    # Load ADMISSIONS data
    print(f"Loading data from {admissions_file}...")
    admissions = pd.read_csv(admissions_file, compression='gzip', low_memory=False)
    admissions.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])

    # Merge ICUSTAYS with ADMISSIONS to include required columns
    icu_stays = pd.merge(
        icu_stays,
        admissions[['subject_id', 'hadm_id', 'DEATHTIME', 'ADMITTIME', 'DISCHTIME', 'ETHNICITY', 'INSURANCE']],
        on=['subject_id', 'hadm_id'],
        how='left'
    )

    # Load PATIENTS data
    print(f"Loading data from {patients_file}...")
    patients = pd.read_csv(patients_file, compression='gzip', low_memory=False)
    patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
    patients['DOB'] = pd.to_datetime(patients['DOB'])

    # Merge ICUSTAYS with PATIENTS to calculate age
    icu_stays = pd.merge(icu_stays, patients[['subject_id', 'DOB', 'GENDER']], on='subject_id', how='left')
    icu_stays['age'] = icu_stays.apply(lambda x: calculate_age(x['DOB'], x['INTIME']), axis=1)
    icu_stays = icu_stays[(icu_stays['age'] >= 15) & (icu_stays['age'] <= 90)]
    icu_stays['age_bucket'] = icu_stays['age'].apply(categorize_age)
    icu_stays['categorized_ethnicity'] = icu_stays['ETHNICITY'].apply(categorize_ethnicity)
    icu_stays['categorized_insurance'] = icu_stays['INSURANCE'].apply(categorize_insurance)

    # Calculate short-term mortality and readmission
    print("Calculating short-term mortality and readmission...")
    icu_stays = calculate_short_term_mortality(icu_stays)
    icu_stays = calculate_readmission(icu_stays)

    # Extract the first ICU stay for each patient
    print("Extracting first ICU stay for each patient...")
    first_icu_stays = icu_stays.groupby('subject_id').first().reset_index()

    # Merge NOTEEVENTS with first ICU stays
    combined_data = noteevents.merge(first_icu_stays, on=['subject_id', 'hadm_id'], how='inner')

    # Concatenate all notes for each patient's first ICU stay
    combined_data['note'] = combined_data.groupby('subject_id')['TEXT'].transform(lambda x: ' '.join(x))

    # Deduplicate to ensure one row per patient
    final_data = combined_data.drop_duplicates(subset=['subject_id']).copy()

    # Split long notes into multiple columns
    print("Splitting long notes into multiple columns...")
    final_data = process_notes(final_data, note_column='note', max_tokens=max_tokens)

    # Save the final data
    print(f"Saving preprocessed data to {output_file}...")
    final_data.to_csv(output_file, index=False)
    print("Preprocessing complete.")

# Usage example with test files
noteevents_file = "NOTEEVENTS.csv.gz"
icustays_file = "ICUSTAYS.csv.gz"
patients_file = "PATIENTS.csv.gz"
admissions_file = "ADMISSIONS.csv.gz"
output_file = "Unstructured.csv"

# Run preprocessing
preprocess_icu_data(noteevents_file, icustays_file, patients_file, admissions_file, output_file, max_tokens=500)

# Analyze the processed data
processed_data = pd.read_csv(output_file)
print("Processed Data Shape:", processed_data.shape)
print(processed_data[['gender', 'age_bucket', 'categorized_ethnicity', 'categorized_insurance']].head())

import pandas as pd

# Reload the processed dataset
processed_data = pd.read_csv("Unstructured.csv")

# Get the shape of the dataset
dataset_shape = processed_data.shape

# Calculate positive short-term mortality and readmissions within 30 days
positive_mortality_count = processed_data['short_term_mortality'].sum()
readmission_count = processed_data['readmission_within_30_days'].sum()

dataset_shape, positive_mortality_count, readmission_count
## Please check the shape and positive numbers and let me know:)
