import pandas as pd
import os
import spacy

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

# Function to calculate short-term mortality for ICU stays using ICUSTAY_ID
def calculate_short_term_mortality(icu_stays):
    # Mark as 1 if DEATHTIME exists within the ICU stay, otherwise 0
    icu_stays['short_term_mortality'] = icu_stays['DEATHTIME'].notnull().astype(int)
    return icu_stays

# Function to calculate short-term mortality
def calculate_short_term_mortality(icu_stays):
    # Mark as 1 if DEATHTIME exists within the ICU stay, otherwise 0
    icu_stays['short_term_mortality'] = icu_stays['DEATHTIME'].notnull().astype(int)
    return icu_stays

# Function to calculate readmission within 30 days considering all ICU stays
def calculate_readmission(icu_stays):
    # Sort by subject and ICU admission time
    icu_stays = icu_stays.sort_values(by=['subject_id', 'ICUSTAY_ID'])
    
    # Calculate the difference between OUTTIME of the current ICU stay and INTIME of the next ICU stay
    icu_stays['time_diff'] = (
        icu_stays.groupby('subject_id')['INTIME']
        .shift(-1) - icu_stays['OUTTIME']
    ).dt.days
    
    # Mark readmission within 30 days
    icu_stays['readmitted_within_30_days'] = (
        (icu_stays['time_diff'] <= 30) & (icu_stays['time_diff'] > 0)
    ).astype(int)
    
    # Fill NaN with 0 for patients with only one ICU stay
    icu_stays['readmitted_within_30_days'] = icu_stays['readmitted_within_30_days'].fillna(0).astype(int)
    
    return icu_stays

# Function to split extremely long text manually
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

    # Find the maximum number of chunks across all rows
    max_chunks = data['note_chunks'].apply(len).max()

    # Dynamically create columns for each chunk
    for i in range(max_chunks):
        col_name = f'note_{i + 1}'
        data[col_name] = data['note_chunks'].apply(lambda x: x[i] if i < len(x) else None)

    # Drop the original note and intermediate chunk column
    data = data.drop(columns=[note_column, 'note_chunks'])
    return data

# Updated Preprocessing Function
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
    noteevents = pd.read_csv(noteevents_file, compression='gzip')
    noteevents.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    noteevents['TEXT'] = noteevents['TEXT'].fillna('')

    # Load ICUSTAYS data
    print(f"Loading data from {icustays_file}...")
    icu_stays = pd.read_csv(icustays_file, compression='gzip')
    icu_stays.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
    icu_stays['INTIME'] = pd.to_datetime(icu_stays['INTIME'])
    icu_stays['OUTTIME'] = pd.to_datetime(icu_stays['OUTTIME'])

    # Load ADMISSIONS data
    print(f"Loading data from {admissions_file}...")
    admissions = pd.read_csv(admissions_file, compression='gzip')
    admissions.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])

    # Merge ICUSTAYS with ADMISSIONS to include HADM_ID and DEATHTIME
    icu_stays = pd.merge(icu_stays, admissions[['subject_id', 'hadm_id', 'DEATHTIME']], on='subject_id', how='left')

    # Load PATIENTS data
    print(f"Loading data from {patients_file}...")
    patients = pd.read_csv(patients_file, compression='gzip')
    patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
    patients['DOB'] = pd.to_datetime(patients['DOB'])

    # Merge ICUSTAYS with PATIENTS to calculate age
    icu_stays = pd.merge(icu_stays, patients[['subject_id', 'DOB']], on='subject_id', how='left')
    icu_stays['age'] = icu_stays.apply(lambda x: calculate_age(x['DOB'], x['INTIME']), axis=1)
    icu_stays = icu_stays[(icu_stays['age'] >= 15) & (icu_stays['age'] <= 90)]
    icu_stays['age_bucket'] = icu_stays['age'].apply(categorize_age)

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

# Paths to input and output files
noteevents_file = "NOTEEVENTS.csv.gz"
icustays_file = "ICUSTAYS.csv.gz"
patients_file = "PATIENTS.csv.gz"
admissions_file = "ADMISSIONS.csv.gz"
output_file = "processed_icu_notes.csv"

# Run the preprocessing function
preprocess_icu_data(noteevents_file, icustays_file, patients_file, admissions_file, output_file, max_tokens=500)
