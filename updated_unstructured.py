import pandas as pd
import os
import spacy

# Load SpaCy English tokenizer
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000 

# Function to calculate age
def calculate_age(dob, admittime):
    return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))

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
    if ethnicity in [
        'WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 
        'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN'
    ]:
        return 'White'
    elif ethnicity in [
        'BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 
        'BLACK/HAITIAN', 'BLACK/AFRICAN', 'CARIBBEAN ISLAND'
    ]:
        return 'Black'
    elif ethnicity in [
        'HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 
        'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - GUATEMALAN', 
        'HISPANIC/LATINO - CUBAN', 'HISPANIC/LATINO - SALVADORAN', 
        'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)', 
        'HISPANIC/LATINO - MEXICAN', 'HISPANIC/LATINO - COLOMBIAN', 
        'HISPANIC/LATINO - HONDURAN'
    ]:
        return 'Hispanic'
    elif ethnicity in [
        'ASIAN', 'ASIAN - CHINESE', 'ASIAN - ASIAN INDIAN', 
        'ASIAN - VIETNAMESE', 'ASIAN - FILIPINO', 'ASIAN - CAMBODIAN', 
        'ASIAN - OTHER', 'ASIAN - KOREAN', 'ASIAN - JAPANESE', 'ASIAN - THAI'
    ]:
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

# Function to preprocess data
def preprocess_data(noteevents_file, admissions_file, patients_file, output_file, max_tokens=500):
    if not os.path.exists(noteevents_file):
        raise FileNotFoundError(f"NOTEEVENTS file {noteevents_file} not found.")
    if not os.path.exists(admissions_file):
        raise FileNotFoundError(f"ADMISSIONS file {admissions_file} not found.")
    if not os.path.exists(patients_file):
        raise FileNotFoundError(f"PATIENTS file {patients_file} not found.")

    # Load NOTEEVENTS data
    print(f"Loading data from {noteevents_file}...")
    noteevents = pd.read_csv(noteevents_file, compression='gzip')
    noteevents.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    noteevents['TEXT'] = noteevents['TEXT'].fillna('')

    # Load ADMISSIONS data
    print(f"Loading data from {admissions_file}...")
    admissions = pd.read_csv(admissions_file, compression='gzip')
    admissions.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
    admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])

    # Load PATIENTS data
    print(f"Loading data from {patients_file}...")
    patients = pd.read_csv(patients_file, compression='gzip')
    patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
    patients['DOB'] = pd.to_datetime(patients['DOB'])

    # Merge PATIENTS and ADMISSIONS to calculate age
    admissions = pd.merge(admissions, patients[['subject_id', 'DOB']], on='subject_id', how='left')
    admissions['age'] = admissions.apply(lambda x: calculate_age(x['DOB'], x['ADMITTIME']), axis=1)
    admissions = admissions[(admissions['age'] >= 15) & (admissions['age'] <= 90)]
    admissions['age_bucket'] = admissions['age'].apply(categorize_age)

    # Categorize ethnicity and insurance
    admissions['categorized_ethnicity'] = admissions['ETHNICITY'].apply(categorize_ethnicity)
    admissions['categorized_insurance'] = admissions['INSURANCE'].apply(categorize_insurance)

    # Calculate short-term mortality
    admissions['short_term_mortality'] = (
        (admissions['DEATHTIME'] - admissions['DISCHTIME']).dt.days <= 30
    ).astype(int)

    # Calculate readmission within 30 days
    admissions = admissions.sort_values(by=['subject_id', 'ADMITTIME'])
    admissions['readmitted_within_30_days'] = (
        admissions.groupby('subject_id')['ADMITTIME'].diff().dt.days <= 30
    ).astype(int)
    admissions['readmitted_within_30_days'] = admissions.groupby('subject_id')['readmitted_within_30_days'].transform('max')

    # Update short-term mortality to reflect any positive case across all admissions
    admissions['short_term_mortality'] = admissions.groupby('subject_id')['short_term_mortality'].transform('max')

    # Extract the first admission for each patient
    first_admissions = admissions.groupby('subject_id').first().reset_index()

    # Merge NOTEEVENTS with first admissions
    combined_data = noteevents.merge(first_admissions, on=['subject_id', 'hadm_id'], how='inner')

    # Concatenate all notes for each patient's first admission
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
admissions_file = "ADMISSIONS.csv.gz"
patients_file = "PATIENTS.csv.gz"
output_file = "processed_notes.csv"

# Run the preprocessing function
preprocess_data(noteevents_file, admissions_file, patients_file, output_file, max_tokens=500)

# Load the processed data
output_file = "processed_notes.csv"
data = pd.read_csv(output_file)

# Verify the shape and columns of the output
data_shape = data.shape
data_columns = data.columns.tolist()

# Count positive cases for mortality and readmission
positive_mortality_count = data['short_term_mortality'].sum()
positive_readmission_count = data['readmitted_within_30_days'].sum()

data_shape, data_columns, positive_mortality_count, positive_readmission_count

# Identify columns containing note chunks
note_columns = [col for col in data.columns if col.startswith('note_')]

# Calculate the number of non-null note chunks for each patient
data['num_note_chunks'] = data[note_columns].notnull().sum(axis=1)

# Display the distribution of note chunks per patient
chunk_distribution = data['num_note_chunks'].value_counts().sort_index()

print("Distribution of number of note chunks per patient:")
print(chunk_distribution)

# Save the chunk information for further analysis if needed
data[['subject_id', 'num_note_chunks']].to_csv("note_chunks_per_patient.csv", index=False)

# Preview the results for a few patients
print(data[['subject_id', 'num_note_chunks']].head())
