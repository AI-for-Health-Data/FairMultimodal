import pandas as pd

# Load MIMIC-III data
admissions = pd.read_csv('ADMISSIONS.csv.gz', compression='gzip')
patients = pd.read_csv('PATIENTS.csv.gz', compression='gzip')
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip')
chartevents = pd.read_csv('CHARTEVENTS.csv.gz', compression='gzip', usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE'])

# Convert relevant columns to datetime format
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
icu_stays['INTIME'] = pd.to_datetime(icu_stays['INTIME'])
chartevents['CHARTTIME'] = pd.to_datetime(chartevents['CHARTTIME'])

# Rename columns for consistency
admissions.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
icu_stays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
chartevents.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)

# Merge patients and admissions tables to get patient and admission details
merged_data = pd.merge(admissions, patients, on='subject_id', how='inner')

# Calculate short-term mortality (death within 30 days of discharge)
merged_data['short_term_mortality'] = ((pd.to_datetime(merged_data['DEATHTIME']) - pd.to_datetime(merged_data['DISCHTIME'])).dt.days <= 30).astype(int)

# Calculate readmission within 30 days
merged_data = merged_data.sort_values(by=['subject_id', 'ADMITTIME'])
merged_data['readmitted_within_30_days'] = (merged_data.groupby('subject_id')['ADMITTIME'].diff().dt.days <= 30).astype(int)

# Define ITEMIDs for mechanical ventilation (common set from academic papers)
mechanical_ventilation_ids = [720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684, 
                              224685, 224684, 224686, 218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747, 543, 5865, 
                              5866, 224707, 224709, 224705, 224706, 60, 437, 505, 506, 686, 220339, 224700, 3459, 501, 502, 
                              503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701]

# Filter CHARTEVENTS based on these ITEMIDs
mechanical_ventilation = chartevents[chartevents['ITEMID'].isin(mechanical_ventilation_ids)].copy()

# Add a column to indicate mechanical ventilation presence
mechanical_ventilation['mechanical_ventilation'] = 1

# Drop duplicates to avoid multiple records for the same admission
mechanical_ventilation = mechanical_ventilation[['subject_id', 'hadm_id', 'mechanical_ventilation', 'CHARTTIME']].drop_duplicates()

# Merge the mechanical ventilation data with ICU stay data to get admission times
mechanical_ventilation = pd.merge(mechanical_ventilation, icu_stays[['subject_id', 'hadm_id', 'ICUSTAY_ID', 'INTIME']], on=['subject_id', 'hadm_id'], how='left')

# Calculate time difference between ventilation and ICU admission (in hours)
mechanical_ventilation['time_since_admission'] = (mechanical_ventilation['CHARTTIME'] - mechanical_ventilation['INTIME']).dt.total_seconds() / 3600

# Create a new column to indicate ventilation within 6 hours
mechanical_ventilation['ventilation_within_6_hours'] = (mechanical_ventilation['time_since_admission'] <= 6).astype(int)

# Drop duplicates to ensure one row per admission
mechanical_ventilation = mechanical_ventilation[['subject_id', 'hadm_id', 'ventilation_within_6_hours']].drop_duplicates()

# Merge the result back with the merged_data to get ventilation within 6 hours for each patient
merged_data = pd.merge(merged_data, mechanical_ventilation, on=['subject_id', 'hadm_id'], how='left')

# Fill NaN values with 0 (for patients with no ventilation within 6 hours)
merged_data['ventilation_within_6_hours'].fillna(0, inplace=True)

# Convert to integer type
merged_data['ventilation_within_6_hours'] = merged_data['ventilation_within_6_hours'].astype(int)

# Extract the first admission of each patient
first_admissions = merged_data.sort_values(by=['subject_id', 'ADMITTIME']).groupby('subject_id').first().reset_index()

# Update the columns to reflect any positive case for each patient across all admissions
# Using transform to propagate the maximum of any occurrence of short-term mortality, readmission, and ventilation
first_admissions['short_term_mortality'] = merged_data.groupby('subject_id')['short_term_mortality'].transform('max')
first_admissions['readmitted_within_30_days'] = merged_data.groupby('subject_id')['readmitted_within_30_days'].transform('max')
first_admissions['ventilation_within_6_hours'] = merged_data.groupby('subject_id')['ventilation_within_6_hours'].transform('max')

# Ensure that we only keep the first admission per patient
first_admissions = first_admissions.drop_duplicates(subset='subject_id', keep='first').reset_index(drop=True)

# Calculate age
def calculate_age(dob, admittime):
    return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))

first_admissions['age'] = first_admissions.apply(lambda x: calculate_age(pd.to_datetime(x['DOB']), x['ADMITTIME']), axis=1)
first_admissions = first_admissions[(first_admissions['age'] >= 15) & (first_admissions['age'] <= 90)]

# Categorize age
def categorize_age(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    else:
        return '70-89'

first_admissions['age_bucket'] = first_admissions['age'].apply(categorize_age)

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
        return 'Other'  # Group 'Native', 'Unknown', and others into 'Other'

first_admissions['categorized_ethnicity'] = first_admissions['ETHNICITY'].apply(categorize_ethnicity)

# Categorize insurance
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

first_admissions['categorized_insurance'] = first_admissions['INSURANCE'].apply(categorize_insurance)

# One-hot encoding for categorical columns
first_admissions = pd.get_dummies(
    first_admissions, 
    columns=['age_bucket', 'categorized_ethnicity', 'categorized_insurance'], 
    drop_first=False
)

# Display dataset shape and key outcomes
print(f"Dataset Shape: {first_admissions.shape}")
print(f"Positive Short-Term Mortality Cases: {first_admissions['short_term_mortality'].sum()}")
print(f"Positive Readmission Cases: {first_admissions['readmitted_within_30_days'].sum()}")
print(f"Positive Mechanical Ventilation Cases: {first_admissions['ventilation_within_6_hours'].sum()}")

# Save structured data
first_admissions.to_csv('structured_first_admissions.csv', index=False)

print(first_admissions.columns)

# Total count of records
total_count = first_admissions.shape[0]

# Ethnicity summary
ethnicity_columns = [
    'categorized_ethnicity_Black',
    'categorized_ethnicity_Hispanic',
    'categorized_ethnicity_Other',
    'categorized_ethnicity_White'
]
ethnicity_summary = first_admissions[ethnicity_columns].sum().reset_index()
ethnicity_summary.columns = ['Ethnicity', 'Count']
ethnicity_summary['Ethnicity'] = ethnicity_summary['Ethnicity'].str.replace('categorized_ethnicity_', '')
ethnicity_summary['Percentage'] = (ethnicity_summary['Count'] / total_count) * 100

# Insurance summary
insurance_columns = [
    'categorized_insurance_Medicaid',
    'categorized_insurance_Medicare',
    'categorized_insurance_Private',
    'categorized_insurance_Self Pay'
]
insurance_summary = first_admissions[insurance_columns].sum().reset_index()
insurance_summary.columns = ['Insurance', 'Count']
insurance_summary['Insurance'] = insurance_summary['Insurance'].str.replace('categorized_insurance_', '')
insurance_summary['Percentage'] = (insurance_summary['Count'] / total_count) * 100

# Gender summary
gender_summary = first_admissions['GENDER'].value_counts().reset_index()
gender_summary.columns = ['Gender', 'Count']
gender_summary['Percentage'] = (gender_summary['Count'] / total_count) * 100

# Age bucket summary
age_summary = first_admissions[['age_bucket_15-29', 'age_bucket_30-49', 'age_bucket_50-69', 'age_bucket_70-89']].sum().reset_index()
age_summary.columns = ['Age Bucket', 'Count']
age_summary['Age Bucket'] = age_summary['Age Bucket'].str.replace('age_bucket_', '')
age_summary['Percentage'] = (age_summary['Count'] / total_count) * 100

# Display summaries
print("Ethnicity Summary:")
print(ethnicity_summary)

print("\nInsurance Summary:")
print(insurance_summary)

print("\nGender Summary:")
print(gender_summary)

print("\nAge Bucket Summary:")
print(age_summary)

import pandas as pd

# Reverse one-hot encoding for ethnicity, insurance, and age bucket
def reverse_one_hot(df, prefix, new_column_name):
    cols = [col for col in df.columns if col.startswith(prefix)]
    df[new_column_name] = df[cols].idxmax(axis=1).str.replace(f'{prefix}_', '', regex=False)
    return df

# Reverse one-hot encoding for ethnicity, insurance, and age bucket
first_admissions = reverse_one_hot(first_admissions, 'categorized_ethnicity', 'categorized_ethnicity')
first_admissions = reverse_one_hot(first_admissions, 'categorized_insurance', 'categorized_insurance')
first_admissions = reverse_one_hot(first_admissions, 'age_bucket', 'age_bucket')

# Function to create summary table for a given column (e.g., ethnicity, insurance, or age bucket)
def create_summary_table(column_name, gender_column='GENDER'):
    # Group by the specified column and gender
    summary = first_admissions.groupby([column_name, gender_column]).size().unstack(fill_value=0)
    summary['Total'] = summary.sum(axis=1)  # Calculate total for each category
    summary['Percentage'] = (summary['Total'] / first_admissions.shape[0]) * 100  # Calculate percentage
    return summary

# Ethnicity summary by gender
ethnicity_summary = create_summary_table('categorized_ethnicity')

# Insurance summary by gender
insurance_summary = create_summary_table('categorized_insurance')

# Age bucket summary by gender
age_bucket_summary = create_summary_table('age_bucket')

# Display summaries
print("Ethnicity Summary by Gender:")
print(ethnicity_summary)

print("\nInsurance Summary by Gender:")
print(insurance_summary)

print("\nAge Bucket Summary by Gender:")
print(age_bucket_summary)

# Save summaries to CSV
ethnicity_summary.to_csv('ethnicity_summary_by_gender.csv', index=True)
insurance_summary.to_csv('insurance_summary_by_gender.csv', index=True)
age_bucket_summary.to_csv('age_bucket_summary_by_gender.csv', index=True)

print("\nSummary tables have been saved as CSV files.")

