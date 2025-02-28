import os
import pandas as pd
import numpy as np

# Process CHARTEVENTS to Identify Mechanical Ventilation
columns_needed = [
    "ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID",
    "CHARTTIME", "STORETIME", "CGID", "VALUE", "VALUENUM", "VALUEUOM",
    "WARNING", "ERROR", "RESULTSTATUS", "STOPPED"
]

chartevents_file = 'CHARTEVENTS.csv.gz'  # Adjust the path if needed
print("Reading CHARTEVENTS file...")
chartevents = pd.read_csv(chartevents_file, compression='gzip', usecols=columns_needed, low_memory=False)
print("CHARTEVENTS shape after reading:", chartevents.shape)

# 2. Preprocess CHARTEVENTS
# Keep only rows with non-null VALUE and where ERROR is not flagged (ERROR != 1)
chartevents = chartevents[chartevents["VALUE"].notnull()]
chartevents = chartevents[(chartevents["ERROR"].isnull()) | (chartevents["ERROR"] != 1)]
print("Shape after filtering nulls and errors:", chartevents.shape)

# Convert CHARTTIME to datetime and drop rows with invalid dates
chartevents["CHARTTIME"] = pd.to_datetime(chartevents["CHARTTIME"], errors='coerce')
chartevents = chartevents.dropna(subset=["CHARTTIME"])

# Standardize column names to lowercase for consistency
chartevents.columns = chartevents.columns.str.lower()

# 3. Flag Ventilation Events
def flag_mech_vent(row):
    try:
        itemid = int(row["itemid"])
        value = str(row["value"]).strip().lower()
    except Exception:
        return 0

    # Apply logic similar to your SQL case statements:
    if itemid == 720 and value != "other/remarks":
        return 1
    if itemid == 223848 and value != "other":
        return 1
    if itemid == 223849:
        return 1
    if itemid == 467 and value == "ventilator":
        return 1

    # A list of additional itemids indicating ventilation settings (minute volume, tidal volume, pressures, etc.)
    vent_itemids = [
        445, 448, 449, 450, 1340, 1486, 1600, 224687,
        639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
        218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747,
        221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
        543, 5865, 5866, 224707, 224709, 224705, 224706,
        60, 437, 505, 506, 686, 220339, 224700,
        3459, 501, 502, 503, 224702,
        223, 667, 668, 669, 670, 671, 672,
        224701
    ]
    if itemid in vent_itemids:
        return 1
    return 0

# Apply the function to create a new column "mechvent"
chartevents["mechvent"] = chartevents.apply(flag_mech_vent, axis=1)
print("Unique values in 'mechvent':", chartevents["mechvent"].unique())

# 4. Aggregate Ventilation Events to Compute Duration and Flag Mechanical Ventilation
# Select only rows flagged as ventilation events
vent_events = chartevents[chartevents["mechvent"] == 1].copy()

# Group by ICU stay and compute the minimum and maximum charttime for each stay
vent_agg = vent_events.groupby("icustay_id")["charttime"].agg(["min", "max"]).reset_index()

# Compute duration in hours
vent_agg["duration_hours"] = (vent_agg["max"] - vent_agg["min"]).dt.total_seconds() / 3600

# Label an ICU stay as mechanically ventilated if the duration is >= 6 hours
vent_agg["mechanical_ventilation"] = (vent_agg["duration_hours"] >= 6).astype(int)

# Keep only the icustay_id and the ventilation flag
ventilation_df = vent_agg[["icustay_id", "mechanical_ventilation"]]
print("Ventilation events aggregated shape:", ventilation_df.shape)

# Process Demographics (ADMISSIONS, PATIENTS, ICUSTAYS)
print("Reading ADMISSIONS, PATIENTS, and ICUSTAYS files...")

# Load the MIMIC-III Datasets
admissions = pd.read_csv('ADMISSIONS.csv.gz', compression='gzip')
patients = pd.read_csv('PATIENTS.csv.gz', compression='gzip')
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip')

# Convert Relevant Columns to Datetime
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])
icu_stays['INTIME'] = pd.to_datetime(icu_stays['INTIME'])
icu_stays['OUTTIME'] = pd.to_datetime(icu_stays['OUTTIME'])

# Rename Columns for Consistency
admissions.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
patients.rename(columns={'SUBJECT_ID': 'subject_id'}, inplace=True)
# Ensure icustay id is renamed as well
icu_stays.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id', 'ICUSTAY_ID': 'icustay_id'}, inplace=True)

# Merge ICU Stays with Admissions and Patients to Get Demographics
icu_stays = pd.merge(
    icu_stays,
    admissions[['subject_id', 'hadm_id', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'INSURANCE']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

icu_stays = pd.merge(icu_stays, patients[['subject_id', 'GENDER', 'DOB']], on='subject_id', how='left')

# For many studies the first ICU stay is used, so we select the first ICU stay per subject:
first_icu_stays = icu_stays.sort_values('INTIME').drop_duplicates(subset=['subject_id'], keep='first')

# Calculate Age at ICU Admission and Filter by Age
def calculate_age(dob, intime):
    return intime.year - dob.year - ((intime.month, intime.day) < (dob.month, dob.day))

first_icu_stays['age'] = first_icu_stays.apply(lambda x: calculate_age(pd.to_datetime(x['DOB']), x['INTIME']), axis=1)
first_icu_stays = first_icu_stays[(first_icu_stays['age'] >= 15) & (first_icu_stays['age'] <= 90)]

# Categorize Age into Buckets
def categorize_age(age):
    if 15 <= age <= 29:
        return '15-29'
    elif 30 <= age <= 49:
        return '30-49'
    elif 50 <= age <= 69:
        return '50-69'
    else:
        return '70-89'

first_icu_stays['age_bucket'] = first_icu_stays['age'].apply(categorize_age)

# Categorize Ethnicity
def categorize_ethnicity(ethnicity):
    ethnicity = str(ethnicity).upper()
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

first_icu_stays['categorized_ethnicity'] = first_icu_stays['ETHNICITY'].apply(categorize_ethnicity)

# Categorize Insurance
def categorize_insurance(insurance):
    insurance = str(insurance)
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

first_icu_stays['categorized_insurance'] = first_icu_stays['INSURANCE'].apply(categorize_insurance)

# One-Hot Encode the Categorical Variables 
first_icu_stays = pd.get_dummies(
    first_icu_stays,
    columns=['age_bucket', 'categorized_ethnicity', 'categorized_insurance'],
    drop_first=False
)

# Merge Ventilation Data with Demographics to Create the Final Structured Dataset
# Merge on icustay_id; note that some ICU stays may not have ventilation events (fill with 0)
final_df = pd.merge(first_icu_stays, ventilation_df, on='icustay_id', how='left')
final_df['mechanical_ventilation'] = final_df['mechanical_ventilation'].fillna(0).astype(int)

# Print shape of final dataset
print("Final dataset shape:", final_df.shape)

# Print the number of ICU stays with positive mechanical ventilation (flag == 1)
num_ventilated = final_df['mechanical_ventilation'].sum()
print("Number of ICU stays with mechanical ventilation:", num_ventilated)

# Save the final structured dataset
output_file = "structured_dataset.csv"
final_df.to_csv(output_file, index=False)
print(f"Final structured dataset saved as '{output_file}'")
