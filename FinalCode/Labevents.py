import os
import pandas as pd
import numpy as np

# Define file paths 
filtered_structured_file = 'filtered_structured_first_icu_stays.csv'
labevents_file = 'LABEVENTS.csv.gz'
chartevents_file = 'CHARTEVENTS.csv.gz'
d_items_file = 'D_ITEMS.csv.gz'

# Check that the filtered structured file exists
if not os.path.exists(filtered_structured_file):
    raise FileNotFoundError(f"File not found: {filtered_structured_file}. "
                            "Please ensure the file exists in the working directory or update the file path.")

# Read the filtered structured dataset (one row per patient ICU stay)
filtered_structured_df = pd.read_csv(filtered_structured_file)

# Display the shape and columns 
print("Filtered Structured ICU Stays shape:", filtered_structured_df.shape)
print("Columns:", filtered_structured_df.columns.tolist())

# Extract the outcome information:
#   subject_id, hadm_id, ICUSTAY_ID, and short_term_mortality.
outcomes = filtered_structured_df[['subject_id', 'hadm_id', 'ICUSTAY_ID', 'short_term_mortality', 'readmission_within_30_days']]
print("\nOutcomes sample:")
print(outcomes.head())

# Read lab events, chart events, and d_items
labevents = pd.read_csv(labevents_file, compression='gzip')
chartevents = pd.read_csv(chartevents_file, compression='gzip')
d_items = pd.read_csv(d_items_file, compression='gzip')

# Convert CHARTTIME columns to datetime (for proper time-based operations)
labevents['CHARTTIME'] = pd.to_datetime(labevents['CHARTTIME'], errors='coerce')
chartevents['CHARTTIME'] = pd.to_datetime(chartevents['CHARTTIME'], errors='coerce')

# Define columns to keep from lab and chart events.
labevents_columns = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM']
chartevents_columns = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM']

# If ICUSTAY_ID exists in these datasets, include it.
if 'ICUSTAY_ID' in labevents.columns:
    labevents_columns.insert(2, 'ICUSTAY_ID')
if 'ICUSTAY_ID' in chartevents.columns:
    chartevents_columns.insert(2, 'ICUSTAY_ID')

# Keep only the necessary columns.
labevents = labevents[labevents_columns]
chartevents = chartevents[chartevents_columns]

# Ensure VALUENUM is numeric; any non-numeric values become NaN.
labevents['VALUENUM'] = pd.to_numeric(labevents['VALUENUM'], errors='coerce')
chartevents['VALUENUM'] = pd.to_numeric(chartevents['VALUENUM'], errors='coerce')

# Drop rows with missing VALUENUM or CHARTTIME (these are required for analysis)
labevents = labevents.dropna(subset=['VALUENUM', 'CHARTTIME'])
chartevents = chartevents.dropna(subset=['VALUENUM', 'CHARTTIME'])

# Fill missing VALUEUOM entries with 'unknown'
labevents['VALUEUOM'] = labevents['VALUEUOM'].fillna('unknown')
chartevents['VALUEUOM'] = chartevents['VALUEUOM'].fillna('unknown')

# Ensure ITEMID is of integer type
labevents['ITEMID'] = labevents['ITEMID'].astype(int)
chartevents['ITEMID'] = chartevents['ITEMID'].astype(int)
d_items['ITEMID'] = d_items['ITEMID'].astype(int)

# Merge d_items with the events to obtain human-readable labels and unit names.
d_items_filtered = d_items[['ITEMID', 'LABEL', 'UNITNAME']]
labevents = labevents.merge(d_items_filtered, on='ITEMID', how='left')
chartevents = chartevents.merge(d_items_filtered, on='ITEMID', how='left')

# For any missing labels or unit names, fill with defaults.
labevents['LABEL'] = labevents['LABEL'].fillna('unknown_label')
labevents['UNITNAME'] = labevents['UNITNAME'].fillna(labevents['VALUEUOM'])
chartevents['LABEL'] = chartevents['LABEL'].fillna('unknown_label')
chartevents['UNITNAME'] = chartevents['UNITNAME'].fillna(chartevents['VALUEUOM'])

# Combine lab and chart events into one DataFrame.
combined_events = pd.concat([labevents, chartevents], ignore_index=True)
print("\nCombined events shape:", combined_events.shape)

# Aggregate the lab values over the entire ICU stay.
# Group by SUBJECT_ID, HADM_ID, ICUSTAY_ID, LABEL, and UNITNAME; take the median of VALUENUM.
groupby_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LABEL', 'UNITNAME']
icu_medians = combined_events.groupby(groupby_cols)['VALUENUM'].median().reset_index()

# Pivot the table so that each unique lab item (combination of LABEL and UNITNAME) becomes its own column.
icu_pivot = icu_medians.pivot_table(
    index=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'],
    columns=['LABEL', 'UNITNAME'],
    values='VALUENUM'
).reset_index()

# Flatten the MultiIndex columns.
icu_pivot.columns = [
    '_'.join([str(c) for c in col if c]).strip('_') if isinstance(col, tuple) else col
    for col in icu_pivot.columns
]

# Rename key columns to match those in the outcomes DataFrame.
icu_pivot.rename(columns={'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}, inplace=True)
print("Aggregated lab items shape:", icu_pivot.shape)

# Merge outcomes with aggregated lab items on subject_id, hadm_id, and ICUSTAY_ID.
final_dataset = pd.merge(
    outcomes,
    icu_pivot,
    on=['subject_id', 'hadm_id', 'ICUSTAY_ID'],
    how='left'
)

print("\nFinal merged dataset shape:", final_dataset.shape)
print("Final dataset sample:")
print(final_dataset.head())

final_output_file = 'final_dataset_with_lab_and_mortality.csv'
final_dataset.to_csv(final_output_file, index=False)
print(f"\nFinal dataset saved as '{final_output_file}'.")
