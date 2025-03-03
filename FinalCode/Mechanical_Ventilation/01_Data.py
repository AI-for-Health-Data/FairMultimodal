import pandas as pd
import numpy as np
import re

# 1. MECHANICAL VENTILATION FLAGS EXTRACTION
# Load CHARTEVENTS with relevant columns for ventilation signals
chartevents = pd.read_csv('CHARTEVENTS.csv.gz', compression='gzip', low_memory=False,
                           usecols=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'ERROR'])
chartevents.columns = chartevents.columns.str.lower()

# Keep only rows with non-null values and where error is not 1 (or is null)
chartevents = chartevents[chartevents['value'].notnull()]
chartevents = chartevents[(chartevents['error'] != 1) | (chartevents['error'].isnull())]

# Define itemids for ventilation settings (both for ventilation and oxygen devices)
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
    # Oxygen device related itemids
    468, 469, 470, 471, 227287, 226732, 223834
]
chartevents = chartevents[chartevents['itemid'].isin(vent_itemids)]

def determine_flags(row):
    """Determine ventilation-related flags for a given row."""
    mechvent = 0
    oxygen = 0
    extubated = 0
    self_extubated = 0

    iv = row['itemid']
    val = row['value']

    # Mechanical Ventilation conditions:
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

    # Oxygen Therapy conditions:
    if iv == 226732 and val in ['Nasal cannula', 'Face tent', 'Aerosol-cool', 'Trach mask ', 
                                'High flow neb', 'Non-rebreather', 'Venti mask ', 'Medium conc mask ',
                                'T-piece', 'High flow nasal cannula', 'Ultrasonic neb', 'Vapomist']:
        oxygen = 1
    if iv == 467 and val in ['Cannula', 'Nasal Cannula', 'Face Tent', 'Aerosol-Cool', 'Trach Mask',
                              'Hi Flow Neb', 'Non-Rebreather', 'Venti Mask', 'Medium Conc Mask',
                              'Vapotherm', 'T-Piece', 'Hood', 'Hut', 'TranstrachealCat',
                              'Heated Neb', 'Ultrasonic Neb']:
        oxygen = 1

    # Extubation conditions:
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

# Apply the flag function to each row in chartevents
vent_flags = chartevents.apply(determine_flags, axis=1)
chartevents = pd.concat([chartevents, vent_flags], axis=1)

# Aggregate ventilation flags by ICU stay and charttime 
vent_chartevents = chartevents.groupby(['icustay_id', 'charttime'], as_index=False).agg({
    'mechvent': 'max',
    'oxygentherapy': 'max',
    'extubated': 'max',
    'selfextubated': 'max'
})

# Process procedureevents_mv to capture extubation events
proc_events = pd.read_csv('PROCEDUREEVENTS_MV.csv.gz', compression='gzip', low_memory=False,
                          usecols=['ICUSTAY_ID', 'STARTTIME', 'ITEMID'])
proc_events.columns = proc_events.columns.str.lower()
proc_events = proc_events[proc_events['itemid'].isin([227194, 225468, 225477])]
proc_events.rename(columns={'starttime': 'charttime'}, inplace=True)
proc_events['mechvent'] = 0
proc_events['oxygentherapy'] = 0
proc_events['extubated'] = 1
proc_events['selfextubated'] = proc_events['itemid'].apply(lambda x: 1 if x == 225468 else 0)
vent_proc = proc_events[['icustay_id', 'charttime', 'mechvent', 'oxygentherapy', 'extubated', 'selfextubated']].drop_duplicates()

# Combine both sources
ventilation_flags = pd.concat([vent_chartevents, vent_proc], ignore_index=True).drop_duplicates(subset=['icustay_id', 'charttime'])

# Map ventilation flags to subject and admission IDs using ICUSTAYS
icu_stays = pd.read_csv('ICUSTAYS.csv.gz', compression='gzip', usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
icu_stays.columns = icu_stays.columns.str.lower()
ventilation_flags = ventilation_flags.merge(
    icu_stays[['icustay_id', 'subject_id', 'hadm_id']],
    on='icustay_id',
    how='left'
)

# Aggregate ventilation flags per subject and admission (max flag value over all charttimes)
ventilation_flags_agg = ventilation_flags.groupby(['subject_id', 'hadm_id'], as_index=False).agg({
    'mechvent': 'max',
    'oxygentherapy': 'max',
    'extubated': 'max',
    'selfextubated': 'max'
})

# Create a single mechanical ventilation column:
# If any of the flags is 1, set mechanical_ventilation to 1, else 0.
ventilation_flags_agg['mechanical_ventilation'] = ventilation_flags_agg[['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']].max(axis=1)


# 2. MERGE INTO STRUCTURED DATASET
# Load your final structured dataset (before ventilation flags merge)
structured_df = pd.read_csv('final_structured_common.csv')

# Merge the ventilation flag aggregation (only subject_id and hadm_id keys)
structured_with_vent = structured_df.merge(ventilation_flags_agg[['subject_id', 'hadm_id', 'mechanical_ventilation']],
                                             on=['subject_id', 'hadm_id'], how='left')
structured_with_vent['mechanical_ventilation'] = structured_with_vent['mechanical_ventilation'].fillna(0).astype(int)

# Drop mortality and readmission columns if they exist
cols_to_drop = ['short_term_mortality', 'readmission_within_30_days']
for col in cols_to_drop:
    if col in structured_with_vent.columns:
        structured_with_vent.drop(columns=[col], inplace=True)

# Drop any ventilation flag columns if present; keep only the single column:
for col in ['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']:
    if col in structured_with_vent.columns:
        structured_with_vent.drop(columns=[col], inplace=True)

# Save the updated structured dataset
structured_outfile = 'final_structured_with_mechanical_ventilation.csv'
structured_with_vent.to_csv(structured_outfile, index=False)
print("Updated structured dataset saved as:", structured_outfile)

# 3. MERGE INTO UNSTRUCTURED DATASET
# Load your final unstructured dataset (before ventilation flags merge)
unstructured_df = pd.read_csv('final_unstructured_common.csv', engine='python', on_bad_lines='skip')

# Merge in the ventilation flag aggregation
unstructured_with_vent = unstructured_df.merge(ventilation_flags_agg[['subject_id', 'hadm_id', 'mechanical_ventilation']],
                                               on=['subject_id', 'hadm_id'], how='left')
unstructured_with_vent['mechanical_ventilation'] = unstructured_with_vent['mechanical_ventilation'].fillna(0).astype(int)

# Drop mortality and readmission columns if they exist
for col in cols_to_drop:
    if col in unstructured_with_vent.columns:
        unstructured_with_vent.drop(columns=[col], inplace=True)

# Drop any individual ventilation flag columns if present
for col in ['mechvent', 'oxygentherapy', 'extubated', 'selfextubated']:
    if col in unstructured_with_vent.columns:
        unstructured_with_vent.drop(columns=[col], inplace=True)

# Save the updated unstructured dataset
unstructured_outfile = 'final_unstructured_with_mechanical_ventilation.csv'
unstructured_with_vent.to_csv(unstructured_outfile, index=False)
print("Updated unstructured dataset saved as:", unstructured_outfile)
