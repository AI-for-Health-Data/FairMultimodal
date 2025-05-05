import os
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast


admissions = pd.read_csv(
    os.path.join("admissions.csv.gz"),
    compression='gzip',
    usecols=['subject_id','hadm_id','admittime','dischtime','deathtime','insurance','race'],
    parse_dates=['admittime','dischtime','deathtime']
)

patients = pd.read_csv(
    os.path.join("patients.csv.gz"),
    compression='gzip',
    usecols=['subject_id','gender','anchor_age','dod'],
    parse_dates=['dod']
)

icustays = pd.read_csv(
    os.path.join("icustays.csv.gz"),
    compression='gzip',
    usecols=['subject_id','hadm_id','stay_id','intime','outtime'],
    parse_dates=['intime','outtime']
)

labevents = pd.read_csv(
    os.path.join("labevents.csv.gz"),
    compression='gzip',
    usecols=['subject_id','hadm_id','itemid','charttime','valuenum'],
    parse_dates=['charttime']
)

discharge = pd.read_csv(
    os.path.join("discharge.csv.gz"),
    compression='gzip',
    parse_dates=['charttime']
).rename(columns={
    'hadm_id':'stay_id',
    'charttime':'note_time',
    'text':'discharge_text'
})

icustays = icustays.sort_values(['subject_id','intime'])
first_icustays = (
    icustays
    .drop_duplicates(subset='subject_id', keep='first')
    .rename(columns={'stay_id':'icustay_id'})
    .reset_index(drop=True)
)[['subject_id','hadm_id','icustay_id','intime','outtime']]

demo = (
    first_icustays[['subject_id','hadm_id','outtime']]
    .merge(admissions[['subject_id','hadm_id','insurance','race']],
           on=['subject_id','hadm_id'], how='left')
    .merge(patients[['subject_id','gender','anchor_age']],
           on='subject_id', how='left')
)
demo = demo[demo.anchor_age.between(18, 90)].copy()

all_stays = icustays.merge(
    first_icustays[['subject_id','outtime']].rename(columns={'outtime':'first_out'}),
    on='subject_id', how='left'
)
all_stays['readmit_within30d'] = (
    (all_stays.intime > all_stays.first_out) &
    (all_stays.intime <= all_stays.first_out + pd.Timedelta(days=30))
)
readm = (
    all_stays.groupby('subject_id')['readmit_within30d']
    .any().astype(int)
    .rename('readmission_30d')
    .reset_index()
)
demo = demo.merge(readm, on='subject_id', how='inner')

valid_subjects = demo.subject_id.unique()


win = first_icustays[first_icustays.subject_id.isin(valid_subjects)][
    ['subject_id','icustay_id','outtime']
].copy()
win['start24h'] = win.outtime - pd.Timedelta(hours=24)

labs = (
    labevents
    .merge(win[['subject_id','start24h','outtime']], on='subject_id', how='inner')
)
labs = labs[(labs.charttime >= labs.start24h) & (labs.charttime <= labs.outtime)]
labs['hrs_from_start'] = ((labs.charttime - labs.start24h).dt.total_seconds() // 3600).astype(int)
labs['bin2h'] = (labs.hrs_from_start // 2) * 2

lab_agg = (
    labs.groupby(['subject_id','bin2h','itemid'])['valuenum']
    .mean().unstack(fill_value=np.nan).reset_index()
)

structured = (
    first_icustays[first_icustays.subject_id.isin(valid_subjects)][['subject_id','icustay_id']]
    .merge(demo, on='subject_id', how='left')
    .merge(lab_agg, on='subject_id', how='left')
)

structured.to_csv(os.path.join('mimiciv_structured_dataset.csv'), index=False)
print("Structured shape:", structured.shape)
print("Structured positive readmissions:", structured.readmission_30d.sum())


first_windows = win[['subject_id','start24h','outtime']].copy()

notes = discharge.merge(first_windows, on='subject_id', how='inner')
notes = notes[(notes.note_time >= notes.start24h) & (notes.note_time <= notes.outtime)]
notes = notes[notes.subject_id.isin(valid_subjects)].copy()

notes_concat = (
    notes.sort_values(['subject_id','note_time'])
         .groupby('subject_id')['discharge_text']
         .apply(lambda texts: "\n\n".join(texts))
         .reset_index()
)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
def truncate_to_512(txt):
    toks = tokenizer.encode(txt, add_special_tokens=True, truncation=True, max_length=512)
    return tokenizer.decode(toks, clean_up_tokenization_spaces=True)

notes_concat['text_512'] = notes_concat.discharge_text.apply(truncate_to_512)

unstructured = pd.DataFrame({'subject_id': valid_subjects})
unstructured = unstructured.merge(notes_concat[['subject_id','text_512']], on='subject_id', how='left')
unstructured['text_512'] = unstructured['text_512'].fillna("")

unstructured = unstructured.merge(
    demo[['subject_id','gender','anchor_age','insurance','race','readmission_30d']],
    on='subject_id', how='left'
)

unstructured.to_csv(os.path.join('mimiciv_unstructured_notes.csv'), index=False)
print("Unstructured shape:", unstructured.shape)
print("Unstructured positive readmissions:", unstructured.readmission_30d.sum())
