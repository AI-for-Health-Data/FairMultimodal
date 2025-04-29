import os
import argparse
from datetime import timedelta
import pandas as pd

FILE_PATHS = {
    'icustays':   'icustays.csv.gz',
    'admissions': 'admissions.csv.gz',
    'patients':   'patients.csv.gz',
    'labevents':  'labevents.csv.gz',
    'd_labitems': 'd_labitems.csv.gz',
    'discharge':  'discharge.csv.gz',
    'radiology':  'radiology.csv.gz',
}
DATE_COLS = {
    'icustays':   ['intime','outtime'],
    'admissions': ['admittime','dischtime','deathtime'],
    'patients':   ['dod'],
    'labevents':  ['charttime'],
    'discharge':  ['charttime'],
    'radiology':  ['charttime'],
}

def map_race(r):
    s = str(r).upper()
    if s.startswith('WHITE'): return 'White'
    if 'BLACK' in s:       return 'Black'
    if s.startswith('ASIAN'): return 'Asian'
    if 'PACIFIC ISLANDER' in s: return 'Pacific Islander'
    if 'AMERICAN INDIAN' in s or 'ALASKA NATIVE' in s:
        return 'American Indian/Alaska Native'
    return 'Other'

def map_ethnicity(r):
    s = str(r).upper()
    return 'Hispanic/Latino' if ('HISPANIC' in s or 'LATINO' in s) else 'Non-Hispanic'

def map_insurance(i):
    s = str(i).upper()
    if 'MEDICARE' in s: return 'Medicare'
    if 'MEDICAID' in s: return 'Medicaid'
    if 'PRIVATE' in s:  return 'Private'
    if 'SELF-PAY' in s or 'SELF PAY' in s: return 'Self-pay'
    if 'NO CHARGE' in s: return 'No charge'
    return 'Other'

def map_age(a):
    if 18 <= a <= 29: return '18-29'
    if 30 <= a <= 49: return '30-49'
    if 50 <= a <= 69: return '50-69'
    if 70 <= a <= 89: return '70-89'
    return 'Other'

def chunk_text(text, max_tokens=512):
    toks = str(text).split()
    return [" ".join(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]

def chunk_notes(win_df, notes_df, prefix, hours=24):
    merged = notes_df.merge(win_df[['subject_id','time0']], on='subject_id', how='inner')
    if prefix == 'mort':
        # first 24h window after ICU admission
        mask = (
            (merged.charttime >= merged.time0) &
            (merged.charttime <= merged.time0 + timedelta(hours=hours))
        )
    else:
        # last 24h before ICU discharge, only Discharge summaries
        merged = merged[merged.category == 'Discharge summary']
        mask = (
            (merged.charttime >= merged.time0 - timedelta(hours=hours)) &
            (merged.charttime <= merged.time0)
        )
    rows = []
    for subj, grp in merged.loc[mask].groupby('subject_id'):
        chunks = []
        for txt in grp.text:
            chunks.extend(chunk_text(txt))
        if chunks:
            entry = {'subject_id': subj}
            for i, c in enumerate(chunks):
                entry[f"{prefix}_chunk_{i}"] = c
            rows.append(entry)
    return pd.DataFrame(rows)

def agg_labs(labevents, win_df, d_labitems, prefix, hours=24):
    df = pd.merge(labevents, win_df[['subject_id','stay_id','time0']],
                  on='subject_id', how='inner')
    if prefix == 'mort':
        mask = (
            (df.charttime >= df.time0) &
            (df.charttime <= df.time0 + timedelta(hours=hours))
        )
    else:
        mask = (
            (df.charttime >= df.time0 - timedelta(hours=hours)) &
            (df.charttime <= df.time0)
        )
    sel = df.loc[mask].copy()
    sel['value'] = pd.to_numeric(sel.value, errors='coerce')
    sel.dropna(subset=['value'], inplace=True)
    sel['window'] = ((sel.charttime - sel.time0).dt.total_seconds() // 7200).astype(int)
    sel = sel.merge(d_labitems[['itemid','label']], on='itemid', how='left')
    agg = sel.groupby(['subject_id','stay_id','label','window'])['value']\
             .median().reset_index()
    wide = agg.pivot(index=['subject_id','stay_id'],
                     columns=['label','window'], values='value')
    wide.columns = [
        f"{prefix}_lab_{lab}_w{int(w)*2}-{int(w)*2+2}h"
        for lab, w in wide.columns
    ]
    return wide.reset_index()

def print_summary(df, name):
    print(f"\n{name} — shape: {df.shape}")
    for task in ('mortality_icu','icu_readmission_30d'):
        if task in df.columns:
            pos = df[task].sum()
            pct = pos / len(df) * 100
            print(f"  {task}: {pos} ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    args = parser.parse_args()

    dfs = {
        k: pd.read_csv(
            os.path.join(args.data_dir, path),
            parse_dates=DATE_COLS.get(k, []),
            low_memory=False
        )
        for k, path in FILE_PATHS.items()
    }

    # Combine discharge & radiology notes
    notes_dr = pd.concat([
        dfs['discharge'].assign(category='Discharge summary'),
        dfs['radiology'].assign(category='Radiology')
    ], ignore_index=True)

    # first ICU stay
    patients = dfs['patients'].query("anchor_age.between(18,89)")
    adm = dfs['admissions']
    adm['race_group']      = adm.race.apply(map_race)
    adm['ethnicity_group'] = adm.race.apply(map_ethnicity)
    adm['insurance_group'] = adm.insurance.apply(map_insurance)

    icu = dfs['icustays']
    first_icu = (
        icu.sort_values('intime')
           .drop_duplicates('subject_id')
           .merge(adm[[
               'subject_id','hadm_id','admittime','dischtime',
               'deathtime','race_group','ethnicity_group',
               'insurance_group'
           ]], on=['subject_id','hadm_id'], how='left')
           .merge(patients[['subject_id','anchor_age','gender']],
                  on='subject_id', how='left')
    )
    first_icu['age_bucket'] = first_icu.anchor_age.apply(map_age)
    first_icu['gender']     = first_icu.gender.map({'M':'Male','F':'Female'}).fillna('Other')

    # ICU mortality: any non-null admission-level deathtime
    first_icu['mortality_icu'] = first_icu.deathtime.notna().astype(int)

    # ICU readmission within 30 days
    icu_all = dfs['icustays'][['subject_id','stay_id','intime']]
    bounds = first_icu[['subject_id','stay_id','outtime']]\
                .rename(columns={'stay_id':'first_stay','outtime':'first_out'})
    next_icus = icu_all.merge(bounds, on='subject_id', how='inner')\
                       .query("stay_id != first_stay")
    mask = (
        (next_icus.intime > next_icus.first_out) &
        (next_icus.intime <= next_icus.first_out + timedelta(days=30))
    )
    flag = next_icus.loc[mask]\
                    .groupby(['subject_id','first_stay'])\
                    .size().reset_index(name='count')
    flag['icu_readmission_30d'] = 1
    first_icu = first_icu.merge(
        flag[['subject_id','first_stay','icu_readmission_30d']],
        left_on=['subject_id','stay_id'],
        right_on=['subject_id','first_stay'],
        how='left'
    ).fillna({'icu_readmission_30d': 0})
    first_icu['icu_readmission_30d'] = first_icu.icu_readmission_30d.astype(int)

    mort_win  = first_icu[['subject_id','stay_id','intime']].rename(columns={'intime':'time0'})
    readm_win = first_icu[['subject_id','stay_id','outtime']].rename(columns={'outtime':'time0'})

    lab_mort  = agg_labs(dfs['labevents'], mort_win,  dfs['d_labitems'], 'mort')
    lab_readm = agg_labs(dfs['labevents'], readm_win, dfs['d_labitems'], 'readm')

    structured = (
        first_icu
        .merge(lab_mort,  on=['subject_id','stay_id'], how='left')
        .merge(lab_readm, on=['subject_id','stay_id'], how='left')
        .rename(columns={'anchor_age':'age'})
    )
    print_summary(structured, 'Structured Dataset (pre-filter)')

    notes_m = chunk_notes(mort_win,  notes_dr, 'mort')
    notes_r = chunk_notes(readm_win, notes_dr, 'readm')

    demo_cols = [
        'subject_id','age','gender','race_group','ethnicity_group',
        'insurance_group','age_bucket','mortality_icu','icu_readmission_30d'
    ]
    notes_m = notes_m.merge(structured[demo_cols], on='subject_id', how='left')
    notes_r = notes_r.merge(structured[demo_cols], on='subject_id', how='left')

    notes_dr_wide = notes_m.merge(
        notes_r, on=demo_cols, how='outer',
        suffixes=('_mort','_readm')
    )
    print_summary(notes_dr_wide, 'Notes Dataset (pre-filter)')

    common_ids = set(structured.subject_id) & set(notes_dr_wide.subject_id)

    structured_data = structured[structured.subject_id.isin(common_ids)].copy()
    unstructured_data = notes_dr_wide[notes_dr_wide.subject_id.isin(common_ids)].copy()

    structured_data.to_csv('all_structured_data.csv', index=False)
    unstructured_data.to_csv('all_unstructured_data.csv', index=False)

    print_summary(structured_data,   'Final Structured Data')
    print_summary(unstructured_data, 'Final Unstructured Data')

if __name__ == '__main__':
    main()
