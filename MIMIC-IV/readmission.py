import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

MIN_AGE       = 15    
MAX_AGE       = 90    
READMIT_DAYS  = 30
NOTE_WINDOW_H = 48

def preprocess_notes(text: str) -> str:
    """
    - Remove [bracketed] sections
    - Strip numeric bullets ("1.", "2.", etc.)
    - Collapse whitespace, lowercase
    """
    x = re.sub(r"\[(.*?)\]", "", str(text))
    x = re.sub(r"[0-9]+\.", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

tqdm.pandas()

patients = pd.read_csv("patients.csv.gz",
    usecols=["subject_id","anchor_age"],
    compression="gzip"
)

icu_stays = pd.read_csv("icustays.csv.gz",
    usecols=["subject_id","hadm_id","stay_id","intime","outtime"],
    parse_dates=["intime","outtime"],
    compression="gzip"
)

stays = (
    icu_stays
    .merge(patients, on="subject_id", how="left")
    .query("@MIN_AGE < anchor_age < @MAX_AGE")
)

# Label 30-day ICU readmission 
stays = stays.sort_values(["subject_id","intime"])
stays["next_intime"] = stays.groupby("subject_id")["intime"].shift(-1)
stays["gap_days"]   = (stays["next_intime"] - stays["outtime"]).dt.days
stays["readmit_30d"]= (stays["gap_days"] <= READMIT_DAYS).fillna(False).astype(int)

# Keep only each patient’s first ICU stay
first_stays = (
    stays
    .sort_values(["subject_id","intime"])
    .drop_duplicates("subject_id", keep="first")
    .loc[:, ["subject_id","hadm_id","stay_id","intime","outtime","readmit_30d"]]
)
first_stays.to_csv(OUTPUT_DIR/"first_stays_labeled.csv", index=False)

# Discharge summaries
discharge = pd.read_csv("discharge.csv.gz",
    usecols=["subject_id","hadm_id","chartdate","text"],
    parse_dates=["chartdate"],
    compression="gzip",
    low_memory=False
).rename(columns={"chartdate":"charttime"})

# Radiology reports
radiology = pd.read_csv("radiology.csv.gz",
    usecols=["subject_id","hadm_id","charttime","text"],
    parse_dates=["charttime"],
    compression="gzip",
    low_memory=False
)

# Combine into one DataFrame of notes
notes = pd.concat(
    [discharge, radiology],
    ignore_index=True,
    sort=False
)
notes = notes.merge(
    first_stays[["subject_id","hadm_id","stay_id","outtime"]],
    on=["subject_id","hadm_id"],
    how="inner"
)

notes["window_start"] = notes["outtime"] - pd.Timedelta(hours=NOTE_WINDOW_H)
mask = (notes["charttime"] >= notes["window_start"]) & (notes["charttime"] <= notes["outtime"])
notes_48h = notes.loc[mask].copy()

notes_48h["clean_text"] = notes_48h["text"].fillna("").progress_map(preprocess_notes)

agg = (
    notes_48h
    .groupby(["subject_id","hadm_id","stay_id"])["clean_text"]
    .apply(lambda segs: " ".join(segs))
    .reset_index()
    .rename(columns={"clean_text":"notes_concat"})
)

final = first_stays.merge(
    agg, on=["subject_id","hadm_id","stay_id"], how="left"
)
out_fp = "unstructured_readmit_first48h.csv"
final.to_csv(out_fp, index=False)

print(f"Saved {out_fp}")
print(f"  total stays: {len(final)}")
print(f"  30-day readmits: {final.readmit_30d.sum()}")
