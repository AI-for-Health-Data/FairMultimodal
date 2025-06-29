import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

MIN_AGE        = 15
MAX_AGE        = 90
MIN_DURATION   = 30      
READMIT_DAYS   = 30     
NOTE_WINDOW_H  = 48

tqdm.pandas()

def preprocess_notes(text: str) -> str:
    x = re.sub(r"\[(.*?)\]", "", str(text))
    x = re.sub(r"[0-9]+\.", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

# (same as before)
patients = pd.read_csv("patients.csv.gz",
    usecols=["subject_id", "anchor_age"],
    compression="gzip"
)

icu_stays = pd.read_csv("icustays.csv.gz",
    usecols=["subject_id","hadm_id","stay_id","intime","outtime"],
    parse_dates=["intime","outtime"],
    compression="gzip"
)

stays = icu_stays.merge(patients, on="subject_id", how="left")\
                 .query("@MIN_AGE < anchor_age < @MAX_AGE")

# compute duration in hours, filter out short stays
stays["duration_h"] = (stays.outtime - stays.intime).dt.total_seconds()/3600
stays = stays.query("duration_h >= @MIN_DURATION").drop(columns="duration_h")

# Label 30-day ICU readmission
stays = stays.sort_values(["subject_id","intime"])
stays["next_intime"] = stays.groupby("subject_id")["intime"].shift(-1)
stays["gap_days"]   = (stays.next_intime - stays.outtime).dt.days
stays["readmit_30d"]= (stays.gap_days <= READMIT_DAYS).fillna(False).astype(int)

# Keep first ICU stay per patient
first_stays = (stays
               .sort_values(["subject_id","intime"])
               .drop_duplicates("subject_id", keep="first")
               [["subject_id","hadm_id","stay_id","outtime","readmit_30d"]])
first_stays.to_csv("first_stays_labeled.csv", index=False)

discharge = pd.read_csv("discharge.csv.gz",
                        usecols=["subject_id","hadm_id","chartdate","text"],
                        parse_dates=["chartdate"],
                        compression="gzip",
                        low_memory=False
                       ).rename(columns={"chartdate":"charttime"})

# Restrict to last 48 h of first ICU stay 
notes = discharge.merge(
    first_stays, on=["subject_id","hadm_id"], how="inner"
)
notes["start48h"] = notes.outtime - pd.Timedelta(hours=NOTE_WINDOW_H)
mask = (notes.charttime >= notes.start48h) & (notes.charttime <= notes.outtime)
notes48h = notes.loc[mask].copy()

notes48h["clean_text"] = notes48h.text.fillna("").progress_map(preprocess_notes)

agg = (notes48h
       .groupby(["subject_id","hadm_id","stay_id"])["clean_text"]
       .apply(" ".join)
       .reset_index()
       .rename(columns={"clean_text":"notes_concat"}))

final = first_stays.merge(agg, on=["subject_id","hadm_id","stay_id"], how="left")
final.to_csv("unstructured_readmit_first48h_discharge_only.csv", index=False)

print("Done:", len(final), "stays;", final.readmit_30d.sum(), "readmits")
