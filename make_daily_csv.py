import os
import pandas as pd

# Input: the grouped file you already made
IN_PATH = r'out\icd10_grouped.csv'
OUT_DIR = r'out'
os.makedirs(OUT_DIR, exist_ok=True)

# Known column names from your data
DATE_COL = 'ED_ADMIT_DATE'
PATIENT_COL = 'DE_IDENTIFIED_PATIENT_KEY'
ICD_COL = 'DIAGNOSIS_CODE'

df = pd.read_csv(IN_PATH)

# make a calendar-Day column
df['Day'] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.date
df = df[df['Day'].notna()].copy()

# A) UNIQUE PATIENTS per (day, category)
dedup = df.drop_duplicates(subset=[PATIENT_COL, 'Day', 'Category'])
pivot_unique = dedup.pivot_table(
    index='Day', columns='Category',
    values=PATIENT_COL, aggfunc='nunique', fill_value=0
).sort_index()
pivot_unique['Total Patients'] = dedup.groupby('Day')[PATIENT_COL].nunique()
pivot_unique.to_csv(os.path.join(OUT_DIR, 'daily_category_patients_unique.csv'))

# B) RAW ROW/DIAGNOSIS COUNTS per (day, category)
pivot_rows = df.pivot_table(
    index='Day', columns='Category',
    values=ICD_COL, aggfunc='count', fill_value=0
).sort_index()
pivot_rows['Total Rows'] = df.groupby('Day')[ICD_COL].size()
pivot_rows.to_csv(os.path.join(OUT_DIR, 'daily_category_counts.csv'))

print('Saved: out\\daily_category_patients_unique.csv')
print('Saved: out\\daily_category_counts.csv')
