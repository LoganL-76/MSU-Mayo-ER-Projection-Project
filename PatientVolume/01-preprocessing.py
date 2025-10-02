import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# Read your CSV
df = pd.read_csv("data/MSU_STUDENT_DATA_DE-IDENTIFIED.csv")


df["ED_ADMIT_DATE"] = pd.to_datetime(df["ED_ADMIT_DATE"])  #df["day_of_week1"] = df["ED_ADMIT_DATE"].dt.day_name()
df["day_of_week"] = df["ED_ADMIT_DATE"].dt.day_name()
df["month"] = df["ED_ADMIT_DATE"].dt.month_name()

def get_season(date):
    m=date.month
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["season"] = df["ED_ADMIT_DATE"].map(get_season)





# Group by date and count how many patients per day
daily_counts = (
    df.groupby("ED_ADMIT_DATE")
      .agg(count=("DE_IDENTIFIED_PATIENT_KEY", "nunique"),
           day_of_week=("day_of_week", "first"),
           month=("month", "first"),
           season=("season", "first"))
      .reset_index()
)
def one_hot_encode(df,col):
    vals=df[col].unique()
    for val in vals:
        df[val] = (df[col] == val).astype(int)
    print(vals)
    return df
# # Save to a new CSV
obj_cols=[col for col in daily_counts.columns if daily_counts[col].dtype == "object"]
obj_cols
for i in obj_cols:
    daily_counts=one_hot_encode(daily_counts,i)

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=daily_counts["ED_ADMIT_DATE"].min(),
                        end=daily_counts["ED_ADMIT_DATE"].max())
daily_counts["is_holiday"] = daily_counts["ED_ADMIT_DATE"].isin(holidays).astype(int)
daily_counts["day_before_holiday"] = daily_counts["ED_ADMIT_DATE"].shift(-1).isin(holidays).astype(int)
daily_counts["day_after_holiday"] = daily_counts["ED_ADMIT_DATE"].shift(1).isin(holidays).astype(int)
wdf= pd.read_csv("data/weather_data.csv")[["DATE","TAVG (Degrees Fahrenheit)","TMAX (Degrees Fahrenheit)","TMIN (Degrees Fahrenheit)","PRCP (Inches)"]]
wdf["PRCP (Inches)"] = wdf["PRCP (Inches)"].fillna(0)
wdf["TAVG (Degrees Fahrenheit)"] = wdf["TAVG (Degrees Fahrenheit)"].fillna(0)
wdf["TMAX (Degrees Fahrenheit)"] = wdf["TMAX (Degrees Fahrenheit)"].fillna(0)
wdf["TMIN (Degrees Fahrenheit)"] = wdf["TMIN (Degrees Fahrenheit)"].fillna(0)
wdf["DATE"] = pd.to_datetime(wdf["DATE"])
merged = daily_counts.merge(wdf, left_on="ED_ADMIT_DATE", right_on="DATE", how="left")
merged = merged[~merged["ED_ADMIT_DATE"].dt.year.isin([2020, 2021])]
merged = merged.drop(columns=["DATE","ED_ADMIT_DATE","day_of_week","month","season","Fall","Spring","Winter","Summer"])

merged["roll14"] = merged["count"].shift(1).rolling(14).mean()
merged["roll7"] = merged["count"].shift(1).rolling(7).mean()
merged["roll30"] = merged["count"].shift(1).rolling(30).mean()
merged["lag1"] = merged["count"].shift(1)    # yesterday
merged["lag2"] = merged["count"].shift(2)    # 2 days ago
merged["lag7"] = merged["count"].shift(7) 
#print(merged.head())
merged.to_csv("data/FullyProccesedData.csv", index=False)
print(merged["count"].mean())



