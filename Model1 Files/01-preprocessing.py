import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
# Read data csv
df = pd.read_csv("data/MSU_STUDENT_DATA_DE-IDENTIFIED.csv")

#group by date and add up patients
df["ED_ADMIT_DATE"] = pd.to_datetime(df["ED_ADMIT_DATE"])  
df = (df.groupby("ED_ADMIT_DATE")
      .agg(count=("DE_IDENTIFIED_PATIENT_KEY", "nunique"))
      .reset_index())

#adding other time date based features
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


#use one hot encoding for categortical data (day and month) data
def one_hot_encode(df,col):
    vals=df[col].unique()
    for val in vals:
        df[val] = (df[col] == val).astype(int)
    return df
obj_cols=[col for col in df.columns if df[col].dtype == "object"]
for i in obj_cols:
    df=one_hot_encode(df,i)
    df.drop(i, axis=1, inplace=True)

#add holiday related columns
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=df["ED_ADMIT_DATE"].min(),end=df["ED_ADMIT_DATE"].max())
df["is_holiday"] = df["ED_ADMIT_DATE"].isin(holidays).astype(int)
df["day_before_holiday"] = df["ED_ADMIT_DATE"].shift(-1).isin(holidays).astype(int)
df["day_after_holiday"] = df["ED_ADMIT_DATE"].shift(1).isin(holidays).astype(int)


#preproccess weather data
wdf= pd.read_csv("data/weather_data.csv")[["DATE","TAVG (Degrees Fahrenheit)","TMAX (Degrees Fahrenheit)","TMIN (Degrees Fahrenheit)","PRCP (Inches)"]]
wdf["PRCP (Inches)"] = wdf["PRCP (Inches)"].fillna(0)
for col in ["TAVG (Degrees Fahrenheit)", "TMAX (Degrees Fahrenheit)", "TMIN (Degrees Fahrenheit)"]:
    wdf[col] = wdf[col].fillna(wdf[col].mean())
wdf["DATE"] = pd.to_datetime(wdf["DATE"])

#add weather data to main data file
merged = df.merge(wdf, left_on="ED_ADMIT_DATE", right_on="DATE", how="left")
merged.drop("DATE", axis=1, inplace=True)


#adding rolling windows and lag features 
merged["roll14"] = merged["count"].shift(1).rolling(14).mean()
merged["roll7"] = merged["count"].shift(1).rolling(7).mean()
merged["roll30"] = merged["count"].shift(1).rolling(30).mean()
merged["lag1"] = merged["count"].shift(1)    # yesterday
merged["lag2"] = merged["count"].shift(2)    # 2 days ago
merged["lag7"] = merged["count"].shift(7) 
merged.dropna(subset=["roll30"], inplace=True)


#remove covid
#merged = merged[~merged["ED_ADMIT_DATE"].dt.year.isin([2020, 2021])]


#save final file to csv
print(merged.head())
merged.to_csv("data/FullyProccesedData.csv", index=False)
print(merged["count"].mean())



