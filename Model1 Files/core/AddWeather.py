from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
import pandas as pd
import requests
def main():
    new_data = pd.DataFrame({
        "lag7":    [1, 2],
        "monday":  [1, 0],
        "tuesday": [0, 1],
        "wed"   :  [0,1],
        "date":    pd.to_datetime(["2025-10-27", "2025-10-28"])
    })
    add_weather(new_data)
def add_weather(df):
    start = df["ED_ADMIT_DATE"].min()
    end = df["ED_ADMIT_DATE"].max()
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    curl = f"https://api.open-meteo.com/v1/forecast?latitude=44.1591&longitude=-94.0092&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&temperature_unit=fahrenheit&precipitation_unit=inch&start_date={start_date}&end_date={end_date}"
    response = requests.get(curl)
    data = response.json()
    print(data)
    features = ["temperature_2m_max","temperature_2m_min","precipitation_sum"]
    df["TMAX (Degrees Fahrenheit)"]=data['daily']["temperature_2m_max"]
    df["TMIN (Degrees Fahrenheit)"]=data['daily']["temperature_2m_min"]
    df["PRCP (Inches)"]=data['daily']["precipitation_sum"]
    print(df.head())
    pass
if __name__ == '__main__':
    main()