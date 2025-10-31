from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import os
from AddWeather import add_weather

def main():

    print(get_forecast("10/30/25"))

    """
    # Load the trained XGBoost model
    model = XGBRegressor()
    model.load_model("xgb_model.json")  # replace with your JSON file path

    # Example new data to predict (replace with your real features)
    # Suppose your model expects 3 features: ["feature1", "feature2", "feature3"]
    new_data = pd.DataFrame({
        "lag7":    [1, 2],
        "monday":  [1, 0],
        "tuesday": [0, 1],
        "wed"   :  [0,0,1]
    })

    # Make predictions
    predictions = model.predict(new_data)
    """

def get_season(date):
    m = date.month
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def one_hot_encode(df, col):
    vals=df[col].unique()
    for val in vals:
        df[val] = (df[col] == val).astype(int)
    return df

def get_week(start_date) -> list[pd.DatetimeIndex]:
    start_date = pd.to_datetime(start_date)
    days = [start_date + pd.Timedelta(days=i) for i in range(7)]
    return days

def add_holidays(df):
    calendar = USFederalHolidayCalendar()
    dates = df["ED_ADMIT_DATE"]
    holidays = calendar.holidays(start=dates.min(), end=dates.max())
    print("holidays:", holidays)
    df["is_holiday"] = dates.isin(holidays).astype(int)
    df["day_before_holiday"] = (dates + pd.Timedelta(days=1)).isin(holidays).astype(int)
    df["day_after_holiday"] = (dates - pd.Timedelta(days=1)).isin(holidays).astype(int)

# Returns a week of volume predictions beginning with the argued start date
def get_forecast(start_date):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "xgb_model.json")

    model = XGBRegressor()
    model.load_model(model_path)

    days = get_week(start_date)

    input = pd.DataFrame({
        "ED_ADMIT_DATE": days,
        "day_of_week": [day.day_name() for day in days],
        "month": [day.month_name() for day in days],
        "season": [get_season(day) for day in days]
    })

    obj_cols = [col for col in input.columns if input[col].dtype == "object"]

    for col in obj_cols:
        input = one_hot_encode(input, col)
        input.drop(col, axis=1, inplace=True)

    add_holidays(input)

    add_weather(input)

    # add weather data to main data file
    # merged = input.merge(wdf, left_on="ED_ADMIT_DATE", right_on="DATE", how="left")
    # merged.drop("DATE", axis=1, inplace=True)

    
    #adding rolling windows and lag features 
    # merged["roll14"] = merged["count"].shift(1).rolling(14).mean()
    # merged["roll7"] = merged["count"].shift(1).rolling(7).mean()
    # merged["roll30"] = merged["count"].shift(20).rolling(30).mean()
    # merged["lag1"] = merged["count"].shift(1)    # yesterday
    # merged["lag2"] = merged["count"].shift(2)    # 2 days ago
    # merged["lag7"] = merged["count"].shift(7) 
    # merged.dropna(subset=["roll30"], inplace=True)

    print(input)

    #return model.predict(merged)
    
if __name__ == '__main__':
    main()