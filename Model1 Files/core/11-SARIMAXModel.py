import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ==========================
# 1️⃣ Load and prepare data
# ==========================
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "MSU_STUDENT_DATA_DE-IDENTIFIED.csv")
df = pd.read_csv(data_path)

# Convert to datetime
df['ED_ADMIT_DATE'] = pd.to_datetime(df['ED_ADMIT_DATE'], errors='coerce')

# Sort and aggregate daily counts
daily_counts = (
    df.groupby('ED_ADMIT_DATE')
    .size()
    .rename('admissions')
    .to_frame()
)

# Ensure datetime index with daily frequency
daily_counts = daily_counts.sort_index()
daily_counts.index = pd.DatetimeIndex(daily_counts.index, name='ED_ADMIT_DATE')
daily_counts = daily_counts.asfreq('D')  # <== adds daily frequency info

print(daily_counts.head())
print(daily_counts.index)

# ==========================
# 2️⃣ Train-test split
# ==========================
train = daily_counts.iloc[:-14]
test = daily_counts.iloc[-14:]

# ==========================
# 3️⃣ Fit SARIMAX model
# ==========================
model = SARIMAX(
    train['admissions'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 365),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
print(results.summary())

# Optional diagnostics
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# ==========================
# 4️⃣ Forecasting
# ==========================
forecast_steps = len(test) + 14  # Forecast for test period + 1 year
forecast = results.get_forecast(steps=forecast_steps)

# Create forecast index starting the day after training data ends
forecast_index = pd.date_range(
    start=train.index[-1] + pd.Timedelta(days=1),
    periods=forecast_steps,
    freq='D'
)

forecast_df = forecast.summary_frame()
forecast_df.index = forecast_index  # align to actual dates

# ==========================
# 5️⃣ Plot results
# ==========================
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts['admissions'], label='Observed', color='blue')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='orange')
plt.fill_between(
    forecast_df.index,
    forecast_df['mean_ci_lower'],
    forecast_df['mean_ci_upper'],
    color='lightgray', alpha=0.5
)
plt.title("Emergency Department (ED) Admissions Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Admissions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================
# 6️⃣ Evaluate (if test data exists)
# ==========================
if not test.empty:
    pred = forecast_df['mean'].iloc[:len(test)]
    mae = mean_absolute_error(test['admissions'], pred)
    rmse = np.sqrt(mean_squared_error(test['admissions'], pred))
    print(f"\n📊 Evaluation on test set:\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}")