import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib 
print(matplotlib.get_backend())

# Load CSV data
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "MSU_STUDENT_DATA_DE-IDENTIFIED.csv"))

# Convert 'ED_ADMIT_DATE' to datetime
df['ED_ADMIT_DATE'] = pd.to_datetime(df['ED_ADMIT_DATE'], errors='coerce')

# Sort and set index
df = df.sort_values('ED_ADMIT_DATE').set_index('ED_ADMIT_DATE')

# Aggregate daily admission counts
daily_counts = (
    df.groupby('ED_ADMIT_DATE')
    .size()
    .rename('admissions')
    .to_frame()
)

# Ensure daily frequency (fill missing days with 0 admissions)
daily_counts = daily_counts.asfreq('D', fill_value=0)

print(daily_counts.head())
print(daily_counts.index)

# Split data into train and test sets
train = daily_counts.iloc[:-14]
test = daily_counts.iloc[-14:]

# Fit SARIMAX model
model = SARIMAX(
    train['admissions'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
print(results.summary())

# Diagnostic plots
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecasting
forecast_steps = len(test) + 14
forecast = results.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

# Plot observed vs forecast
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

# Evaluate on test set
if not test.empty:
    pred = forecast_df['mean'].iloc[:len(test)]
    mae = mean_absolute_error(test['admissions'], pred)
    rmse = mean_squared_error(test['admissions'], pred) ** 0.5  # fixed 'squared' issue
    print(f"\n📊 Evaluation on test set:\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}")

