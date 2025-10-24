import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Load your data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR,"..", "data", "FullyProccesedData.csv")
df = pd.read_csv(file_path)
# Convert date column to datetime and set as index
df['ED_ADMIT_DATE'] = pd.to_datetime(df['ED_ADMIT_DATE'])
df.set_index('ED_ADMIT_DATE', inplace=True)

# Sort by date to ensure proper time series order
df.sort_index(inplace=True)

# Prepare exogenous variables (day of week)
# Create day of week feature matrix
exog_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
exog = df[exog_columns]

# Target variable
y = df['count']

# Split data into train and test sets (e.g., 80-20 split)
train_size = int(len(df) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
exog_train, exog_test = exog[:train_size], exog[train_size:]

print(f"Training set size: {len(y_train)}")
print(f"Test set size: {len(y_test)}")
print(f"\nDate range - Train: {y_train.index[0]} to {y_train.index[-1]}")
print(f"Date range - Test: {y_test.index[0]} to {y_test.index[-1]}")

# Visualize the data
plt.figure(figsize=(12, 4))
plt.plot(y_train.index, y_train, label='Training Data')
plt.plot(y_test.index, y_test, label='Test Data')
plt.xlabel('Date')
plt.ylabel('Patient Count')
plt.title('ER Patient Volume Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Check for stationarity and seasonality patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y_train, lags=40, ax=axes[0])
plot_pacf(y_train, lags=40, ax=axes[1])
axes[0].set_title('Autocorrelation Function')
axes[1].set_title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# Define and fit SARIMAX model
# SARIMAX parameters: (p, d, q) x (P, D, Q, s)
# p, d, q: non-seasonal AR, differencing, MA orders
# P, D, Q, s: seasonal AR, differencing, MA orders, and seasonal period
# For daily data with weekly patterns, s=7

# Starting with a reasonable parameter set
# You may need to tune these based on ACF/PACF plots and model diagnostics
model = SARIMAX(
    y_train,
    exog=exog_train,
    order=(1, 1, 1),           # (p, d, q) - adjust based on your data
    seasonal_order=(1, 0, 1, 7),  # (P, D, Q, s) - weekly seasonality
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit the model
print("\nFitting SARIMAX model...")
results = model.fit(disp=False)

# Print model summary
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(results.summary())

# Make predictions on test set
print("\nGenerating forecasts...")
forecast = results.forecast(steps=len(y_test), exog=exog_test)

# Check for NaN values in forecast
if forecast.isna().any():
    print(f"Warning: {forecast.isna().sum()} NaN values found in forecast")
    print("This may indicate model issues. Consider adjusting parameters.")
    
    # Drop NaN values for evaluation
    valid_idx = ~(forecast.isna() | y_test.isna())
    y_test_clean = y_test[valid_idx]
    forecast_clean = forecast[valid_idx]
else:
    y_test_clean = y_test
    forecast_clean = forecast

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_clean, forecast_clean)
rmse = np.sqrt(mean_squared_error(y_test_clean, forecast_clean))
mape = np.mean(np.abs((y_test_clean - forecast_clean) / y_test_clean)) * 100
r2 = r2_score(y_test_clean, forecast_clean)

print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared (R²): {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual', marker='o', markersize=3)
plt.plot(y_test.index, forecast, label='Predicted', marker='x', markersize=3)
plt.xlabel('Date')
plt.ylabel('Patient Count')
plt.title('SARIMAX Model: Actual vs Predicted ER Patient Volume')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residual diagnostics
residuals = y_test_clean - forecast_clean

# Ensure alignment by using the same index for both
residuals_clean = residuals.dropna()
forecast_for_plot = forecast_clean.loc[residuals_clean.index]

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(residuals_clean.index, residuals_clean.values)
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(residuals_clean.values, bins=20, edgecolor='black')
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.scatter(forecast_for_plot.values, residuals_clean.values)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
from scipy import stats
stats.probplot(residuals_clean.values, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Function to make future predictions
def predict_future(model_results, days_ahead=7):
    """
    Generate predictions for future dates
    
    Parameters:
    - model_results: fitted SARIMAX model results
    - days_ahead: number of days to forecast
    """
    # Create future dates
    last_date = y.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    
    # Create exogenous variables for future dates
    future_exog = pd.DataFrame(0, index=future_dates, columns=exog_columns)
    for date in future_dates:
        day_name = date.day_name()
        future_exog.loc[date, day_name] = 1
    
    # Generate forecast
    future_forecast = model_results.forecast(steps=days_ahead, exog=future_exog)
    
    return pd.DataFrame({'Date': future_dates, 'Predicted_Count': future_forecast.values})

# Example: Predict next 14 days
print("\n" + "="*80)
print("FUTURE PREDICTIONS (Next 14 Days)")
print("="*80)
future_predictions = predict_future(results, days_ahead=14)
print(future_predictions.to_string(index=False))

# Function to make future predictions
def predict_future(model_results, days_ahead=7):
    """
    Generate predictions for future dates
    
    Parameters:
    - model_results: fitted SARIMAX model results
    - days_ahead: number of days to forecast
    """
    # Create future dates
    last_date = y.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    
    # Create exogenous variables for future dates
    future_exog = pd.DataFrame(0, index=future_dates, columns=exog_columns)
    for date in future_dates:
        day_name = date.day_name()
        future_exog.loc[date, day_name] = 1
    
    # Generate forecast
    future_forecast = model_results.forecast(steps=days_ahead, exog=future_exog)
    
    return pd.DataFrame({'Date': future_dates, 'Predicted_Count': future_forecast.values})

# Example: Predict next 14 days
print("\n" + "="*80)
print("FUTURE PREDICTIONS (Next 14 Days)")
print("="*80)
future_predictions = predict_future(results, days_ahead=14)
print(future_predictions.to_string(index=False))

# Optional: Grid search for optimal parameters (computationally intensive)
def grid_search_sarima(y_train, exog_train, y_test, exog_test):
    """
    Perform grid search to find optimal SARIMAX parameters
    Warning: This can take a long time to run
    """
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)
    s = 7  # weekly seasonality
    
    best_aic = np.inf
    best_params = None
    
    results_list = []
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            try:
                                model = SARIMAX(y_train, exog=exog_train,
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, s),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
                                fitted = model.fit(disp=False)
                                
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_params = ((p, d, q), (P, D, Q, s))
                                
                                results_list.append({
                                    'order': (p, d, q),
                                    'seasonal_order': (P, D, Q, s),
                                    'AIC': fitted.aic
                                })
                                
                            except:
                                continue
    
    print(f"\nBest parameters: order={best_params[0]}, seasonal_order={best_params[1]}")
    print(f"Best AIC: {best_aic:.2f}")
    
    return best_params, pd.DataFrame(results_list)