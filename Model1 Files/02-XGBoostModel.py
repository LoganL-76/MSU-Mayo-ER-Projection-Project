from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
import pandas as pd

#
df = pd.read_csv("weather+.csv")
#train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
val_df   = df.iloc[split_idx:]

#Features and target
target = "count"
features = [col for col in train_df.columns if col != target]

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]

#Train XGBoost model
model = XGBRegressor(   
    max_depth=2,                
    n_estimators=10000,        
    early_stopping_rounds=25,  
    random_state=42,
    n_jobs=-1,
    eval_metric="mape"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)

#Predict and evaluate
y_pred = model.predict(X_val)


# import lightgbm as lgb
# model = lgb.LGBMRegressor()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)


mse = mean_absolute_error(y_val, y_pred)
print(f"Validation mae: {mse:.2f}")

#Baseline 1: Naive (yesterday)
y_pred_naive = val_df["lag1"]
mae_naive  = mean_absolute_error(y_val, y_pred_naive)
rmse_naive = mean_squared_error(y_val, y_pred_naive)
print(f"Naive mae: {mae_naive:.2f}")

#Baseline 2: Rolling 7-day average
y_pred_roll7 = val_df["roll7"]
mae_roll7  = mean_absolute_error(y_val, y_pred_roll7)
rmse_roll7 = mean_squared_error(y_val, y_pred_roll7)
print(f"Rolling 7 mae: {mae_roll7 :.2f}")