from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
import pandas as pd

#
df = pd.read_csv("data/FullyProccesedData.csv")
#train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
val_df   = df.iloc[split_idx:]

#Features and target
target = "count"
features = [col for col in train_df.columns if col != target and col != "ED_ADMIT_DATE"]

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]

#Train XGBoost model
model = XGBRegressor(
    max_depth=2,             
    learning_rate=0.03,      
    n_estimators=2000,          
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    eval_metric="mae"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)



#Predict and evaluate
y_pred = model.predict(X_val)
#analyzing results
mse = mean_absolute_error(y_val, y_pred)
mape = mean_absolute_percentage_error(y_val,y_pred)
print(f"Validation mae: {mse:.2f}")
print(f"Validation mape: {mape:.2f}")

#Baseline 1: Naive (yesterday)
y_pred_naive = val_df["lag1"]
mae_naive  = mean_absolute_error(y_val, y_pred_naive)
mape_naive = mean_absolute_percentage_error(y_val, y_pred_naive)
print(f"Naive mae: {mae_naive:.2f}")
print(f"Naive mape: {mape_naive:.2f}")
