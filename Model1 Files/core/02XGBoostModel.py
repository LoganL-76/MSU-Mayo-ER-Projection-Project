from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
import pandas as pd
import os

def main():


    # Get the directory of the current script and build path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR,"..", "data", "FullyProccesedData.csv")

    df = pd.read_csv(file_path)
    #train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

    split_idx = int(len(df) * 0.7)
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
        max_depth=3,             
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
    model.save_model("xgb_model.json")

    
    #Predict and evaluate
    y_pred = model.predict(X_val)
    #analyzing results
    mse = mean_absolute_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val,y_pred)
    print(f"Validation mae: {mse:.2f}")
    print(f"Validation mape: {mape:.2f}")

    import matplotlib.pyplot as plt
    import numpy as np  
    plt.figure(figsize=(10, 5))
    plt.plot(y_val.reset_index(drop=True), label="Actual")
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.legend()
    plt.xlabel("Index (e.g., Time)")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted over Time")
    plt.show()
if __name__ == '__main__':
    main()