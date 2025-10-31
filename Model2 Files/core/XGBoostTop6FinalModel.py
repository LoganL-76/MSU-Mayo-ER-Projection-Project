import pandas as pd
from xgboost import XGBRegressor
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# This script implements an XGBoost regression model for predicting patient counts in the top 6 ICD-10 categories.
# Uses best hyperparameters stored in a JSON file to train models for each category.
def main():
    # Load Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR,'..', "icd10_grouping", 'daily_category_patients_top6.csv')
    params_path = os.path.join(BASE_DIR, "XGBoostTop6BestParams.json")

    # Load Data
    df = pd.read_csv(file_path, parse_dates=['Day'])
    df = df.sort_values('Day')
    print(df.head())

    # Load best parameters from JSON
    with open(params_path, "r") as f:
        all_best_params = json.load(f)

    # Create Lag Features
    lag_days = 7
    category_cols = [c for c in df.columns if c not in ['Day', 'Total Patients']]
    for col in category_cols:
        for lag in range(1, lag_days + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create Temporal Features
    df['day_of_week'] = df['Day'].dt.dayofweek
    df['month'] = df['Day'].dt.month

    # Drop NA values after lagging
    df_features = df.dropna().reset_index(drop=True)

    # Prepare Features and Targets
    feature_cols = [col for col in df_features.columns if col not in category_cols and col != 'Day']
    X = df_features[feature_cols]
    Y = df_features[category_cols]

    models = {}

    # Train and Evaluate Models for Each Category
    for col in category_cols:
        print(f"\nTraining model for {col}...")

        y = Y[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        best_params = all_best_params[col]

        model = XGBRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42
        )
        model.fit(X_train, y_train)
        models[col] = model

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Evaluation for {col}: MAE={mae}, RMSE={rmse}, R2={r2}")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red')
        plt.title(f"Predicted vs Actual for {col}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()