import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# This script implements an XGBoost regression model for predicting patient counts in the top 6 ICD-10 categories.
# Stores best hyperparameters for each category in a JSON file after tuning.
# Very Computationally Intensive - May take a long time to run. 
def main():
    # Load Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR,'..', "icd10_grouping", 'daily_category_patients_top6.csv')
    df = pd.read_csv(file_path, parse_dates=['Day'])
    df = df.sort_values('Day')
    print(df.head())

    # Lag Features
    lag_days = 7
    category_cols = [c for c in df.columns if c not in ['Day', 'Total Patients']]
    for col in category_cols:
        for lag in range(1, lag_days + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Temporal Features
    df['day_of_week'] = df['Day'].dt.dayofweek
    df['month'] = df['Day'].dt.month

    df_features = df.dropna().reset_index(drop=True)

    feature_cols = [col for col in df_features.columns if col not in category_cols and col != 'Day']
    X = df_features[feature_cols]
    Y = df_features[category_cols]

    models = {}
    all_best_params = {}
    for col in category_cols:
        print(f"\nTraining model for {col}...")

        y = Y[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        params = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        model = GridSearchCV(
            estimator=XGBRegressor(random_state=42),
            param_grid=params,
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        models[col] = model.best_estimator_
        all_best_params[col] = model.best_params_
        print(f"Best parameters for {col}: {model.best_params_}")

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"{col} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

        plt.figure(figsize=(8, 4))
        plt.plot(y_test.values, label="Actual", alpha=0.7)
        plt.plot(y_pred, label="Predicted", alpha=0.7)
        plt.title(f"Predicted vs Actual for {col}")
        plt.legend()
        plt.show()
        # Save best parameters after tuning
        save_path = os.path.join(BASE_DIR, "XGBoostTop6BestParams.json")
        with open(save_path, "w") as f:
            json.dump(all_best_params, f, indent=4)
    

if __name__ == "__main__":
    main()