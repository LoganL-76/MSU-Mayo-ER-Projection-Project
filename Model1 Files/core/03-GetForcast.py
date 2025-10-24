from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
import pandas as pd

def main():
    # Load the trained XGBoost model
    model = XGBRegressor()
    model.load_model("xgb_model.json")  # replace with your JSON file path

    # Example new data to predict (replace with your real features)
    # Suppose your model expects 3 features: ["feature1", "feature2", "feature3"]
    new_data = pd.DataFrame({
        "feature1": [5.1, 6.3],
        "feature2": [3.5, 2.9],
        "feature3": [1.4, 4.5]
    })

    # Make predictions
    predictions = model.predict(new_data)
    
if __name__ == '__main__':
    main()