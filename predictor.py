import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

def load_and_predict():
    # Sample qualifying and clean air race pace data (replace with actual if needed)
    qualifying_data = pd.DataFrame({
        "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
                   "HAM", "STR", "GAS", "ALO", "HUL"],
        "QualifyingTime (s)": [
            70.669, 69.954, 70.129, 71.000, 71.362, 71.213, 70.063, 70.942,
            70.382, 72.563, 71.994, 70.924, 71.596
        ],
        "CleanAirRacePace (s)": [
            93.191, 93.429, 93.232, 93.833, 94.497, 94.850, 93.418, 95.682,
            94.021, 95.318, 95.032, 94.784, 95.345
        ],
        "TeamPerformanceScore": [
            0.9, 1.0, 1.0, 0.8, 0.85, 0.6, 0.85, 0.5, 0.8, 0.5, 0.45, 0.45, 0.4
        ],
        "AveragePositionChange": [
            -1.0, 1.0, 0.2, 0.5, -0.3, 0.8, -1.5, -0.2, 0.3, 1.1, -0.4, -0.6, 0.0
        ],
        "RainProbability": [0.2] * 13,
        "Temperature": [24] * 13
    })

    # Define features and target
    X = qualifying_data[[
        "QualifyingTime (s)", "CleanAirRacePace (s)", "TeamPerformanceScore",
        "AveragePositionChange", "RainProbability", "Temperature"
    ]]
    y = qualifying_data["CleanAirRacePace (s)"] + np.random.normal(0, 1, size=len(X))  # Synthetic target

    # Handle missing values (if any)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    # Model training
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred_full = model.predict(X_imputed)
    qualifying_data["PredictedRaceTime (s)"] = y_pred_full

    # Evaluation
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Sort results
    results = qualifying_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
    podium = results["Driver"].iloc[:3].tolist()

    return results, podium, mae
