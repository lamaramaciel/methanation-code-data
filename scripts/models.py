import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_excel('data/data_art1.xlsx', sheet_name='PlanML')
data.dropna(axis=0, inplace=True)

# Function to train and evaluate a model
def train_and_evaluate_model(model, target_variable):
    """
    Train and evaluate a machine learning model for the given target variable.

    Args:
        model: Machine learning model (e.g., XGBRegressor, RandomForestRegressor).
        target_variable (str): Name of the target column in the dataset.

    Returns:
        None
    """
    # Define features (X) and target (y)
    X = data.drop(columns=['CO2 Conversion (%)', 'CH4 Selectivity (%)', 'Promoter (Z)', 
                           'Promoter', 'Support Type', 'DOI', 'Year', 'Synthesis Method'])
    y = data[target_variable]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Results for {target_variable} with {model.__class__.__name__}:")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print("-" * 50)

# Define models to evaluate
models = [
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42),
    RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42),
    LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42),
    GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
]

# Evaluate models for both target variables
for model in models:
    train_and_evaluate_model(model, 'CO2 Conversion (%)')
    train_and_evaluate_model(model, 'CH4 Selectivity (%)')
