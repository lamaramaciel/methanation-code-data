import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_excel('data/data_art1.xlsx', sheet_name='PlanML')
data.dropna(axis=0, inplace=True)

# Define features (X) and target (y)
X = data.drop(columns=['CO2 Conversion (%)', 'CH4 Selectivity (%)', 'Promoter (Z)', 
                       'Promoter', 'Support Type', 'DOI', 'Year', 'Synthesis Method'])
y = data['CO2 Conversion (%)']  # You can change to 'CH4 Selectivity (%)'

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# SHAP analysis
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)

# Plot Partial Dependence for a specific feature
shap.dependence_plot('Ni (wt%)', shap_values.values, X_test)
