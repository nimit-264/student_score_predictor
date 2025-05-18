# Student Performance Prediction Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset (mock data used here, replace with actual CSV)
data = pd.DataFrame({
    'study_hours': [2, 4, 3, 5, 1, 6, 2.5, 3.5, 4.5, 5.5],
    'attendance': [75, 90, 85, 95, 60, 100, 80, 88, 92, 96],
    'previous_scores': [60, 85, 70, 90, 50, 95, 65, 75, 80, 88],
    'internet_usage': [3, 1, 2, 1, 4, 0.5, 3.5, 2.5, 1.5, 1],
    'health': [4, 5, 3, 4, 2, 5, 3, 4, 5, 4],
    'final_score': [65, 90, 75, 95, 55, 98, 70, 78, 88, 93]
})

# Features and labels
X = data.drop('final_score', axis=1)
y = data['final_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Final Scores')
plt.grid(True)
plt.show()
