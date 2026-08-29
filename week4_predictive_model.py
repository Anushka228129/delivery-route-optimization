import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------
# 1. Load Logistics Dataset
# -----------------------------------------

df = pd.read_csv("sample_logistics_data (2).csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())


# -----------------------------------------
# 2. Data Cleaning
# -----------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Convert numerical columns to numeric values
numeric_columns = [
    "distance_km",
    "scheduled_days",
    "actual_days",
    "order_value",
    "items"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows where target value is missing
df = df.dropna(subset=["actual_days"])


# -----------------------------------------
# 3. Define Features and Target
# -----------------------------------------

# Target variable: actual delivery time
X = df[
    [
        "shipping_mode",
        "distance_km",
        "scheduled_days",
        "order_value",
        "items",
        "traffic_level",
        "customer_region"
    ]
]

y = df["actual_days"]


# -----------------------------------------
# 4. Identify Feature Types
# -----------------------------------------

categorical_features = [
    "shipping_mode",
    "traffic_level",
    "customer_region"
]

numerical_features = [
    "distance_km",
    "scheduled_days",
    "order_value",
    "items"
]


# -----------------------------------------
# 5. Preprocessing
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# -----------------------------------------
# 6. Build Predictive Model
# -----------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# -----------------------------------------
# 7. Train-Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------------------
# 8. Train Model
# -----------------------------------------

model.fit(X_train, y_train)


# -----------------------------------------
# 9. Make Predictions
# -----------------------------------------

predictions = model.predict(X_test)


# -----------------------------------------
# 10. Model Evaluation
# -----------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------------")
print("Mean Absolute Error (MAE):", round(mae, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("R-squared (R2):", round(r2, 3))


# -----------------------------------------
# 11. Prediction Comparison
# -----------------------------------------

results = pd.DataFrame({
    "Actual_Days": y_test.values,
    "Predicted_Days": np.round(predictions, 2)
})

print("\nActual vs Predicted Delivery Days:")
print(results)


# -----------------------------------------
# 12. Logistics Optimization Insights
# -----------------------------------------

print("\nOptimization Recommendations")
print("----------------------------")
print("1. Prioritize shipments with high predicted delivery times.")
print("2. Prefer shorter routes where possible to reduce delivery duration.")
print("3. Monitor high-traffic routes and consider alternative routes.")
print("4. Use predicted delivery times to improve scheduling.")
print("5. Allocate additional resources to shipments at high risk of delay.")


# -----------------------------------------
# End of Analysis
# -----------------------------------------

print("\nWeek 4 predictive analysis completed successfully.")
