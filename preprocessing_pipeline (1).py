"""
Week 2 - Logistics Data Collection, Cleaning and Preprocessing
Uses the included sample_logistics_data.csv.

The sample contains intentionally introduced quality issues so that
the cleaning pipeline can be demonstrated reproducibly.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "sample_logistics_data.csv"
OUTPUT_FILE = "cleaned_logistics_data.csv"

df = pd.read_csv(INPUT_FILE)

# 1. Remove exact duplicate records
df = df.drop_duplicates().copy()

# 2. Standardize categorical text
for col in ["shipping_mode", "traffic_level", "customer_region"]:
    df[col] = df[col].astype("string").str.strip()

# 3. Treat blank/placeholder categories as missing
df["traffic_level"] = df["traffic_level"].replace(
    {"": pd.NA, "nan": pd.NA, "None": pd.NA}
)

# 4. Validate numeric ranges
df.loc[df["items"] <= 0, "items"] = np.nan
df.loc[df["scheduled_days"] <= 0, "scheduled_days"] = np.nan
df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan
df.loc[df["order_value"] < 0, "order_value"] = np.nan

# 5. Detect distance outliers using IQR and convert extreme values to missing.
#    In production, investigate the original transaction before changing it.
q1 = df["distance_km"].quantile(0.25)
q3 = df["distance_km"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df.loc[(df["distance_km"] < lower) | (df["distance_km"] > upper),
       "distance_km"] = np.nan

# 6. Impute numeric missing values with median
numeric_cols = ["distance_km", "scheduled_days", "order_value", "items"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 7. Impute categorical missing values with mode
categorical_cols = ["shipping_mode", "traffic_level", "customer_region"]
for col in categorical_cols:
    mode = df[col].mode(dropna=True)
    fill_value = mode.iloc[0] if not mode.empty else "Unknown"
    df[col] = df[col].fillna(fill_value)

# 8. Recalculate the late-delivery target after cleaning
df["late_delivery"] = (
    df["actual_days"] > df["scheduled_days"]
).astype(int)

# 9. Normalize selected continuous variables for ML workflows
scaler = MinMaxScaler()
df[["distance_km_norm", "order_value_norm"]] = scaler.fit_transform(
    df[["distance_km", "order_value"]]
)

df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned rows:", len(df))
print("Missing values after cleaning:")
print(df.isna().sum())
print("Saved:", OUTPUT_FILE)
