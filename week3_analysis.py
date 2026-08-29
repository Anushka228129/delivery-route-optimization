import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sample_logistics_data.csv")

# Basic EDA
print(df.shape)
print(df.describe(numeric_only=True))
print("\nMissing values:\n", df.isna().sum())

# Prepare analysis fields
df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
df["scheduled_days"] = df["scheduled_days"].fillna(df["scheduled_days"].median())
df["order_value"] = df["order_value"].fillna(df["order_value"].median())
df["traffic_level"] = df["traffic_level"].fillna(df["traffic_level"].mode()[0])

# Late-delivery rates
print("\nLate rate by shipping mode:")
print(df.groupby("shipping_mode")["late_delivery"].mean().mul(100).sort_values(ascending=False))

print("\nLate rate by traffic level:")
print(df.groupby("traffic_level")["late_delivery"].mean().mul(100).sort_values(ascending=False))

# Correlation for analysis features
features = ["distance_km", "scheduled_days", "order_value", "items", "late_delivery"]
print("\nCorrelation with late_delivery:")
print(df[features].corr()["late_delivery"].sort_values())

# Visualizations
df.groupby("shipping_mode")["late_delivery"].mean().mul(100).sort_values().plot(kind="barh")
plt.title("Late Delivery Rate by Shipping Mode")
plt.xlabel("Late delivery rate (%)")
plt.tight_layout()
plt.show()

df.groupby("traffic_level")["late_delivery"].mean().mul(100).sort_values().plot(kind="bar")
plt.title("Late Delivery Rate by Traffic Level")
plt.ylabel("Late delivery rate (%)")
plt.tight_layout()
plt.show()

plt.scatter(df["distance_km"], df["order_value"])
plt.xlabel("Distance (km)")
plt.ylabel("Order value")
plt.title("Distance vs Order Value")
plt.tight_layout()
plt.show()
