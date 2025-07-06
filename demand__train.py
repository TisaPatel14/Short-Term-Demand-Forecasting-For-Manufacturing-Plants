import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("demandForcast.csv")
df['forecast_diff'] = abs(df['Units Sold'] - df['Demand Forecast'])
df['sold_inventory_ratio'] = df['Units Sold'] / df['Inventory Level']
df['order_vs_sales_ratio'] = df['Units Ordered'] / (df['Units Sold'] + 1e-5)
df.fillna(0, inplace=True)
def label_demand_accuracy(row):
    forecast_tolerance = 0.25  # 25% deviation allowed
    if row['Units Sold'] > row['Inventory Level']:
        return 0
    if row['forecast_diff'] > forecast_tolerance * row['Demand Forecast']:
        return 0
    if row['Units Ordered'] < 0.5 * row['Units Sold']:
        return 0
    return 1

df['is_demand_accurate'] = df.apply(label_demand_accuracy, axis=1)
cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
feature_cols = [
    'Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level',
    'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount',
    'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality',
    'forecast_diff', 'sold_inventory_ratio', 'order_vs_sales_ratio'
]
X = df[feature_cols]
y = df['is_demand_accurate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save model
joblib.dump(model, "demand_accuracy_model.pkl")
print("Demand accuracy model saved as 'demand_accuracy_model.pkl'")
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices] , y=[feature_cols[i] for i in indices])
plt.title("Feature Importances")
plt.tight_layout()

# Show or Save based on backend
plt.savefig("feature_importance.png")
print("Feature importance plot saved as 'feature_importance.png'")
