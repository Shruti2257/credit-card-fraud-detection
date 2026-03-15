import pandas as pd, joblib, numpy as np
import pandas as pd

df = pd.read_csv("creditcard.csv")
fraud_row = df[df['Class']==1].iloc[0]
print(fraud_row)


# Load model/scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset and pick a fraud sample from the CSV
df = pd.read_csv("creditcard.csv")
fraud_row = df[df['Class'] == 1].iloc[0]  # first fraud sample

# Correct FEATURE_ORDER
FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

X_raw = fraud_row[FEATURE_ORDER].values.reshape(1, -1)

X_scaled = scaler.transform(X_raw)
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0,1]

print("Using known fraud sample (first found in dataset):")
print("Raw features (first 10):", X_raw[0][:10])
print("Scaled features (first 10):", X_scaled[0][:10])
print("Prediction label:", pred)
print("Probability of fraud:", proba)
