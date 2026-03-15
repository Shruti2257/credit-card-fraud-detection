# train_advanced_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# 1. Load the Dataset
print("📂 Loading dataset...")
# Make sure the 'creditcard.csv' file is in the same directory as this script.
try:
    data = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("❌ Error: 'creditcard.csv' not found. Please ensure the dataset is in the same folder.")
    exit()

# 2. Split Features and Labels
X = data.drop('Class', axis=1)
y = data['Class']

# 3. Train-Test Split (Stratified to maintain class ratios in splits)
# Test set must remain untouched for objective evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

# 4. Data Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✨ Data scaling complete.")

# 5. Handle Imbalance with SMOTE on Scaled Training Data
print("🔬 Applying SMOTE to balance training data...")
# SMOTE is applied only to the training data AFTER scaling.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Original training fraud count (Class 1): {y_train.sum()}")
print(f"SMOTE training fraud count (Class 1): {y_train_smote.sum()}")
print(f"SMOTE training size: {len(X_train_smote)}")

# 6. Train Random Forest Model
# Random Forest achieved the best performance (99.1% Accuracy, 94.1% F1) in your report[cite: 71, 72].
print("⚙️ Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train_smote, y_train_smote)
print("\n✅ Model Training Completed!")

# 7. Evaluate Model on Original Test Set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("📊 Evaluation on ORIGINAL Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report (Focus on Recall/F1 for Fraud Class):\n", classification_report(y_test, y_pred))

# 8. Save Model and Scaler
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n💾 Random Forest Model and Scaler saved successfully as 'random_forest_model.pkl' and 'scaler.pkl'!\n")