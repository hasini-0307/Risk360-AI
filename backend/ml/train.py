import pandas as pd
from catboost import CatBoostClassifier
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

print("Loading training and testing datasets...")

train_df = pd.read_csv("backend/data/processed/train.csv")
test_df = pd.read_csv("backend/data/processed/test.csv")

print("Datasets loaded successfully!")

# -----------------------------
# Separate Features and Target
# -----------------------------

X_train = train_df.drop(columns=["Default"])
y_train = train_df["Default"]

X_test = test_df.drop(columns=["Default"])
y_test = test_df["Default"]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nNumber of features:", X_train.shape[1])

# -----------------------------
# Create CatBoost Model
# -----------------------------

model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, 4],
    random_seed=42,
    verbose=100
)

print("\nTraining CatBoost model...")

model.fit(X_train, y_train)
print("\nEvaluating model...")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nTraining completed!")

# -----------------------------
# Save Model
# -----------------------------

os.makedirs("backend/models", exist_ok=True)

model.save_model("backend/models/catboost_model.cbm")
joblib.dump(
    X_train.columns.tolist(),
    "backend/models/feature_columns.pkl"
)

print("Feature list saved successfully!")

print("\nModel saved successfully!")
print("Location: backend/models/catboost_model.cbm")