from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "synthetic" / "Loan_default_multimodal.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Load Dataset
# ==========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# ==========================
# Separate Text Data
# ==========================

TEXT_COLUMNS = [
    "branch_notes",
    "call_summary",
    "verification_notes",
]

text_df = df[["LoanID"] + TEXT_COLUMNS].copy()

# Keep only structured data for ML
structured_df = df.drop(columns=["LoanID"] + TEXT_COLUMNS)

# ==========================
# Encode Categorical Columns
# ==========================

print("\nEncoding categorical columns...")

label_encoders = {}

categorical_columns = structured_df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:
    encoder = LabelEncoder()
    structured_df[column] = encoder.fit_transform(structured_df[column])
    label_encoders[column] = encoder

# ==========================
# Train/Test Split
# ==========================

print("Splitting dataset...")

X = structured_df.drop(columns=["Default"])
y = structured_df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Matching text rows
train_text = text_df.loc[X_train.index]
test_text = text_df.loc[X_test.index]

# ==========================
# Save Structured Data
# ==========================

train_df = X_train.copy()
train_df["Default"] = y_train

test_df = X_test.copy()
test_df["Default"] = y_test

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

# ==========================
# Save Text Data
# ==========================

train_text.to_csv(PROCESSED_DIR / "train_text.csv", index=False)
test_text.to_csv(PROCESSED_DIR / "test_text.csv", index=False)

print("\nPreprocessing completed successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")