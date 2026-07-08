import random
import pandas as pd
from pathlib import Path

# -----------------------------
# Load Dataset
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "raw" / "Loan_default.csv"
output_file = BASE_DIR / "data" / "processed" / "Loan_default_multimodal.csv"

df = pd.read_csv(input_file, nrows=100000)

random.seed(42)

# -------------------------------------------------
# Templates
# -------------------------------------------------

positive_notes = [
    "Repayment history has remained consistent.",
    "Customer demonstrates stable repayment behaviour.",
    "Salary credits appear regular.",
    "No repayment concerns identified.",
    "Financial profile appears stable.",
    "Customer maintains healthy repayment habits.",
    "Existing obligations are well managed.",
    "Income appears sufficient for current obligations.",
]

risk_notes = [
    "Debt obligations are relatively high.",
    "Credit score remains below preferred threshold.",
    "Recent employment change may affect repayment stability.",
    "Existing financial commitments increase repayment burden.",
    "Borrower may require closer monitoring.",
    "Income level may not comfortably support current loan.",
    "Higher credit utilization observed.",
    "Financial risk appears elevated.",
]

events = [
    "Customer requested EMI restructuring.",
    "Collection reminder issued.",
    "Temporary cash flow issues reported.",
    "Borrower requested payment extension.",
    "Medical expenses affected repayment capacity.",
    "Recent job transition discussed.",
    "Customer expects salary increment next quarter.",
    "Salary credited later than usual.",
    "Business revenue declined recently.",
    "Family expenses increased significantly.",
    "Employer confirmed stable employment.",
    "Employer verification pending.",
    "Partial payment received last month.",
    "No significant repayment events reported.",
]

call_templates = [
    "Customer confirmed next EMI payment.",
    "Customer requested repayment schedule clarification.",
    "Customer denied financial difficulties.",
    "Customer acknowledged repayment responsibilities.",
    "Customer requested temporary payment extension.",
    "Customer expects salary credit within one week.",
    "Customer unavailable. Follow-up scheduled.",
    "Customer reported temporary financial stress.",
    "Customer confirmed employment details.",
]

verification_templates = [
    "Employment verified successfully.",
    "Income documents validated.",
    "Address verification completed.",
    "Salary slips verified.",
    "Bank statements verified.",
    "Minor document discrepancy observed.",
    "Additional KYC requested.",
    "Employer verification pending.",
    "Identity documents verified successfully.",
]


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def generate_branch_note(row):

    notes = []

    credit = row["CreditScore"]
    income = row["Income"]
    dti = row["DTIRatio"]
    employment = row["EmploymentType"]
    default = row["Default"]

    if credit < 600:
        notes.append("Credit score is below the bank's preferred threshold.")

    elif credit > 750:
        notes.append("Strong credit history observed.")

    if dti > 0.40:
        notes.append("Debt obligations represent a significant portion of monthly income.")

    if income < 40000:
        notes.append("Income level requires careful repayment assessment.")

    if employment == "Unemployed":
        notes.append("Current employment status increases repayment uncertainty.")

    elif employment == "Self-employed":
        notes.append("Income may fluctuate due to self-employment.")

    if default == 1:
        notes.append(random.choice(risk_notes))
    else:
        notes.append(random.choice(positive_notes))

    if random.random() < 0.7:
        notes.append(random.choice(events))

    random.shuffle(notes)

    return " ".join(notes)


def generate_call_summary(row):

    sentences = [random.choice(call_templates)]

    if row["Default"] == 1:
        sentences.append(
            random.choice([
                "Customer discussed delayed payments.",
                "Borrower requested additional repayment flexibility.",
                "Customer acknowledged outstanding dues."
            ])
        )

    else:
        sentences.append(
            random.choice([
                "Customer confirmed timely repayment.",
                "Customer expressed confidence in repayment schedule.",
                "No repayment concerns raised during discussion."
            ])
        )

    return " ".join(sentences)


def generate_verification(row):

    notes = [random.choice(verification_templates)]

    if row["HasCoSigner"] == "Yes":
        notes.append("Co-applicant information verified.")

    if row["HasMortgage"] == "Yes":
        notes.append("Existing mortgage verified.")

    if row["EmploymentType"] == "Full-time":
        notes.append("Stable employment confirmed.")

    elif row["EmploymentType"] == "Part-time":
        notes.append("Part-time income verified.")

    return " ".join(notes)


# -------------------------------------------------
# Generate Columns
# -------------------------------------------------

df["branch_notes"] = df.apply(generate_branch_note, axis=1)

df["call_summary"] = df.apply(generate_call_summary, axis=1)

df["verification_notes"] = df.apply(generate_verification, axis=1)

# -------------------------------------------------
# Save
# -------------------------------------------------

output_file.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)

print("Done!")
print(output_file)