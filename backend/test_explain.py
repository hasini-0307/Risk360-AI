from ml.explain import explain_prediction

sample = {
    "Age": 45,
    "Income": 52000,
    "LoanAmount": 180000,
    "CreditScore": 610,
    "MonthsEmployed": 48,
    "NumCreditLines": 5,
    "InterestRate": 11.5,
    "LoanTerm": 36,
    "DTIRatio": 0.42,

    "Education": 2,
    "EmploymentType": 1,
    "MaritalStatus": 1,
    "HasMortgage": 0,
    "HasDependents": 1,
    "LoanPurpose": 3,
    "HasCoSigner": 0,
}

result = explain_prediction(sample)

print("\nTop Feature Contributions\n")

for feature, value in result:
    print(f"{feature:<20} {value:.5f}")