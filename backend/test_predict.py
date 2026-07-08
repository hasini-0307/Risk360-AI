from ml.predict import predict_default

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
    "HasMortgage": 1,
    "HasDependents": 0,
    "LoanPurpose": 3,
    "HasCoSigner": 0
}

result = predict_default(sample)

print("\nPrediction Result")
print("-----------------")
print("Prediction:", result["prediction"])
print("Probability of Default:", round(result["default_probability"], 4))