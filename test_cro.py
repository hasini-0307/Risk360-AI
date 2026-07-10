from ai_engine.agents.behavior_agent import BehaviorAgent
from ai_engine.agents.income_agent import IncomeAgent
from ai_engine.agents.transaction_agent import TransactionAgent
from ai_engine.agents.document_agent import DocumentAgent

from ai_engine.cro.chief_risk_officer import ChiefRiskOfficer

from ai_engine.schemas.request_schema import AIRequest


request = AIRequest(

    borrower_id="B100",

    loan_id="L100",

    loan_type="Home Loan",

    default_probability=0.81,

    model_confidence=0.91,

    borrower_profile={

        "missed_payments":4,

        "days_past_due":45,

        "previous_defaults":1,

        "loan_restructures":0,

        "employment_status":"Employed",

        "employment_type":"Temporary",

        "debt_to_income_ratio":0.58,

        "salary_trend":"Declining",

        "employment_years":0.8,

        "credit_utilization":0.92,

        "average_balance":3200,

        "cash_withdrawals":12,

        "large_transactions":5,

        "emi_bounces":2,

    },

    shap_values={

        "missed_payments":0.42,

        "days_past_due":0.28,

        "credit_utilization":0.21,

        "debt_to_income_ratio":0.18,

        "document_verification":0.15,

    },

    branch_notes="""
Employer could not verify employment.
Customer appeared nervous.
Address mismatch found.
""",

    document_text="""
Income certificate appears suspicious.
One supporting document missing.
Property ownership mismatch detected.
""",

    similar_cases=[

        {"case":1},

        {"case":2},

        {"case":3},

    ]

)

print("\nRunning Specialist Agents...\n")

behavior = BehaviorAgent().analyze(request)

income = IncomeAgent().analyze(request)

transaction = TransactionAgent().analyze(request)

document = DocumentAgent().analyze(request)

print("✓ Behavior Complete")

print("✓ Income Complete")

print("✓ Transaction Complete")

print("✓ Document Complete")


cro = ChiefRiskOfficer()

result = cro.evaluate(

    request,

    [

        behavior,

        income,

        transaction,

        document,

    ]

)

print("\n========== FINAL CRO OUTPUT ==========\n")

print(result.model_dump_json(indent=4))