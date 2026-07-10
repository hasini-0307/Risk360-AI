"""
Shared rule-based risk reason builder for all AI agents.
"""

from __future__ import annotations

from typing import List

from ai_engine.schemas.request_schema import AIRequest


class RiskReasonBuilder:

    @staticmethod
    def behavior(request: AIRequest) -> List[str]:

        profile = request.borrower_profile
        reasons = []

        if profile.get("missed_payments", 0) >= 3:
            reasons.append("Multiple missed repayments detected.")

        if profile.get("days_past_due", 0) >= 30:
            reasons.append("High Days Past Due (DPD).")

        if profile.get("previous_defaults", 0) > 0:
            reasons.append("Borrower has previous loan defaults.")

        if profile.get("loan_restructures", 0) > 0:
            reasons.append("Previous loan restructuring observed.")

        if not reasons:
            reasons.append("Repayment behaviour appears stable.")

        return reasons

    @staticmethod
    def income(request: AIRequest) -> List[str]:

        profile = request.borrower_profile
        reasons = []

        if profile.get("employment_status", "").lower() == "unemployed":
            reasons.append("Borrower is unemployed.")

        if profile.get("employment_type", "").lower() == "temporary":
            reasons.append("Temporary employment reduces income stability.")

        if profile.get("debt_to_income_ratio", 0) >= 0.5:
            reasons.append("High debt-to-income ratio.")

        if profile.get("salary_trend", "").lower() == "declining":
            reasons.append("Income trend is declining.")

        if profile.get("employment_years", 99) < 1:
            reasons.append("Employment history is less than one year.")

        if not reasons:
            reasons.append("Income profile appears stable.")

        return reasons

    @staticmethod
    def transaction(request: AIRequest) -> List[str]:

        profile = request.borrower_profile

        reasons = []

        if profile.get("credit_utilization", 0) >= 0.80:
            reasons.append("Very high credit utilization.")

        if profile.get("average_balance", 999999) < 5000:
            reasons.append("Low average account balance.")

        if profile.get("cash_withdrawals", 0) >= 10:
            reasons.append("Frequent cash withdrawals.")

        if profile.get("large_transactions", 0) >= 3:
            reasons.append("Multiple unusually large transactions.")

        if profile.get("emi_bounces", 0) > 0:
            reasons.append("EMI payment bounce history detected.")

        if not reasons:
            reasons.append("Transaction behaviour appears financially healthy.")

        return reasons
    


    @staticmethod
    def document(request: AIRequest) -> List[str]:

        reasons = []

        notes = (request.branch_notes or "").lower()
        docs = (request.document_text or "").lower()

        combined = notes + " " + docs
        
        keywords = {
            "fraud": "Possible fraud indicator detected.",
            "fake": "Potential fake document detected.",
            "forged": "Possible forged document detected.",
            "mismatch": "Document information mismatch detected.",
            "suspicious": "Suspicious wording found in verification notes.",
            "missing": "Required supporting documents are missing.",
            "unverifiable": "Some information could not be verified.",
            "unable to verify": "Verification process failed.",
            "employment could not verify": "Employment verification failed.",
            "address mismatch": "Address verification mismatch detected.",
        }
        

        for word in keywords:
            if word in combined:
                reasons.append(
                    f"Possible document concern: '{word}'."
                )

        if not reasons:
            reasons.append("No obvious document issues detected.")

        return reasons