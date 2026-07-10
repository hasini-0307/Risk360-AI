"""
Common enums and shared constants used across schemas.
"""

from typing import Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

LoanType = Literal[
    "Home Loan",
    "Personal Loan",
    "Vehicle Loan",
    "Education Loan",
    "Business Loan",
    "Gold Loan",
    "Credit Card"
]