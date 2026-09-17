from utils.ai_assistant import generate_audit_summary


test_claim = {
    "claim_id": "CLM004",
    "provider": "Provider C",
    "procedure_code": "99215",
    "billed_amount": 1200,
    "allowed_amount": 250,
    "units": 1,
    "variance": 950,
    "risk_score": 4,
    "final_risk": "High",
    "risk_reason": (
        "Billed amount is more than 2x the allowed amount"
    ),
    "audit_recommendation": (
        "Review billing justification and supporting documentation"
    )
}


result = generate_audit_summary(
    test_claim
)

print(result)