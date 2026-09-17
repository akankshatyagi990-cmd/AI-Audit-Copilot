import os

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


# --------------------------------------------------
# GENERATE SINGLE CLAIM AUDIT SUMMARY
# --------------------------------------------------

def generate_audit_summary(claim):

    client = get_gemini_client()

    if client is None:

        return (
            "Gemini API key was not found. "
            "Please check your .env file."
        )


    prompt = f"""
You are an AI assistant supporting a healthcare claims auditor.

Analyze only the information provided below.

Claim ID: {claim['claim_id']}
Provider: {claim['provider']}
Procedure Code: {claim['procedure_code']}
Billed Amount: ${claim['billed_amount']}
Allowed Amount: ${claim['allowed_amount']}
Units: {claim['units']}
Variance: ${claim['variance']}
Risk Score: {claim['risk_score']}
Risk Level: {claim['final_risk']}
Detected Reason: {claim['risk_reason']}
Current Audit Recommendation: {claim['audit_recommendation']}

Create a concise audit review summary.

Use exactly this structure:

Finding:
Explain the main issue detected.

Why It Matters:
Explain why this claim may require additional review.

Recommended Review:
Explain what the auditor should verify next.

Rules:
- Use only the information provided.
- Do not invent medical facts.
- Do not invent claim information.
- Do not approve or deny the claim.
- Do not state that fraud or overbilling is confirmed.
- Treat detected issues as risk indicators only.
- Keep the response concise.
"""


    try:

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return interaction.output_text


    except Exception as error:

        return (
            "AI summary could not be generated.\n\n"
            f"Error: {error}"
        )


# --------------------------------------------------
# ASK AUDIT COPILOT
# --------------------------------------------------

def ask_audit_copilot(
    question,
    audit_data
):

    client = get_gemini_client()

    if client is None:

        return (
            "Gemini API key was not found. "
            "Please check your .env file."
        )


    if not question.strip():

        return (
            "Please enter a question."
        )


    prompt = f"""
You are AI Audit Copilot.

You support healthcare claims auditors by answering questions
about claim data that has already been analyzed by the audit system.

Below is the audit dataset.

--------------------------------
AUDIT DATA
--------------------------------

{audit_data}

--------------------------------
AUDITOR QUESTION
--------------------------------

{question}

--------------------------------
INSTRUCTIONS
--------------------------------

Answer using ONLY the audit data provided above.

Important rules:

- Do not invent claims.
- Do not invent providers.
- Do not invent procedure codes.
- Do not invent amounts.
- Do not invent risk findings.
- Do not make medical conclusions.
- Do not approve or deny claims.
- Do not claim that fraud or overbilling is confirmed.
- Risk indicators are only signals for additional review.
- If information is not available in the dataset, clearly say so.
- Mention claim IDs when discussing individual claims.
- Explain why claims were flagged when relevant.
- Keep the answer concise and useful for an auditor.
- Use bullet points when discussing multiple claims.
- Treat all responses as audit decision support only.
"""


    try:

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return interaction.output_text


    except Exception as error:

        return (
            "Audit Copilot could not answer the question.\n\n"
            f"Error: {error}"
        )