import os

from dotenv import load_dotenv
from google import genai

from utils.rag_engine import (
    format_policy_context,
    get_policy_sources,
    retrieve_relevant_policy_chunks,
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


def get_gemini_client():

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception as error:

        print(
            f"Gemini client error: {error}"
        )

        return None


# ============================================================
# SAFE GEMINI CALL
# ============================================================

def run_gemini_prompt(prompt):

    client = get_gemini_client()

    if client is None:

        return (
            "Gemini is not configured. "
            "Please verify GEMINI_API_KEY in the .env file."
        )

    try:

        interaction = (
            client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
            )
        )

        return (
            interaction.output_text
        )

    except Exception as error:

        return (
            "Gemini request failed: "
            f"{error}"
        )


# ============================================================
# STANDARD CLAIM SUMMARY
# ============================================================

def generate_audit_summary(claim):

    prompt = f"""
You are an AI assistant supporting a healthcare claims auditor.

Use only the structured claim findings supplied below.

CLAIM INFORMATION

Claim ID:
{claim.get("claim_id")}

Provider:
{claim.get("provider")}

Procedure Code:
{claim.get("procedure_code")}

Billed Amount:
${claim.get("billed_amount")}

Allowed Amount:
${claim.get("allowed_amount")}

Units:
{claim.get("units")}

Financial Variance:
${claim.get("variance")}

Risk Score:
{claim.get("risk_score")}

Risk Level:
{claim.get("final_risk")}

Detected Risk Evidence:
{claim.get("risk_reason")}

Current Audit Recommendation:
{claim.get("audit_recommendation")}


Create a concise audit-support summary using exactly
these three sections:

Finding

Why It Matters

Recommended Review


IMPORTANT RULES

Do not approve or deny the claim.

Do not state that fraud has occurred.

Do not state that overbilling has been confirmed.

Do not make clinical conclusions.

Do not invent medical facts.

Do not invent policy requirements.

Describe the risk findings only as indicators
that may require human auditor review.

Keep the answer concise and professional.
"""

    return run_gemini_prompt(
        prompt
    )


# ============================================================
# STANDARD AUDIT COPILOT
# ============================================================

def ask_audit_copilot(
    question,
    audit_data,
):

    prompt = f"""
You are an AI Audit Copilot supporting a healthcare claims auditor.

Answer the user's question using ONLY the audit dataset
provided below.

AUDIT DATA

{audit_data}


USER QUESTION

{question}


RULES

Use only information contained in the supplied dataset.

Do not invent claims.

Do not invent providers.

Do not invent procedure codes.

Do not invent amounts.

Do not invent risk findings.

Do not make medical conclusions.

Do not approve or deny claims.

Do not confirm fraud.

Do not confirm overbilling.

When discussing unusual claims, describe them as
risk indicators that may require auditor review.

Mention specific claim IDs when relevant.

Use bullet points when discussing multiple claims.

Keep the response concise and professional.
"""

    return run_gemini_prompt(
        prompt
    )


# ============================================================
# BUILD POLICY RETRIEVAL QUERY FOR A CLAIM
# ============================================================

def build_claim_policy_query(
    claim,
):

    query_parts = [
        "healthcare claims audit policy",
    ]


    billed_amount = claim.get(
        "billed_amount",
        0,
    )

    allowed_amount = claim.get(
        "allowed_amount",
        0,
    )

    units = claim.get(
        "units",
        0,
    )

    variance = claim.get(
        "variance",
        0,
    )

    risk_reason = str(
        claim.get(
            "risk_reason",
            "",
        )
    )

    recommendation = str(
        claim.get(
            "audit_recommendation",
            "",
        )
    )


    if (
        allowed_amount
        and
        billed_amount
        > allowed_amount * 2
    ):

        query_parts.append(
            "billed amount significantly higher than allowed amount"
        )

        query_parts.append(
            "billing variance documentation review"
        )


    if units and units > 3:

        query_parts.append(
            "multiple billed units utilization review"
        )


    if variance and variance > 0:

        query_parts.append(
            "financial variance audit review"
        )


    if risk_reason:

        query_parts.append(
            risk_reason
        )


    if recommendation:

        query_parts.append(
            recommendation
        )


    return " ".join(
        query_parts
    )


# ============================================================
# RAG: POLICY-GROUNDED CLAIM REVIEW
# ============================================================

def generate_policy_grounded_audit_summary(
    claim,
    top_k=4,
):

    retrieval_query = (
        build_claim_policy_query(
            claim
        )
    )


    results = (
        retrieve_relevant_policy_chunks(
            query=retrieval_query,
            top_k=top_k,
        )
    )


    policy_context = (
        format_policy_context(
            results
        )
    )


    sources = (
        get_policy_sources(
            results
        )
    )


    if not results:

        return {
            "answer": (
                "No relevant policy passages were retrieved. "
                "The claim can still be reviewed using the "
                "rule-based and statistical findings."
            ),
            "sources": [],
            "retrieval_query":
                retrieval_query,
        }


    prompt = f"""
You are an AI assistant supporting a healthcare claims auditor.

Your task is to explain the claim using:

1. Structured findings already produced by the audit engine.
2. Policy passages retrieved from the audit knowledge base.

You must remain grounded in the supplied information.


CLAIM FINDINGS

Claim ID:
{claim.get("claim_id")}

Provider:
{claim.get("provider")}

Procedure Code:
{claim.get("procedure_code")}

Billed Amount:
${claim.get("billed_amount")}

Allowed Amount:
${claim.get("allowed_amount")}

Units:
{claim.get("units")}

Financial Variance:
${claim.get("variance")}

Risk Score:
{claim.get("risk_score")}

Risk Level:
{claim.get("final_risk")}

Risk Evidence:
{claim.get("risk_reason")}

Current Audit Recommendation:
{claim.get("audit_recommendation")}


RETRIEVED POLICY CONTEXT

{policy_context}


Create the response using exactly these sections:

Finding

Policy Context

Recommended Review


FINDING

Explain why the claim was identified by the audit engine.

Do not treat the policy text as evidence that the claim
is incorrect.


POLICY CONTEXT

Explain only the relevant guidance found in the
retrieved policy passages.

When referring to policy guidance, include the exact
policy filename and page number supplied in the
retrieved context.

Example:

Source: Example_Policy.pdf, page 2


RECOMMENDED REVIEW

Describe reasonable human auditor review steps based
on the claim findings and retrieved policy guidance.


STRICT RULES

Use only the supplied claim information and retrieved
policy context.

Do not invent policy text.

Do not invent policy requirements.

Do not invent page numbers.

Do not invent source names.

Do not approve or deny the claim.

Do not confirm fraud.

Do not confirm overbilling.

Do not state that a coding error has been confirmed.

Do not make clinical conclusions.

Do not treat statistical anomalies as proof that a
claim is incorrect.

If the retrieved policy does not clearly support a
statement, do not make that statement.

Keep the response concise, professional and suitable
for an auditor.
"""


    answer = run_gemini_prompt(
        prompt
    )


    return {
        "answer":
            answer,

        "sources":
            sources,

        "retrieval_query":
            retrieval_query,
    }


# ============================================================
# RAG: POLICY-GROUNDED AUDIT COPILOT
# ============================================================

def ask_policy_audit_copilot(
    question,
    audit_data,
    top_k=4,
):

    results = (
        retrieve_relevant_policy_chunks(
            query=question,
            top_k=top_k,
        )
    )


    policy_context = (
        format_policy_context(
            results
        )
    )


    sources = (
        get_policy_sources(
            results
        )
    )


    prompt = f"""
You are an AI Audit Copilot supporting a healthcare claims auditor.

Answer the user's question using BOTH:

1. The analyzed audit dataset.
2. Retrieved policy passages from the audit knowledge base.

Do not use outside knowledge.


AUDIT DATA

{audit_data}


RETRIEVED POLICY CONTEXT

{policy_context}


USER QUESTION

{question}


RESPONSE RULES

Answer only from the supplied audit data and retrieved
policy passages.

Clearly distinguish:

Audit Finding

Policy Guidance

Recommended Review


When policy guidance is used, identify the exact
policy filename and page number.

Example:

Source: Example_Policy.pdf, page 2


Do not invent claims.

Do not invent providers.

Do not invent procedure codes.

Do not invent amounts.

Do not invent policies.

Do not invent source filenames.

Do not invent page numbers.

Do not approve or deny claims.

Do not confirm fraud.

Do not confirm overbilling.

Do not make clinical conclusions.

Do not state that a statistical anomaly proves that
a claim is incorrect.

If the policy context does not answer part of the
question, clearly say that the retrieved policy
context does not provide that information.

Keep the answer concise and professional.
"""


    answer = run_gemini_prompt(
        prompt
    )


    return {
        "answer":
            answer,

        "sources":
            sources,
    }


# ============================================================
# FORMAT RETRIEVED SOURCES FOR STREAMLIT
# ============================================================

def format_source_list(
    sources,
):

    if not sources:

        return (
            "No policy sources retrieved."
        )


    source_lines = []


    for source in sources:

        source_lines.append(
            f"- {source['source']} "
            f"(page {source['page']})"
        )


    return "\n".join(
        source_lines
    )