import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.ai_assistant import (
    generate_audit_summary,
    ask_audit_copilot,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Audit Copilot",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(120,120,120,0.15);
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .app-subtitle {
        opacity: 0.7;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .claim-card {
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .small-note {
        font-size: 0.86rem;
        opacity: 0.68;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def calculate_rule_risk(row):

    reasons = []

    if row["billed_amount"] > row["allowed_amount"] * 2:
        reasons.append(
            "Billed amount is more than 2x the allowed amount"
        )

    if row["units"] > 3:
        reasons.append(
            "Units are greater than 3"
        )

    if reasons:
        return "High Risk", "; ".join(reasons)

    return "Normal", "No rule-based issue detected"


def calculate_final_risk(row):

    score = 0
    reasons = []

    if row["rule_risk"] == "High Risk":
        score += 2
        reasons.append(row["rule_reason"])

    if row["anomaly_score"] == 1:

        score += 1

        reasons.append(
            "Statistical anomaly detected in: "
            + row["anomaly_reason"]
        )

    elif row["anomaly_score"] >= 2:

        score += 2

        reasons.append(
            "Multiple statistical anomalies detected in: "
            + row["anomaly_reason"]
        )

    if score >= 3:
        risk_level = "High"

    elif score >= 1:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    if not reasons:

        reasons.append(
            "No significant rule-based or statistical risk detected"
        )

    return score, risk_level, "; ".join(reasons)


def generate_recommendation(row):

    recommendations = []

    if row["billed_amount"] > row["allowed_amount"] * 2:

        recommendations.append(
            "Review billing justification and supporting documentation"
        )

    if row["units"] > 3:

        recommendations.append(
            "Verify billed units against supporting documentation"
        )

    if "billed_allowed_ratio" in row["anomaly_reason"]:

        recommendations.append(
            "Review the billed-to-allowed amount ratio"
        )

    if "variance" in row["anomaly_reason"]:

        recommendations.append(
            "Investigate the unusual billed-versus-allowed variance"
        )

    if "billed_amount" in row["anomaly_reason"]:

        recommendations.append(
            "Compare billed amount with similar claims and historical patterns"
        )

    if "allowed_amount" in row["anomaly_reason"]:

        recommendations.append(
            "Validate the allowed amount against applicable reimbursement rules"
        )

    if row["final_risk"] == "Low":

        return (
            "No immediate audit action required. "
            "Continue standard review."
        )

    if not recommendations:

        return (
            "Perform manual review of the claim "
            "and supporting documentation."
        )

    return "; ".join(recommendations)


def prepare_audit_data(df):

    required_columns = [
        "claim_id",
        "provider",
        "procedure_code",
        "billed_amount",
        "allowed_amount",
        "units",
        "status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

        st.stop()

    numeric_columns = [
        "billed_amount",
        "allowed_amount",
        "units",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    ).copy()

    if df.empty:

        st.error(
            "No valid claims remain after data validation."
        )

        st.stop()

    # --------------------------------------------------------
    # BASIC CALCULATIONS
    # --------------------------------------------------------

    df["variance"] = (
        df["billed_amount"]
        - df["allowed_amount"]
    )

    df["billed_allowed_ratio"] = np.where(
        df["allowed_amount"] != 0,
        df["billed_amount"]
        / df["allowed_amount"],
        0,
    )

    # --------------------------------------------------------
    # RULE DETECTION
    # --------------------------------------------------------

    rule_results = df.apply(
        calculate_rule_risk,
        axis=1,
        result_type="expand",
    )

    df["rule_risk"] = rule_results[0]

    df["rule_reason"] = rule_results[1]

    # --------------------------------------------------------
    # STATISTICAL ANOMALY DETECTION
    # --------------------------------------------------------

    anomaly_features = [
        "billed_amount",
        "allowed_amount",
        "units",
        "variance",
        "billed_allowed_ratio",
    ]

    df["anomaly_score"] = 0
    df["anomaly_reason"] = ""

    for feature in anomaly_features:

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)

        anomaly_mask = (
            (df[feature] < lower_limit)
            |
            (df[feature] > upper_limit)
        )

        df.loc[
            anomaly_mask,
            "anomaly_score",
        ] += 1

        df.loc[
            anomaly_mask,
            "anomaly_reason",
        ] += feature + ", "

    df["anomaly_reason"] = (
        df["anomaly_reason"]
        .str.rstrip(", ")
    )

    df["anomaly_status"] = np.where(
        df["anomaly_score"] > 0,
        "Anomaly",
        "Normal",
    )

    # --------------------------------------------------------
    # FINAL RISK
    # --------------------------------------------------------

    final_results = df.apply(
        calculate_final_risk,
        axis=1,
        result_type="expand",
    )

    df["risk_score"] = final_results[0]
    df["final_risk"] = final_results[1]
    df["risk_reason"] = final_results[2]

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    df["audit_recommendation"] = df.apply(
        generate_recommendation,
        axis=1,
    )

    return df


# ============================================================
# SIDEBAR - DATA SOURCE
# ============================================================

st.sidebar.title("AI Audit Copilot")

st.sidebar.caption(
    "Healthcare audit intelligence prototype"
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Audit CSV",
    type=["csv"],
)


if uploaded_file is not None:

    raw_df = pd.read_csv(uploaded_file)

    data_source = uploaded_file.name

    st.sidebar.success(
        "Dataset loaded"
    )

else:

    default_file = (
        PROJECT_ROOT
        / "data"
        / "audit_data.csv"
    )

    raw_df = pd.read_csv(default_file)

    data_source = "Sample Audit Dataset"

    st.sidebar.info(
        "Using sample dataset"
    )


df = prepare_audit_data(raw_df)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Claims Analysis",
        "Priority Audit Queue",
        "Claim Review",
        "AI Audit Copilot",
        "Executive Report",
    ],
)


st.sidebar.divider()

st.sidebar.caption(
    f"Data source: {data_source}"
)

st.sidebar.caption(
    f"{len(df)} valid claims loaded"
)


# ============================================================
# GLOBAL METRICS
# ============================================================

total_claims = len(df)

high_risk_claims = len(
    df[
        df["final_risk"]
        == "High"
    ]
)

medium_risk_claims = len(
    df[
        df["final_risk"]
        == "Medium"
    ]
)

low_risk_claims = len(
    df[
        df["final_risk"]
        == "Low"
    ]
)

total_billed = (
    df["billed_amount"].sum()
)

total_allowed = (
    df["allowed_amount"].sum()
)

total_variance = (
    df["variance"].sum()
)


# ============================================================
# DATA FOR GEMINI
# ============================================================

ai_data_columns = [
    "claim_id",
    "provider",
    "procedure_code",
    "billed_amount",
    "allowed_amount",
    "units",
    "variance",
    "anomaly_score",
    "risk_score",
    "final_risk",
    "risk_reason",
    "audit_recommendation",
]


audit_data_for_ai = (
    df[
        ai_data_columns
    ]
    .to_csv(
        index=False
    )
)


# ============================================================
# SESSION STATE
# ============================================================

if "ai_summaries" not in st.session_state:

    st.session_state[
        "ai_summaries"
    ] = {}


if "copilot_history" not in st.session_state:

    st.session_state[
        "copilot_history"
    ] = []


if "executive_report" not in st.session_state:

    st.session_state[
        "executive_report"
    ] = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">AI Audit Copilot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'AI-assisted claims analysis, risk intelligence and audit decision support.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader(
        "Audit Overview"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Claims Analyzed",
        total_claims,
    )

    col2.metric(
        "High Risk",
        high_risk_claims,
    )

    col3.metric(
        "Medium Risk",
        medium_risk_claims,
    )

    col4.metric(
        "Low Risk",
        low_risk_claims,
    )


    finance1, finance2, finance3 = (
        st.columns(3)
    )

    finance1.metric(
        "Total Billed",
        f"${total_billed:,.2f}",
    )

    finance2.metric(
        "Total Allowed",
        f"${total_allowed:,.2f}",
    )

    finance3.metric(
        "Total Variance",
        f"${total_variance:,.2f}",
    )


    st.divider()


    chart_col1, chart_col2 = (
        st.columns(2)
    )


    with chart_col1:

        risk_distribution = (
            df["final_risk"]
            .value_counts()
            .reset_index()
        )

        risk_distribution.columns = [
            "Risk Level",
            "Claims",
        ]

        risk_chart = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Claims",
            title="Risk Distribution",
            category_orders={
                "Risk Level": [
                    "High",
                    "Medium",
                    "Low",
                ]
            },
        )

        st.plotly_chart(
            risk_chart,
            use_container_width=True,
        )


    with chart_col2:

        provider_risk = (
            df[
                df["final_risk"]
                != "Low"
            ]
            .groupby(
                "provider"
            )
            .size()
            .reset_index(
                name="Risk Claims"
            )
        )

        if not provider_risk.empty:

            provider_chart = px.bar(
                provider_risk,
                x="provider",
                y="Risk Claims",
                title="Risk Claims by Provider",
            )

            st.plotly_chart(
                provider_chart,
                use_container_width=True,
            )

        else:

            st.info(
                "No elevated-risk provider activity detected."
            )


    st.subheader(
        "Billed vs Allowed Amount"
    )

    amount_chart = px.scatter(
        df,
        x="allowed_amount",
        y="billed_amount",
        hover_name="claim_id",
        size="units",
        symbol="final_risk",
        title="Claim Billing Pattern",
    )

    st.plotly_chart(
        amount_chart,
        use_container_width=True,
    )


    st.subheader(
        "Top Priority Claims"
    )

    top_priority = (
        df[
            df["final_risk"].isin(
                [
                    "High",
                    "Medium",
                ]
            )
        ]
        .sort_values(
            by=[
                "risk_score",
                "variance",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(5)
    )

    if not top_priority.empty:

        st.dataframe(
            top_priority[
                [
                    "claim_id",
                    "provider",
                    "procedure_code",
                    "billed_amount",
                    "allowed_amount",
                    "variance",
                    "risk_score",
                    "final_risk",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No priority claims detected."
        )


# ============================================================
# CLAIMS ANALYSIS
# ============================================================

elif page == "Claims Analysis":

    st.subheader(
        "Claims Analysis"
    )

    st.caption(
        "Search, filter and review analyzed claims."
    )


    filter1, filter2, filter3 = (
        st.columns(3)
    )


    with filter1:

        selected_risks = (
            st.multiselect(
                "Risk Level",
                options=[
                    "High",
                    "Medium",
                    "Low",
                ],
                default=[
                    "High",
                    "Medium",
                    "Low",
                ],
            )
        )


    with filter2:

        providers = sorted(
            df[
                "provider"
            ]
            .astype(str)
            .unique()
        )

        selected_providers = (
            st.multiselect(
                "Provider",
                options=providers,
                default=providers,
            )
        )


    with filter3:

        search_claim = (
            st.text_input(
                "Search Claim ID",
                placeholder="Example: CLM004",
            )
        )


    filtered_df = df[
        df["final_risk"].isin(
            selected_risks
        )
        &
        df["provider"].astype(
            str
        ).isin(
            selected_providers
        )
    ].copy()


    if search_claim.strip():

        filtered_df = filtered_df[
            filtered_df[
                "claim_id"
            ]
            .astype(str)
            .str.contains(
                search_claim,
                case=False,
                na=False,
            )
        ]


    st.write(
        f"**{len(filtered_df)} claims found**"
    )


    claim_columns = [
        "claim_id",
        "provider",
        "procedure_code",
        "billed_amount",
        "allowed_amount",
        "units",
        "variance",
        "anomaly_score",
        "risk_score",
        "final_risk",
        "risk_reason",
        "audit_recommendation",
    ]


    st.dataframe(
        filtered_df[
            claim_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


    csv_download = (
        filtered_df[
            claim_columns
        ]
        .to_csv(
            index=False
        )
    )


    st.download_button(
        "Download Filtered Claims",
        data=csv_download,
        file_name=(
            "filtered_audit_claims.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# PRIORITY AUDIT QUEUE
# ============================================================

elif page == "Priority Audit Queue":

    st.subheader(
        "Priority Audit Queue"
    )

    st.caption(
        "Claims ranked for auditor review using risk level, risk score and financial variance."
    )


    priority_df = df[
        df["final_risk"].isin(
            [
                "High",
                "Medium",
            ]
        )
    ].copy()


    risk_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }


    priority_df[
        "risk_order"
    ] = priority_df[
        "final_risk"
    ].map(
        risk_order
    )


    priority_df = (
        priority_df
        .sort_values(
            by=[
                "risk_order",
                "risk_score",
                "variance",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )


    if not priority_df.empty:

        priority_display = (
            priority_df[
                [
                    "claim_id",
                    "provider",
                    "procedure_code",
                    "billed_amount",
                    "allowed_amount",
                    "variance",
                    "risk_score",
                    "final_risk",
                    "risk_reason",
                    "audit_recommendation",
                ]
            ]
        )

        st.dataframe(
            priority_display,
            use_container_width=True,
            hide_index=True,
        )


        priority_csv = (
            priority_display
            .to_csv(
                index=False
            )
        )


        st.download_button(
            "Download Priority Queue",
            data=priority_csv,
            file_name=(
                "priority_audit_queue.csv"
            ),
            mime="text/csv",
        )

    else:

        st.success(
            "No claims currently require priority review."
        )


# ============================================================
# CLAIM REVIEW
# ============================================================

elif page == "Claim Review":

    st.subheader(
        "Claim Review Assistant"
    )

    st.caption(
        "Review an individual claim, understand its risk indicators and generate an AI-assisted audit summary."
    )


    selected_claim = st.selectbox(
        "Select Claim",
        df[
            "claim_id"
        ].tolist(),
    )


    selected_row = df[
        df["claim_id"]
        == selected_claim
    ].iloc[0]


    info1, info2, info3, info4 = (
        st.columns(4)
    )


    info1.metric(
        "Risk Level",
        selected_row[
            "final_risk"
        ],
    )

    info2.metric(
        "Risk Score",
        int(
            selected_row[
                "risk_score"
            ]
        ),
    )

    info3.metric(
        "Billed Amount",
        f"${selected_row['billed_amount']:,.2f}",
    )

    info4.metric(
        "Variance",
        f"${selected_row['variance']:,.2f}",
    )


    st.markdown(
        "### Claim Details"
    )


    detail1, detail2 = (
        st.columns(2)
    )


    with detail1:

        st.write(
            "**Claim ID:**",
            selected_row[
                "claim_id"
            ],
        )

        st.write(
            "**Provider:**",
            selected_row[
                "provider"
            ],
        )

        st.write(
            "**Procedure Code:**",
            selected_row[
                "procedure_code"
            ],
        )

        st.write(
            "**Units:**",
            selected_row[
                "units"
            ],
        )


    with detail2:

        st.write(
            "**Billed Amount:**",
            f"${selected_row['billed_amount']:,.2f}",
        )

        st.write(
            "**Allowed Amount:**",
            f"${selected_row['allowed_amount']:,.2f}",
        )

        st.write(
            "**Anomaly Score:**",
            selected_row[
                "anomaly_score"
            ],
        )

        st.write(
            "**Original Status:**",
            selected_row[
                "status"
            ],
        )


    st.markdown(
        "### Risk Explanation"
    )

    st.info(
        selected_row[
            "risk_reason"
        ]
    )


    st.markdown(
        "### Recommended Audit Action"
    )

    st.warning(
        selected_row[
            "audit_recommendation"
        ]
    )


    st.divider()


    st.markdown(
        "### AI Audit Summary"
    )


    if st.button(
        "Generate AI Audit Summary",
        type="primary",
    ):

        claim_data = {

            "claim_id":
                selected_row[
                    "claim_id"
                ],

            "provider":
                selected_row[
                    "provider"
                ],

            "procedure_code":
                selected_row[
                    "procedure_code"
                ],

            "billed_amount":
                selected_row[
                    "billed_amount"
                ],

            "allowed_amount":
                selected_row[
                    "allowed_amount"
                ],

            "units":
                selected_row[
                    "units"
                ],

            "variance":
                selected_row[
                    "variance"
                ],

            "risk_score":
                selected_row[
                    "risk_score"
                ],

            "final_risk":
                selected_row[
                    "final_risk"
                ],

            "risk_reason":
                selected_row[
                    "risk_reason"
                ],

            "audit_recommendation":
                selected_row[
                    "audit_recommendation"
                ],
        }


        with st.spinner(
            "Gemini is analyzing the claim..."
        ):

            ai_summary = (
                generate_audit_summary(
                    claim_data
                )
            )


        st.session_state[
            "ai_summaries"
        ][
            selected_claim
        ] = ai_summary


    if (
        selected_claim
        in st.session_state[
            "ai_summaries"
        ]
    ):

        st.success(
            "AI analysis completed."
        )

        st.markdown(
            st.session_state[
                "ai_summaries"
            ][
                selected_claim
            ]
        )


    st.caption(
        "AI-generated content is intended for audit support and requires human review."
    )


# ============================================================
# AI AUDIT COPILOT
# ============================================================

elif page == "AI Audit Copilot":

    st.subheader(
        "Ask Audit Copilot"
    )

    st.caption(
        "Ask natural-language questions about the currently loaded audit dataset."
    )


    example1, example2, example3 = (
        st.columns(3)
    )


    with example1:

        st.info(
            "Which claims are high risk?"
        )


    with example2:

        st.info(
            "Which provider has the most risky claims?"
        )


    with example3:

        st.info(
            "What should I review first?"
        )


    question = st.text_input(
        "Ask a question",
        placeholder=(
            "Example: Why was CLM004 flagged?"
        ),
    )


    button1, button2 = (
        st.columns(
            [
                1,
                4,
            ]
        )
    )


    with button1:

        ask_button = st.button(
            "Ask Copilot",
            type="primary",
        )


    with button2:

        clear_button = st.button(
            "Clear Conversation"
        )


    if clear_button:

        st.session_state[
            "copilot_history"
        ] = []

        st.rerun()


    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Audit Copilot is analyzing the dataset..."
            ):

                answer = (
                    ask_audit_copilot(
                        question,
                        audit_data_for_ai,
                    )
                )


            st.session_state[
                "copilot_history"
            ].append(
                {
                    "question":
                        question,

                    "answer":
                        answer,
                }
            )


    if st.session_state[
        "copilot_history"
    ]:

        st.divider()

        for conversation in (
            st.session_state[
                "copilot_history"
            ]
        ):

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    conversation[
                        "question"
                    ]
                )


            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    conversation[
                        "answer"
                    ]
                )

    else:

        st.info(
            "No questions asked yet."
        )


# ============================================================
# EXECUTIVE REPORT
# ============================================================

elif page == "Executive Report":

    st.subheader(
        "AI Executive Audit Report"
    )

    st.caption(
        "Generate an AI-assisted management summary of the complete audit dataset."
    )


    report1, report2, report3 = (
        st.columns(3)
    )


    report1.metric(
        "Claims",
        total_claims,
    )

    report2.metric(
        "Elevated Risk",
        high_risk_claims
        + medium_risk_claims,
    )

    report3.metric(
        "Total Variance",
        f"${total_variance:,.2f}",
    )


    st.divider()


    if st.button(
        "Generate Executive Audit Report",
        type="primary",
    ):

        report_prompt = f"""
Create an executive audit report for the current claims dataset.

SUMMARY STATISTICS

Claims analyzed: {total_claims}
High-risk claims: {high_risk_claims}
Medium-risk claims: {medium_risk_claims}
Low-risk claims: {low_risk_claims}

Total billed: ${total_billed:,.2f}
Total allowed: ${total_allowed:,.2f}
Total variance: ${total_variance:,.2f}

Create the report using these sections:

Executive Summary

Highest Priority Findings

Provider-Level Observations

Financial Variance Observations

Recommended Audit Focus

Mention specific claim IDs where relevant.

Important:
Do not state that fraud, overbilling, coding errors or improper claims
are confirmed.

Describe findings only as indicators requiring auditor review.

Use only the dataset supplied.
"""


        with st.spinner(
            "Gemini is generating the executive audit report..."
        ):

            report = (
                ask_audit_copilot(
                    report_prompt,
                    audit_data_for_ai,
                )
            )


        st.session_state[
            "executive_report"
        ] = report


    if st.session_state[
        "executive_report"
    ]:

        st.success(
            "Executive report generated."
        )

        st.markdown(
            st.session_state[
                "executive_report"
            ]
        )


        st.download_button(
            "Download Executive Report",
            data=st.session_state[
                "executive_report"
            ],
            file_name=(
                "ai_audit_executive_report.txt"
            ),
            mime="text/plain",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Audit Copilot Prototype • "
    "Use synthetic or properly de-identified data only. "
    "Risk indicators and AI outputs are decision-support signals "
    "and do not represent final claim determinations."
)