import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.ai_assistant import (
    ask_audit_copilot,
    generate_audit_summary,
    generate_policy_grounded_audit_summary,
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
# KNOWLEDGE BASE STATUS
# ============================================================

POLICY_DIRECTORY = (
    PROJECT_ROOT
    / "knowledge_base"
    / "policies"
)

policy_files = []

if POLICY_DIRECTORY.exists():
    policy_files = sorted(
        POLICY_DIRECTORY.glob("*.pdf")
    )

policy_count = len(policy_files)


# ============================================================
# AUDIT ENGINE
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

        reasons.append(
            row["rule_reason"]
        )

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

    return (
        score,
        risk_level,
        "; ".join(reasons),
    )


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
            "Compare the billed amount with similar claims "
            "and historical patterns"
        )

    if "allowed_amount" in row["anomaly_reason"]:

        recommendations.append(
            "Validate the allowed amount against applicable "
            "reimbursement rules"
        )

    if row["final_risk"] == "Low":

        return (
            "No immediate elevated-risk action identified. "
            "Continue the standard audit workflow."
        )

    if not recommendations:

        return (
            "Perform a manual review of the claim "
            "and supporting documentation."
        )

    return "; ".join(
        recommendations
    )


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

    df["variance_percentage"] = np.where(
        df["allowed_amount"] != 0,
        (
            df["variance"]
            / df["allowed_amount"]
        )
        * 100,
        0,
    )


    # --------------------------------------------------------
    # RULE ENGINE
    # --------------------------------------------------------

    rule_results = df.apply(
        calculate_rule_risk,
        axis=1,
        result_type="expand",
    )

    df["rule_risk"] = (
        rule_results[0]
    )

    df["rule_reason"] = (
        rule_results[1]
    )


    # --------------------------------------------------------
    # IQR STATISTICAL ANOMALY DETECTION
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

        q1 = df[
            feature
        ].quantile(
            0.25
        )

        q3 = df[
            feature
        ].quantile(
            0.75
        )

        iqr = q3 - q1

        lower_limit = (
            q1 - (1.5 * iqr)
        )

        upper_limit = (
            q3 + (1.5 * iqr)
        )

        anomaly_mask = (
            (
                df[feature]
                < lower_limit
            )
            |
            (
                df[feature]
                > upper_limit
            )
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

    df["risk_score"] = (
        final_results[0]
    )

    df["final_risk"] = (
        final_results[1]
    )

    df["risk_reason"] = (
        final_results[2]
    )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    df["audit_recommendation"] = df.apply(
        generate_recommendation,
        axis=1,
    )

    return df


# ============================================================
# SUPPORT FUNCTIONS
# ============================================================

def get_detected_signals(row):

    signals = []

    if (
        row["billed_amount"]
        > row["allowed_amount"] * 2
    ):

        signals.append(
            "Large billed-to-allowed amount difference"
        )

    if row["units"] > 3:

        signals.append(
            "Unusual number of billed units"
        )


    readable_names = {

        "billed_amount":
            "Billed amount is statistically unusual",

        "allowed_amount":
            "Allowed amount differs from the dataset pattern",

        "units":
            "Unit count is statistically unusual",

        "variance":
            "Billing variance is statistically unusual",

        "billed_allowed_ratio":
            "Billed-to-allowed ratio is statistically unusual",
    }


    if row["anomaly_reason"]:

        for feature in (
            row["anomaly_reason"]
            .split(",")
        ):

            feature = (
                feature.strip()
            )

            if not feature:
                continue

            signal = readable_names.get(
                feature,
                feature,
            )

            if signal not in signals:

                signals.append(
                    signal
                )

    return signals


def get_review_actions(row):

    recommendation = str(
        row["audit_recommendation"]
    )

    return [
        item.strip()
        for item in recommendation.split(";")
        if item.strip()
    ]


def open_claim_review(claim_id):

    st.session_state[
        "claim_review_selector"
    ] = claim_id

    st.session_state[
        "page_navigation"
    ] = "Claim Review"


def set_copilot_question(question):

    st.session_state[
        "copilot_question"
    ] = question


def clear_copilot():

    st.session_state[
        "copilot_history"
    ] = []

    st.session_state[
        "copilot_question"
    ] = ""


def show_risk_status(risk):

    if risk == "High":

        st.error(
            "HIGH RISK"
        )

    elif risk == "Medium":

        st.warning(
            "MEDIUM RISK"
        )

    else:

        st.success(
            "LOW RISK"
        )


def build_claim_dictionary(
    selected_row,
):

    return {

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


# ============================================================
# SESSION STATE
# ============================================================

if "page_navigation" not in st.session_state:

    st.session_state[
        "page_navigation"
    ] = "Dashboard"


if "ai_summaries" not in st.session_state:

    st.session_state[
        "ai_summaries"
    ] = {}


if "rag_summaries" not in st.session_state:

    st.session_state[
        "rag_summaries"
    ] = {}


if "copilot_history" not in st.session_state:

    st.session_state[
        "copilot_history"
    ] = []


if "copilot_question" not in st.session_state:

    st.session_state[
        "copilot_question"
    ] = ""


if "executive_report" not in st.session_state:

    st.session_state[
        "executive_report"
    ] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "AI Audit Copilot"
)

st.sidebar.caption(
    "Healthcare Audit Intelligence"
)

st.sidebar.divider()


uploaded_file = st.sidebar.file_uploader(
    "Upload Audit CSV",
    type=[
        "csv"
    ],
    help=(
        "Upload a synthetic or properly "
        "de-identified audit CSV."
    ),
)


if uploaded_file is not None:

    raw_df = pd.read_csv(
        uploaded_file
    )

    data_source = (
        uploaded_file.name
    )

else:

    default_file = (
        PROJECT_ROOT
        / "data"
        / "audit_data.csv"
    )

    raw_df = pd.read_csv(
        default_file
    )

    data_source = (
        "Sample Audit Dataset"
    )


df = prepare_audit_data(
    raw_df
)


with st.sidebar.container(
    border=True
):

    st.caption(
        "CURRENT DATASET"
    )

    st.write(
        f"**{data_source}**"
    )

    st.write(
        f"{len(df)} claims loaded"
    )

    st.success(
        "Analysis complete"
    )


# ============================================================
# POLICY KNOWLEDGE BASE STATUS
# ============================================================

with st.sidebar.container(
    border=True
):

    st.caption(
        "POLICY KNOWLEDGE BASE"
    )

    if policy_count > 0:

        st.write(
            f"**{policy_count} policy PDF(s)**"
        )

        st.success(
            "RAG knowledge base ready"
        )

    else:

        st.write(
            "**No policy PDFs detected**"
        )

        st.warning(
            "Policy-grounded review unavailable"
        )


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
    key="page_navigation",
)


st.sidebar.divider()


with st.sidebar.expander(
    "Prototype Architecture",
    expanded=False,
):

    st.write(
        "**Rule-Based Detection**"
    )

    st.caption(
        "Deterministic prototype checks identify "
        "predefined audit signals."
    )


    st.write(
        "**Statistical Detection**"
    )

    st.caption(
        "IQR analysis identifies values outside "
        "expected dataset patterns."
    )


    st.write(
        "**RAG Retrieval**"
    )

    st.caption(
        "Relevant passages are retrieved from "
        "the local audit policy knowledge base."
    )


    st.write(
        "**Generative AI**"
    )

    st.caption(
        "Gemini explains structured findings and "
        "retrieved policy context."
    )


    st.info(
        "All outputs are decision-support signals. "
        "Human auditor review remains required."
    )


st.sidebar.caption(
    "Prototype • Human review required"
)


# ============================================================
# GLOBAL METRICS
# ============================================================

total_claims = len(
    df
)


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
    df[
        "billed_amount"
    ].sum()
)


total_allowed = (
    df[
        "allowed_amount"
    ].sum()
)


total_variance = (
    df[
        "variance"
    ].sum()
)


priority_df_global = df[
    df[
        "final_risk"
    ].isin(
        [
            "High",
            "Medium",
        ]
    )
].copy()


priority_variance = (
    priority_df_global[
        "variance"
    ].sum()
)


multi_signal_claims = len(
    df[
        df[
            "anomaly_score"
        ]
        >= 2
    ]
)


provider_priority_counts = (
    priority_df_global
    .groupby(
        "provider"
    )
    .size()
    .sort_values(
        ascending=False
    )
)


if not provider_priority_counts.empty:

    top_provider = (
        provider_priority_counts
        .index[0]
    )

    top_provider_count = int(
        provider_priority_counts
        .iloc[0]
    )

else:

    top_provider = "None"

    top_provider_count = 0


highest_variance_claim = (
    df.sort_values(
        by="variance",
        ascending=False,
    )
    .iloc[0]
)


# ============================================================
# GEMINI DATA CONTEXT
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
# GLOBAL HEADER
# ============================================================

st.title(
    "AI Audit Copilot"
)

st.caption(
    "AI-assisted claims analysis, risk intelligence "
    "and auditor decision support."
)


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    st.header(
        "Audit Overview"
    )

    st.caption(
        "Monitor claim risk, financial variance and "
        "audit priorities from a single workspace."
    )


    k1, k2, k3, k4 = (
        st.columns(4)
    )


    with k1:

        with st.container(
            border=True
        ):

            st.metric(
                "Claims Analyzed",
                total_claims,
            )

            st.caption(
                "Valid claims in the current dataset"
            )


    with k2:

        with st.container(
            border=True
        ):

            st.metric(
                "High Risk",
                high_risk_claims,
            )

            st.caption(
                "Claims requiring priority review"
            )


    with k3:

        with st.container(
            border=True
        ):

            st.metric(
                "Medium Risk",
                medium_risk_claims,
            )

            st.caption(
                "Claims requiring additional review"
            )


    with k4:

        with st.container(
            border=True
        ):

            st.metric(
                "Total Variance",
                f"${total_variance:,.0f}",
            )

            st.caption(
                "Billed amount minus allowed amount"
            )


    st.write("")


    risk_col, attention_col = (
        st.columns(
            [
                1.6,
                1,
            ]
        )
    )


    with risk_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Risk Distribution"
            )

            st.caption(
                "Claims segmented by the current "
                "audit risk classification."
            )


            distribution = (
                df[
                    "final_risk"
                ]
                .value_counts()
                .reindex(
                    [
                        "High",
                        "Medium",
                        "Low",
                    ],
                    fill_value=0,
                )
                .reset_index()
            )


            distribution.columns = [
                "Risk",
                "Claims",
            ]


            chart = px.pie(
                distribution,
                names="Risk",
                values="Claims",
                hole=0.62,
                color="Risk",
                color_discrete_map={
                    "High":
                        "#E45756",

                    "Medium":
                        "#F2C14E",

                    "Low":
                        "#54A24B",
                },
            )


            chart.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10,
                ),
                legend_title_text="Risk",
            )


            chart.update_traces(
                textposition="inside",
                textinfo="label+value",
            )


            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    with attention_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Audit Attention"
            )

            st.caption(
                "Current signals that may deserve "
                "closer auditor review."
            )


            st.metric(
                "High-Risk Claims",
                high_risk_claims,
            )


            st.metric(
                "Multiple Risk Signals",
                multi_signal_claims,
            )


            st.metric(
                "Priority Variance",
                f"${priority_variance:,.0f}",
            )


            st.write(
                "**Provider with most elevated-risk claims**"
            )

            st.write(
                top_provider
            )

            st.caption(
                f"{top_provider_count} elevated-risk claim(s)"
            )


    st.subheader(
        "Financial Overview"
    )


    f1, f2, f3 = (
        st.columns(3)
    )


    f1.metric(
        "Total Billed",
        f"${total_billed:,.0f}",
    )


    f2.metric(
        "Total Allowed",
        f"${total_allowed:,.0f}",
    )


    f3.metric(
        "Total Variance",
        f"${total_variance:,.0f}",
    )


    st.divider()


    st.subheader(
        "Risk Analytics"
    )

    st.caption(
        "Compare provider activity and claim-level "
        "financial patterns."
    )


    analytic1, analytic2 = (
        st.columns(2)
    )


    with analytic1:

        with st.container(
            border=True
        ):

            st.write(
                "**Elevated-Risk Claims by Provider**"
            )


            provider_summary = (
                df.groupby(
                    "provider"
                )
                .agg(
                    total_claims=(
                        "claim_id",
                        "count",
                    ),
                    elevated_risk=(
                        "final_risk",
                        lambda x:
                            x.isin(
                                [
                                    "High",
                                    "Medium",
                                ]
                            ).sum(),
                    ),
                )
                .reset_index()
            )


            chart = px.bar(
                provider_summary,
                x="provider",
                y="elevated_risk",
                hover_data=[
                    "total_claims"
                ],
                labels={
                    "provider":
                        "Provider",

                    "elevated_risk":
                        "Elevated-Risk Claims",
                },
            )


            chart.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=20,
                ),
            )


            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    with analytic2:

        with st.container(
            border=True
        ):

            st.write(
                "**Billed vs Allowed Amount**"
            )


            chart = px.scatter(
                df,
                x="allowed_amount",
                y="billed_amount",
                hover_name="claim_id",
                hover_data=[
                    "provider",
                    "variance",
                    "risk_score",
                ],
                size="units",
                color="final_risk",
                color_discrete_map={
                    "High":
                        "#E45756",

                    "Medium":
                        "#F2C14E",

                    "Low":
                        "#4C78A8",
                },
                labels={
                    "allowed_amount":
                        "Allowed Amount",

                    "billed_amount":
                        "Billed Amount",

                    "final_risk":
                        "Risk",
                },
            )


            chart.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=20,
                ),
                legend_title_text="Risk",
            )


            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    st.subheader(
        "Top Priority Claims"
    )

    st.caption(
        "Claims ranked using risk score "
        "and financial variance."
    )


    top_priority = (
        priority_df_global
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
            column_config={

                "claim_id":
                    "Claim ID",

                "provider":
                    "Provider",

                "procedure_code":
                    "Procedure",

                "billed_amount":
                    st.column_config.NumberColumn(
                        "Billed",
                        format="$%.0f",
                    ),

                "allowed_amount":
                    st.column_config.NumberColumn(
                        "Allowed",
                        format="$%.0f",
                    ),

                "variance":
                    st.column_config.NumberColumn(
                        "Variance",
                        format="$%.0f",
                    ),

                "risk_score":
                    st.column_config.ProgressColumn(
                        "Risk Score",
                        min_value=0,
                        max_value=4,
                    ),

                "final_risk":
                    "Risk",
            },
        )

    else:

        st.success(
            "No elevated-risk claims were identified."
        )


# ============================================================
# CLAIMS ANALYSIS
# ============================================================

def render_claims_analysis():

    st.header(
        "Claims Analysis Workspace"
    )

    st.caption(
        "Explore patterns, narrow the dataset and "
        "inspect claims before full investigation."
    )


    with st.container(
        border=True
    ):

        st.subheader(
            "Filter Claims"
        )


        filter1, filter2, filter3, filter4 = (
            st.columns(4)
        )


        with filter1:

            selected_risks = st.multiselect(
                "Risk Level",
                [
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


        with filter2:

            providers = sorted(
                df[
                    "provider"
                ]
                .astype(str)
                .unique()
            )


            selected_providers = st.multiselect(
                "Provider",
                providers,
                default=providers,
            )


        with filter3:

            procedures = sorted(
                df[
                    "procedure_code"
                ]
                .astype(str)
                .unique()
            )


            selected_procedures = st.multiselect(
                "Procedure",
                procedures,
                default=procedures,
            )


        with filter4:

            search_claim = st.text_input(
                "Claim ID",
                placeholder="Example: CLM004",
            )


        min_variance_value = float(
            df[
                "variance"
            ].min()
        )


        minimum_variance = st.number_input(
            "Minimum Financial Variance",
            min_value=min_variance_value,
            value=min_variance_value,
            step=50.0,
        )


    filtered_df = df[
        df[
            "final_risk"
        ].isin(
            selected_risks
        )
        &
        df[
            "provider"
        ].astype(str).isin(
            selected_providers
        )
        &
        df[
            "procedure_code"
        ].astype(str).isin(
            selected_procedures
        )
        &
        (
            df["variance"]
            >= minimum_variance
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


    st.subheader(
        "Current View"
    )


    m1, m2, m3, m4 = (
        st.columns(4)
    )


    elevated_count = len(
        filtered_df[
            filtered_df[
                "final_risk"
            ].isin(
                [
                    "High",
                    "Medium",
                ]
            )
        ]
    )


    with m1:

        with st.container(
            border=True
        ):

            st.metric(
                "Claims",
                len(filtered_df),
            )


    with m2:

        with st.container(
            border=True
        ):

            st.metric(
                "Elevated Risk",
                elevated_count,
            )


    with m3:

        with st.container(
            border=True
        ):

            st.metric(
                "Billed Amount",
                f"${filtered_df['billed_amount'].sum():,.0f}",
            )


    with m4:

        with st.container(
            border=True
        ):

            st.metric(
                "Financial Variance",
                f"${filtered_df['variance'].sum():,.0f}",
            )


    visual_col, insight_col = (
        st.columns(
            [
                1.55,
                1,
            ]
        )
    )


    with visual_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Claim Financial Pattern"
            )


            if not filtered_df.empty:

                claims_chart = px.scatter(
                    filtered_df,
                    x="allowed_amount",
                    y="billed_amount",
                    size="units",
                    color="final_risk",
                    hover_name="claim_id",
                    hover_data=[
                        "provider",
                        "procedure_code",
                        "variance",
                        "risk_score",
                    ],
                    color_discrete_map={
                        "High":
                            "#E45756",

                        "Medium":
                            "#F2C14E",

                        "Low":
                            "#4C78A8",
                    },
                )


                claims_chart.update_layout(
                    height=350,
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=20,
                    ),
                )


                st.plotly_chart(
                    claims_chart,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )

            else:

                st.info(
                    "No claims match the current filters."
                )


    with insight_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Risk Mix"
            )


            if not filtered_df.empty:

                risk_mix = (
                    filtered_df[
                        "final_risk"
                    ]
                    .value_counts()
                    .reindex(
                        [
                            "High",
                            "Medium",
                            "Low",
                        ],
                        fill_value=0,
                    )
                    .reset_index()
                )


                risk_mix.columns = [
                    "Risk",
                    "Claims",
                ]


                chart = px.pie(
                    risk_mix,
                    names="Risk",
                    values="Claims",
                    hole=0.60,
                    color="Risk",
                    color_discrete_map={
                        "High":
                            "#E45756",

                        "Medium":
                            "#F2C14E",

                        "Low":
                            "#54A24B",
                    },
                )


                chart.update_layout(
                    height=300,
                    margin=dict(
                        l=5,
                        r=5,
                        t=5,
                        b=5,
                    ),
                )


                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )


    st.divider()


    st.subheader(
        "Claim Inspector"
    )


    if not filtered_df.empty:

        filtered_claim_ids = (
            filtered_df[
                "claim_id"
            ].tolist()
        )


        selected_claim = st.selectbox(
            "Inspect Claim",
            filtered_claim_ids,
            key="claims_analysis_inspector",
        )


        selected_row = filtered_df[
            filtered_df[
                "claim_id"
            ]
            == selected_claim
        ].iloc[0]


        claim_col, risk_col = (
            st.columns(
                [
                    1,
                    1.25,
                ]
            )
        )


        with claim_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    selected_row[
                        "claim_id"
                    ]
                )

                st.write(
                    f"**{selected_row['provider']}**"
                )

                st.metric(
                    "Billed",
                    f"${selected_row['billed_amount']:,.0f}",
                )

                st.metric(
                    "Allowed",
                    f"${selected_row['allowed_amount']:,.0f}",
                )

                st.metric(
                    "Variance",
                    f"${selected_row['variance']:,.0f}",
                )

                show_risk_status(
                    selected_row[
                        "final_risk"
                    ]
                )


        with risk_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Audit Snapshot"
                )


                st.write(
                    "**Risk Evidence**"
                )

                st.write(
                    selected_row[
                        "risk_reason"
                    ]
                )


                st.write(
                    "**Recommended Review**"
                )

                st.write(
                    selected_row[
                        "audit_recommendation"
                    ]
                )


                st.button(
                    "Open Full Claim Investigation",
                    type="primary",
                    use_container_width=True,
                    key="analysis_open_claim",
                    on_click=open_claim_review,
                    args=(
                        selected_claim,
                    ),
                )


    st.divider()


    st.subheader(
        "Claims Explorer"
    )


    st.dataframe(
        filtered_df[
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


    st.download_button(
        "Download Current View",
        data=filtered_df.to_csv(
            index=False
        ),
        file_name="filtered_audit_claims.csv",
        mime="text/csv",
    )


# ============================================================
# PRIORITY AUDIT QUEUE
# ============================================================

def render_priority_queue():

    st.header(
        "Priority Audit Queue"
    )

    st.caption(
        "Triage elevated-risk claims using "
        "risk severity and financial exposure."
    )


    priority_df = (
        priority_df_global
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
        .copy()
    )


    if priority_df.empty:

        st.success(
            "No claims currently require priority review."
        )

        return


    s1, s2, s3, s4 = (
        st.columns(4)
    )


    with s1:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Claims",
                len(priority_df),
            )


    with s2:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Variance",
                f"${priority_df['variance'].sum():,.0f}",
            )


    with s3:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Providers",
                priority_df[
                    "provider"
                ].nunique(),
            )


    with s4:

        with st.container(
            border=True
        ):

            st.metric(
                "Average Risk Score",
                f"{priority_df['risk_score'].mean():.1f}/4",
            )


    top_claim = (
        priority_df.iloc[0]
    )


    hero_col, triage_col = (
        st.columns(
            [
                1.15,
                1,
            ]
        )
    )


    with hero_col:

        with st.container(
            border=True
        ):

            st.caption(
                "TOP PRIORITY CLAIM"
            )


            st.subheader(
                top_claim[
                    "claim_id"
                ]
            )

            st.write(
                f"**{top_claim['provider']}**"
            )

            show_risk_status(
                top_claim[
                    "final_risk"
                ]
            )


            c1, c2, c3 = (
                st.columns(3)
            )


            c1.metric(
                "Billed",
                f"${top_claim['billed_amount']:,.0f}",
            )


            c2.metric(
                "Allowed",
                f"${top_claim['allowed_amount']:,.0f}",
            )


            c3.metric(
                "Variance",
                f"${top_claim['variance']:,.0f}",
            )


            st.progress(
                min(
                    float(
                        top_claim[
                            "risk_score"
                        ]
                    )
                    / 4,
                    1.0,
                ),
                text=(
                    f"Risk Score "
                    f"{int(top_claim['risk_score'])}/4"
                ),
            )


            st.write(
                "**Detected Signals**"
            )


            for signal in (
                get_detected_signals(
                    top_claim
                )
            ):

                st.write(
                    f"• {signal}"
                )


            st.info(
                top_claim[
                    "audit_recommendation"
                ]
            )


            st.button(
                "Open Claim Investigation",
                type="primary",
                use_container_width=True,
                key="open_top_priority",
                on_click=open_claim_review,
                args=(
                    top_claim[
                        "claim_id"
                    ],
                ),
            )


    with triage_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Financial Exposure by Claim"
            )


            exposure_chart = px.bar(
                priority_df.sort_values(
                    by="variance",
                    ascending=True,
                ),
                x="variance",
                y="claim_id",
                orientation="h",
                color="final_risk",
                text="variance",
                color_discrete_map={
                    "High":
                        "#E45756",

                    "Medium":
                        "#F2C14E",
                },
            )


            exposure_chart.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside",
            )


            exposure_chart.update_layout(
                height=310,
                margin=dict(
                    l=10,
                    r=45,
                    t=10,
                    b=30,
                ),
            )


            st.plotly_chart(
                exposure_chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    remaining_claims = (
        priority_df.iloc[
            1:4
        ]
    )


    if not remaining_claims.empty:

        st.subheader(
            "Next in Queue"
        )


        queue_columns = st.columns(
            len(
                remaining_claims
            )
        )


        for position, (
            (_, row),
            column,
        ) in enumerate(
            zip(
                remaining_claims.iterrows(),
                queue_columns,
            ),
            start=2,
        ):

            with column:

                with st.container(
                    border=True
                ):

                    st.caption(
                        f"PRIORITY #{position}"
                    )

                    st.subheader(
                        row[
                            "claim_id"
                        ]
                    )

                    st.write(
                        f"**{row['provider']}**"
                    )

                    show_risk_status(
                        row[
                            "final_risk"
                        ]
                    )

                    st.metric(
                        "Variance",
                        f"${row['variance']:,.0f}",
                    )

                    st.button(
                        "Review Claim",
                        use_container_width=True,
                        key=(
                            f"review_"
                            f"{row['claim_id']}"
                        ),
                        on_click=open_claim_review,
                        args=(
                            row[
                                "claim_id"
                            ],
                        ),
                    )


    st.divider()


    with st.expander(
        "View Full Priority Queue",
        expanded=False,
    ):

        st.dataframe(
            priority_df,
            use_container_width=True,
            hide_index=True,
        )


        st.download_button(
            "Download Priority Queue",
            data=priority_df.to_csv(
                index=False
            ),
            file_name="priority_audit_queue.csv",
            mime="text/csv",
        )


# ============================================================
# CLAIM REVIEW
# ============================================================

def render_claim_review():

    st.header(
        "Claim Investigation"
    )

    st.caption(
        "Review claim evidence, inspect detected signals "
        "and use AI-assisted policy context."
    )


    claim_ids = (
        df[
            "claim_id"
        ].tolist()
    )


    if (
        "claim_review_selector"
        not in st.session_state
        or
        st.session_state[
            "claim_review_selector"
        ]
        not in claim_ids
    ):

        st.session_state[
            "claim_review_selector"
        ] = claim_ids[0]


    selector_col, status_col = (
        st.columns(
            [
                3,
                1,
            ]
        )
    )


    with selector_col:

        selected_claim = st.selectbox(
            "Select Claim",
            claim_ids,
            key="claim_review_selector",
        )


    selected_row = df[
        df[
            "claim_id"
        ]
        == selected_claim
    ].iloc[0]


    with status_col:

        st.write(
            "**Current Risk**"
        )

        show_risk_status(
            selected_row[
                "final_risk"
            ]
        )


    st.subheader(
        f"{selected_row['claim_id']} · "
        f"{selected_row['provider']}"
    )


    st.caption(
        f"Procedure "
        f"{selected_row['procedure_code']} "
        f"• Original status: "
        f"{selected_row['status']}"
    )


    m1, m2, m3, m4, m5 = (
        st.columns(5)
    )


    m1.metric(
        "Billed",
        f"${selected_row['billed_amount']:,.0f}",
    )

    m2.metric(
        "Allowed",
        f"${selected_row['allowed_amount']:,.0f}",
    )

    m3.metric(
        "Variance",
        f"${selected_row['variance']:,.0f}",
    )

    m4.metric(
        "Risk Score",
        f"{int(selected_row['risk_score'])}/4",
    )

    m5.metric(
        "Anomaly Score",
        int(
            selected_row[
                "anomaly_score"
            ]
        ),
    )


    st.divider()


    overview_tab, evidence_tab, ai_tab = (
        st.tabs(
            [
                "Overview",
                "Risk Evidence",
                "AI Review",
            ]
        )
    )


    # ========================================================
    # OVERVIEW TAB
    # ========================================================

    with overview_tab:

        financial_col, detail_col = (
            st.columns(2)
        )


        with financial_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Financial Snapshot"
                )


                st.metric(
                    "Billed Amount",
                    f"${selected_row['billed_amount']:,.2f}",
                )


                st.metric(
                    "Allowed Amount",
                    f"${selected_row['allowed_amount']:,.2f}",
                )


                st.metric(
                    "Financial Variance",
                    f"${selected_row['variance']:,.2f}",
                )


                st.write(
                    "**Billed / Allowed Ratio**"
                )

                st.write(
                    f"{selected_row['billed_allowed_ratio']:.2f}x"
                )


        with detail_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Claim Information"
                )


                st.write(
                    "**Claim ID**"
                )

                st.write(
                    selected_row[
                        "claim_id"
                    ]
                )


                st.write(
                    "**Provider**"
                )

                st.write(
                    selected_row[
                        "provider"
                    ]
                )


                st.write(
                    "**Procedure Code**"
                )

                st.write(
                    selected_row[
                        "procedure_code"
                    ]
                )


                st.write(
                    "**Units**"
                )

                st.write(
                    selected_row[
                        "units"
                    ]
                )


                st.write(
                    "**Original Status**"
                )

                st.write(
                    selected_row[
                        "status"
                    ]
                )


        st.subheader(
            "Recommended Audit Action"
        )


        st.info(
            selected_row[
                "audit_recommendation"
            ]
        )


    # ========================================================
    # RISK EVIDENCE TAB
    # ========================================================

    with evidence_tab:

        st.subheader(
            "Risk Assessment"
        )


        st.progress(
            min(
                float(
                    selected_row[
                        "risk_score"
                    ]
                )
                / 4,
                1.0,
            ),
            text=(
                f"Risk Score: "
                f"{int(selected_row['risk_score'])}/4"
            ),
        )


        r1, r2 = (
            st.columns(2)
        )


        with r1:

            with st.container(
                border=True
            ):

                st.write(
                    "**Rule-Based Detection**"
                )


                if (
                    selected_row[
                        "rule_risk"
                    ]
                    == "High Risk"
                ):

                    st.error(
                        "RULE SIGNAL DETECTED"
                    )

                else:

                    st.success(
                        "NO RULE SIGNAL"
                    )


                st.caption(
                    selected_row[
                        "rule_reason"
                    ]
                )


        with r2:

            with st.container(
                border=True
            ):

                st.write(
                    "**Statistical Detection**"
                )


                if (
                    selected_row[
                        "anomaly_status"
                    ]
                    == "Anomaly"
                ):

                    st.warning(
                        "ANOMALY DETECTED"
                    )

                else:

                    st.success(
                        "WITHIN EXPECTED PATTERN"
                    )


                st.caption(
                    f"Anomaly score: "
                    f"{selected_row['anomaly_score']}"
                )


        st.subheader(
            "Detected Signals"
        )


        signals = (
            get_detected_signals(
                selected_row
            )
        )


        if signals:

            signal_columns = (
                st.columns(2)
            )


            for index, signal in enumerate(
                signals
            ):

                with signal_columns[
                    index % 2
                ]:

                    st.warning(
                        signal
                    )

        else:

            st.success(
                "No significant audit signals detected."
            )


        st.subheader(
            "Risk Evidence"
        )


        st.info(
            selected_row[
                "risk_reason"
            ]
        )


        st.subheader(
            "Auditor Review Checklist"
        )


        for index, action in enumerate(
            get_review_actions(
                selected_row
            )
        ):

            st.checkbox(
                action,
                key=(
                    f"check_"
                    f"{selected_claim}_"
                    f"{index}"
                ),
            )


    # ========================================================
    # AI REVIEW TAB
    # ========================================================

    with ai_tab:

        ai_summary_tab, rag_tab = (
            st.tabs(
                [
                    "AI Summary",
                    "Policy-Grounded Review",
                ]
            )
        )


        # ----------------------------------------------------
        # STANDARD AI SUMMARY
        # ----------------------------------------------------

        with ai_summary_tab:

            st.subheader(
                "Gemini-Assisted Audit Summary"
            )


            st.caption(
                "Gemini explains findings already "
                "produced by the audit engine."
            )


            with st.container(
                border=True
            ):

                st.write(
                    "**AI context includes**"
                )

                st.write(
                    "• Claim financial information"
                )

                st.write(
                    "• Risk score and classification"
                )

                st.write(
                    "• Rule and anomaly findings"
                )

                st.write(
                    "• Audit recommendation"
                )


                st.info(
                    "Gemini does not approve, deny "
                    "or adjudicate the claim."
                )


                generate_button = st.button(
                    "Generate AI Audit Summary",
                    type="primary",
                    use_container_width=True,
                    key="generate_standard_ai_summary",
                )


            if generate_button:

                claim_data = (
                    build_claim_dictionary(
                        selected_row
                    )
                )


                with st.spinner(
                    "Gemini is preparing the audit summary..."
                ):

                    result = (
                        generate_audit_summary(
                            claim_data
                        )
                    )


                st.session_state[
                    "ai_summaries"
                ][
                    selected_claim
                ] = result


            if (
                selected_claim
                in st.session_state[
                    "ai_summaries"
                ]
            ):

                st.success(
                    "AI-assisted summary generated."
                )


                with st.container(
                    border=True
                ):

                    st.markdown(
                        st.session_state[
                            "ai_summaries"
                        ][
                            selected_claim
                        ]
                    )


        # ----------------------------------------------------
        # RAG POLICY-GROUNDED REVIEW
        # ----------------------------------------------------

        with rag_tab:

            st.subheader(
                "Policy-Grounded Audit Review"
            )


            st.caption(
                "Retrieve relevant audit policy passages "
                "and use them as context for the AI review."
            )


            if policy_count == 0:

                st.warning(
                    "No audit policy PDFs are available "
                    "in knowledge_base/policies."
                )

            else:

                with st.container(
                    border=True
                ):

                    st.write(
                        "**RAG workflow**"
                    )

                    st.write(
                        "Claim findings"
                    )

                    st.write(
                        "↓"
                    )

                    st.write(
                        f"Search across {policy_count} policy PDF(s)"
                    )

                    st.write(
                        "↓"
                    )

                    st.write(
                        "Retrieve relevant policy passages"
                    )

                    st.write(
                        "↓"
                    )

                    st.write(
                        "Gemini generates a grounded audit explanation"
                    )


                    st.info(
                        "The retrieved policy context supports "
                        "human review. It does not determine "
                        "whether the claim is valid or invalid."
                    )


                    rag_button = st.button(
                        "Generate Policy-Grounded Review",
                        type="primary",
                        use_container_width=True,
                        key="generate_rag_review",
                    )


                if rag_button:

                    claim_data = (
                        build_claim_dictionary(
                            selected_row
                        )
                    )


                    with st.spinner(
                        "Retrieving policy context and "
                        "generating grounded review..."
                    ):

                        try:

                            rag_result = (
                                generate_policy_grounded_audit_summary(
                                    claim_data
                                )
                            )


                            st.session_state[
                                "rag_summaries"
                            ][
                                selected_claim
                            ] = rag_result


                        except Exception as error:

                            st.error(
                                "Policy-grounded review failed: "
                                f"{error}"
                            )


                if (
                    selected_claim
                    in st.session_state[
                        "rag_summaries"
                    ]
                ):

                    rag_result = (
                        st.session_state[
                            "rag_summaries"
                        ][
                            selected_claim
                        ]
                    )


                    st.success(
                        "Policy-grounded review generated."
                    )


                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            rag_result.get(
                                "answer",
                                "No answer was generated.",
                            )
                        )


                    st.subheader(
                        "Retrieved Policy Sources"
                    )


                    sources = (
                        rag_result.get(
                            "sources",
                            [],
                        )
                    )


                    if sources:

                        source_columns = st.columns(
                            min(
                                len(sources),
                                3,
                            )
                        )


                        for index, source in enumerate(
                            sources
                        ):

                            with source_columns[
                                index
                                % len(
                                    source_columns
                                )
                            ]:

                                with st.container(
                                    border=True
                                ):

                                    st.caption(
                                        f"SOURCE {index + 1}"
                                    )

                                    st.write(
                                        f"**{source['source']}**"
                                    )

                                    st.write(
                                        f"Page {source['page']}"
                                    )

                                    st.caption(
                                        f"Retrieval score: "
                                        f"{source['score']:.2f}"
                                    )

                    else:

                        st.info(
                            "No policy passages were retrieved."
                        )


                    with st.expander(
                        "View Retrieval Query",
                        expanded=False,
                    ):

                        st.code(
                            rag_result.get(
                                "retrieval_query",
                                "",
                            )
                        )


                    st.caption(
                        "Policy references come from the local "
                        "demo knowledge base. Human review is required."
                    )


# ============================================================
# AI AUDIT COPILOT
# ============================================================

def render_ai_copilot():

    st.header(
        "Audit Copilot Workspace"
    )

    st.caption(
        "Ask natural-language questions about claims, "
        "providers, financial variance and audit risk."
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Claims in Context",
        total_claims,
    )

    c2.metric(
        "High Risk",
        high_risk_claims,
    )

    c3.metric(
        "Priority Variance",
        f"${priority_variance:,.0f}",
    )

    c4.metric(
        "Policy PDFs",
        policy_count,
    )


    st.divider()


    st.subheader(
        "Quick Investigations"
    )


    q1, q2, q3, q4 = (
        st.columns(4)
    )


    q1.button(
        "High-Risk Claims",
        use_container_width=True,
        on_click=set_copilot_question,
        args=(
            "Which claims are high risk and why were they flagged?",
        ),
    )


    q2.button(
        "Largest Variances",
        use_container_width=True,
        on_click=set_copilot_question,
        args=(
            "Which claims have the largest financial variance?",
        ),
    )


    q3.button(
        "Provider Risk",
        use_container_width=True,
        on_click=set_copilot_question,
        args=(
            "Which provider has the most elevated-risk claims?",
        ),
    )


    q4.button(
        "Review Priority",
        use_container_width=True,
        on_click=set_copilot_question,
        args=(
            "Which claims should receive audit attention first?",
        ),
    )


    question_col, context_col = (
        st.columns(
            [
                1.6,
                1,
            ]
        )
    )


    with question_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Ask Audit Copilot"
            )


            question = st.text_area(
                "Audit Question",
                key="copilot_question",
                height=120,
                placeholder=(
                    "Example: Why was CLM004 flagged?"
                ),
            )


            b1, b2 = (
                st.columns(2)
            )


            with b1:

                ask_button = st.button(
                    "Analyze with Copilot",
                    type="primary",
                    use_container_width=True,
                )


            with b2:

                st.button(
                    "Clear Conversation",
                    use_container_width=True,
                    on_click=clear_copilot,
                )


    with context_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Dataset Context"
            )


            st.write(
                "**Highest-variance claim**"
            )

            st.write(
                highest_variance_claim[
                    "claim_id"
                ]
            )


            st.divider()


            st.write(
                "**Priority provider**"
            )

            st.write(
                top_provider
            )


            st.divider()


            st.write(
                "**Policy knowledge base**"
            )

            st.write(
                f"{policy_count} PDF(s)"
            )


    if ask_button:

        if not question.strip():

            st.warning(
                "Enter an audit question."
            )

        else:

            with st.spinner(
                "Audit Copilot is analyzing..."
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


    st.divider()


    conversation_tab, guardrail_tab = (
        st.tabs(
            [
                "Conversation",
                "Copilot Guardrails",
            ]
        )
    )


    with conversation_tab:

        if st.session_state[
            "copilot_history"
        ]:

            for conversation in reversed(
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
                "No investigation started yet."
            )


    with guardrail_tab:

        with st.container(
            border=True
        ):

            st.write(
                "**Copilot supports**"
            )

            st.write(
                "• Finding elevated-risk claims"
            )

            st.write(
                "• Comparing providers"
            )

            st.write(
                "• Explaining risk signals"
            )

            st.write(
                "• Prioritizing records for review"
            )

            st.write(
                "• Summarizing financial variance"
            )


            st.write(
                "**Copilot does not**"
            )

            st.write(
                "• Approve or deny claims"
            )

            st.write(
                "• Confirm fraud"
            )

            st.write(
                "• Make clinical conclusions"
            )

            st.write(
                "• Replace auditor judgment"
            )


# ============================================================
# EXECUTIVE REPORT
# ============================================================

def render_executive_report():

    st.header(
        "Executive Audit Intelligence"
    )

    st.caption(
        "Management-level view of audit risk, "
        "financial variance and priority findings."
    )


    elevated_risk = (
        high_risk_claims
        + medium_risk_claims
    )


    elevated_rate = (
        (
            elevated_risk
            / total_claims
        )
        * 100
        if total_claims
        else 0
    )


    k1, k2, k3, k4 = (
        st.columns(4)
    )


    k1.metric(
        "Claims Analyzed",
        total_claims,
    )


    k2.metric(
        "Elevated Risk",
        elevated_risk,
    )


    k3.metric(
        "Financial Variance",
        f"${total_variance:,.0f}",
    )


    k4.metric(
        "Policy PDFs",
        policy_count,
    )


    risk_col, exposure_col = (
        st.columns(2)
    )


    with risk_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Portfolio Risk Profile"
            )


            risk_summary = (
                df[
                    "final_risk"
                ]
                .value_counts()
                .reindex(
                    [
                        "High",
                        "Medium",
                        "Low",
                    ],
                    fill_value=0,
                )
                .reset_index()
            )


            risk_summary.columns = [
                "Risk",
                "Claims",
            ]


            chart = px.pie(
                risk_summary,
                names="Risk",
                values="Claims",
                hole=0.62,
                color="Risk",
                color_discrete_map={
                    "High":
                        "#E45756",

                    "Medium":
                        "#F2C14E",

                    "Low":
                        "#54A24B",
                },
            )


            chart.update_layout(
                height=320,
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=10,
                ),
            )


            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    with exposure_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Financial Variance by Provider"
            )


            provider_exposure = (
                df.groupby(
                    "provider"
                )
                .agg(
                    variance=(
                        "variance",
                        "sum",
                    ),
                )
                .reset_index()
                .sort_values(
                    by="variance",
                    ascending=True,
                )
            )


            chart = px.bar(
                provider_exposure,
                x="variance",
                y="provider",
                orientation="h",
                text="variance",
            )


            chart.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside",
            )


            chart.update_layout(
                height=320,
                margin=dict(
                    l=10,
                    r=60,
                    t=10,
                    b=30,
                ),
            )


            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    st.subheader(
        "Priority Findings"
    )


    priority_findings = (
        priority_df_global
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


    st.dataframe(
        priority_findings,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    st.subheader(
        "Generate Executive Audit Report"
    )


    st.info(
        "The executive report summarizes the "
        "current structured audit findings. "
        "Human review is required."
    )


    generate_report = st.button(
        "Generate Executive Audit Report",
        type="primary",
    )


    if generate_report:

        report_prompt = f"""
Create a concise professional executive audit report.

Claims analyzed: {total_claims}

High-risk claims: {high_risk_claims}

Medium-risk claims: {medium_risk_claims}

Low-risk claims: {low_risk_claims}

Elevated-risk percentage:
{elevated_rate:.1f}%

Total billed:
${total_billed:,.2f}

Total allowed:
${total_allowed:,.2f}

Total variance:
${total_variance:,.2f}

Priority variance:
${priority_variance:,.2f}


Use these sections:

Executive Summary

Audit Risk Overview

Highest Priority Findings

Provider-Level Observations

Financial Variance Analysis

Recommended Audit Focus


Do not confirm fraud.

Do not confirm overbilling.

Do not approve or deny claims.

Use only the supplied dataset.
"""


        with st.spinner(
            "Gemini is preparing the report..."
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

        st.subheader(
            "Executive Audit Report"
        )


        with st.container(
            border=True
        ):

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
            file_name="ai_audit_executive_report.txt",
            mime="text/plain",
        )


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Dashboard":

    render_dashboard()


elif page == "Claims Analysis":

    render_claims_analysis()


elif page == "Priority Audit Queue":

    render_priority_queue()


elif page == "Claim Review":

    render_claim_review()


elif page == "AI Audit Copilot":

    render_ai_copilot()


elif page == "Executive Report":

    render_executive_report()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Audit Copilot • Prototype • "
    "Rules and statistical anomaly indicators are decision-support signals only. "
    "RAG retrieves supporting policy context. "
    "AI-generated content requires human review. "
    "Use synthetic or properly de-identified data."
)