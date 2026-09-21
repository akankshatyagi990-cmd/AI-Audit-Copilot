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
    # REVIEW RECOMMENDATIONS
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
        "Upload a properly structured, "
        "synthetic or de-identified audit CSV."
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
    "Prototype Status",
    expanded=False,
):

    st.write(
        "**Risk detection**"
    )

    st.caption(
        "Deterministic prototype rules identify "
        "predefined audit signals."
    )


    st.write(
        "**Anomaly detection**"
    )

    st.caption(
        "IQR-based statistical analysis identifies "
        "values outside expected dataset patterns."
    )


    st.write(
        "**Generative AI**"
    )

    st.caption(
        "Gemini explains existing findings, answers "
        "dataset questions and prepares summaries."
    )


    st.info(
        "Risk indicators support human review. "
        "They do not confirm claim validity, fraud, "
        "coding errors or payment decisions."
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
# GLOBAL PRODUCT HEADER
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


    # --------------------------------------------------------
    # KPI SUMMARY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RISK DISTRIBUTION + ATTENTION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FINANCIAL OVERVIEW
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RISK ANALYTICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRIORITY CLAIMS
    # --------------------------------------------------------

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
            "No elevated-risk claims were identified "
            "in the current dataset."
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
        "inspect claims before opening a full investigation."
    )


    # --------------------------------------------------------
    # FILTER WORKSPACE
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Filter Claims"
        )

        st.caption(
            "Refine the current dataset using "
            "audit and claim attributes."
        )


        filter1, filter2, filter3, filter4 = (
            st.columns(4)
        )


        with filter1:

            selected_risks = st.multiselect(
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
                options=providers,
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
                options=procedures,
                default=procedures,
            )


        with filter4:

            search_claim = st.text_input(
                "Claim ID",
                placeholder=(
                    "Example: CLM004"
                ),
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


    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = df[
        df[
            "final_risk"
        ].isin(
            selected_risks
        )
        &
        df[
            "provider"
        ]
        .astype(str)
        .isin(
            selected_providers
        )
        &
        df[
            "procedure_code"
        ]
        .astype(str)
        .isin(
            selected_procedures
        )
        &
        (
            df[
                "variance"
            ]
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


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Current View"
    )


    m1, m2, m3, m4 = (
        st.columns(4)
    )


    with m1:

        with st.container(
            border=True
        ):

            st.metric(
                "Claims",
                len(filtered_df),
            )

            st.caption(
                "Records matching current filters"
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


    with m2:

        with st.container(
            border=True
        ):

            st.metric(
                "Elevated Risk",
                elevated_count,
            )

            st.caption(
                "High and medium-risk claims"
            )


    with m3:

        with st.container(
            border=True
        ):

            st.metric(
                "Billed Amount",
                f"${filtered_df['billed_amount'].sum():,.0f}",
            )

            st.caption(
                "Total billed amount in current view"
            )


    with m4:

        with st.container(
            border=True
        ):

            st.metric(
                "Financial Variance",
                f"${filtered_df['variance'].sum():,.0f}",
            )

            st.caption(
                "Billed amount minus allowed amount"
            )


    st.write("")


    # --------------------------------------------------------
    # VISUAL ANALYSIS
    # --------------------------------------------------------

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

            st.caption(
                "Compare billed amount, allowed amount "
                "and risk across the filtered claims."
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
                    labels={
                        "allowed_amount":
                            "Allowed Amount",

                        "billed_amount":
                            "Billed Amount",

                        "final_risk":
                            "Risk",
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
                    legend_title_text="Risk",
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
                    "No claims match the current filters. "
                    "Adjust one or more filters to continue."
                )


    with insight_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Risk Mix"
            )

            st.caption(
                "Risk distribution within the "
                "current filtered view."
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


                risk_chart = px.pie(
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


                risk_chart.update_traces(
                    textinfo="label+value",
                    textposition="inside",
                )


                risk_chart.update_layout(
                    height=275,
                    margin=dict(
                        l=5,
                        r=5,
                        t=5,
                        b=5,
                    ),
                    showlegend=False,
                )


                st.plotly_chart(
                    risk_chart,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )


                high_view = len(
                    filtered_df[
                        filtered_df[
                            "final_risk"
                        ]
                        == "High"
                    ]
                )


                if high_view > 0:

                    st.warning(
                        f"{high_view} high-risk claim(s) "
                        "are present in the current view."
                    )

                else:

                    st.success(
                        "No high-risk claims are present "
                        "in the current view."
                    )

            else:

                st.info(
                    "Risk distribution will appear "
                    "when claims match the filters."
                )


    # --------------------------------------------------------
    # CLAIM INSPECTOR
    # --------------------------------------------------------

    st.divider()


    st.subheader(
        "Claim Inspector"
    )

    st.caption(
        "Perform a quick review before opening "
        "the full Claim Investigation workspace."
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

                st.caption(
                    "CLAIM SNAPSHOT"
                )


                st.subheader(
                    selected_row[
                        "claim_id"
                    ]
                )


                st.write(
                    f"**{selected_row['provider']}**"
                )


                st.caption(
                    f"Procedure "
                    f"{selected_row['procedure_code']}"
                )


                financial1, financial2 = (
                    st.columns(2)
                )


                financial1.metric(
                    "Billed",
                    f"${selected_row['billed_amount']:,.0f}",
                )


                financial2.metric(
                    "Allowed",
                    f"${selected_row['allowed_amount']:,.0f}",
                )


                st.metric(
                    "Variance",
                    f"${selected_row['variance']:,.0f}",
                )


                st.write(
                    "**Risk Score**"
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
                        f"{int(selected_row['risk_score'])}/4"
                    ),
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
                    "**Why the claim was classified this way**"
                )


                if (
                    selected_row[
                        "final_risk"
                    ]
                    == "High"
                ):

                    st.error(
                        selected_row[
                            "risk_reason"
                        ]
                    )

                elif (
                    selected_row[
                        "final_risk"
                    ]
                    == "Medium"
                ):

                    st.warning(
                        selected_row[
                            "risk_reason"
                        ]
                    )

                else:

                    st.info(
                        selected_row[
                            "risk_reason"
                        ]
                    )


                st.write(
                    "**Recommended review action**"
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


    else:

        st.warning(
            "No claims are available for inspection "
            "with the current filters."
        )


    # --------------------------------------------------------
    # CLAIMS EXPLORER
    # --------------------------------------------------------

    st.divider()


    st.subheader(
        "Claims Explorer"
    )

    st.caption(
        "Detailed claim records for the "
        "current filtered view."
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
                    format="%d",
                ),

            "final_risk":
                "Risk",
        },
    )


    st.download_button(
        "Download Current View",
        data=filtered_df.to_csv(
            index=False
        ),
        file_name=(
            "filtered_audit_claims.csv"
        ),
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
            "No claims currently require elevated-risk review."
        )

        return


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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

            st.caption(
                "Claims in the elevated-risk queue"
            )


    with s2:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Variance",
                f"${priority_df['variance'].sum():,.0f}",
            )

            st.caption(
                "Financial variance within the queue"
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

            st.caption(
                "Providers represented in the queue"
            )


    with s4:

        with st.container(
            border=True
        ):

            st.metric(
                "Average Risk Score",
                f"{priority_df['risk_score'].mean():.1f}/4",
            )

            st.caption(
                "Average queue risk score"
            )


    st.write("")


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


    # --------------------------------------------------------
    # TOP PRIORITY CLAIM
    # --------------------------------------------------------

    with hero_col:

        with st.container(
            border=True
        ):

            st.caption(
                "TOP PRIORITY CLAIM"
            )


            title_col, risk_col = (
                st.columns(
                    [
                        3,
                        1,
                    ]
                )
            )


            with title_col:

                st.subheader(
                    top_claim[
                        "claim_id"
                    ]
                )

                st.write(
                    f"**{top_claim['provider']}**"
                )

                st.caption(
                    f"Procedure "
                    f"{top_claim['procedure_code']}"
                )


            with risk_col:

                show_risk_status(
                    top_claim[
                        "final_risk"
                    ]
                )


            st.divider()


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


            st.write(
                "**Risk Score**"
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
                    f"{int(top_claim['risk_score'])}/4"
                ),
            )


            st.write(
                "**Detected signals**"
            )


            signals = (
                get_detected_signals(
                    top_claim
                )
            )


            if signals:

                for signal in signals:

                    st.write(
                        f"• {signal}"
                    )

            else:

                st.write(
                    "No additional statistical signals detected."
                )


            st.write(
                "**Recommended review**"
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


    # --------------------------------------------------------
    # TRIAGE VISUAL
    # --------------------------------------------------------

    with triage_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Audit Triage"
            )

            st.caption(
                "How the current dataset narrows "
                "into the elevated-risk queue."
            )


            t1, t2 = (
                st.columns(2)
            )


            t1.metric(
                "All Claims",
                total_claims,
            )


            t2.metric(
                "Elevated Risk",
                len(priority_df),
            )


            queue_ratio = (
                len(priority_df)
                / total_claims
                if total_claims
                else 0
            )


            st.progress(
                queue_ratio,
                text=(
                    f"{len(priority_df)} of "
                    f"{total_claims} claims "
                    "are in the priority queue"
                ),
            )


            st.divider()


            st.write(
                "**Financial Variance by Priority Claim**"
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
                labels={
                    "variance":
                        "Variance",

                    "claim_id":
                        "Claim",

                    "final_risk":
                        "Risk",
                },
            )


            exposure_chart.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside",
            )


            exposure_chart.update_layout(
                height=245,
                margin=dict(
                    l=10,
                    r=45,
                    t=10,
                    b=30,
                ),
                legend_title_text="Risk",
            )


            st.plotly_chart(
                exposure_chart,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    # --------------------------------------------------------
    # NEXT CLAIMS
    # --------------------------------------------------------

    remaining_claims = (
        priority_df.iloc[
            1:4
        ]
    )


    if not remaining_claims.empty:

        st.subheader(
            "Next in Queue"
        )

        st.caption(
            "Claims following the highest-priority record."
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


                    st.progress(
                        min(
                            float(
                                row[
                                    "risk_score"
                                ]
                            )
                            / 4,
                            1.0,
                        ),
                        text=(
                            f"Risk "
                            f"{int(row['risk_score'])}/4"
                        ),
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


    # --------------------------------------------------------
    # FULL QUEUE
    # --------------------------------------------------------

    st.divider()


    with st.expander(
        "View Full Priority Queue",
        expanded=False,
    ):

        display_df = (
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
                ]
            ]
        )


        st.dataframe(
            display_df,
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

                "risk_reason":
                    "Risk Evidence",
            },
        )


        st.download_button(
            "Download Priority Queue",
            data=display_df.to_csv(
                index=False
            ),
            file_name=(
                "priority_audit_queue.csv"
            ),
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
        "Review claim details, inspect detected signals "
        "and use AI-assisted explanations to support audit review."
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
            key=(
                "claim_review_selector"
            ),
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


    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

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
                    delta=(
                        f"{selected_row['variance_percentage']:.0f}% "
                        "relative to allowed"
                    ),
                    delta_color="inverse",
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


        if (
            selected_row[
                "final_risk"
            ]
            == "High"
        ):

            st.warning(
                selected_row[
                    "audit_recommendation"
                ]
            )

        else:

            st.info(
                selected_row[
                    "audit_recommendation"
                ]
            )


    # --------------------------------------------------------
    # RISK EVIDENCE
    # --------------------------------------------------------

    with evidence_tab:

        st.subheader(
            "Risk Assessment"
        )

        st.caption(
            "Risk combines prototype rules "
            "and statistical anomaly indicators."
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
                "No significant audit signals "
                "were detected for this claim."
            )


        st.subheader(
            "Why This Claim Needs Attention"
        )


        if (
            selected_row[
                "final_risk"
            ]
            == "High"
        ):

            st.error(
                selected_row[
                    "risk_reason"
                ]
            )

        elif (
            selected_row[
                "final_risk"
            ]
            == "Medium"
        ):

            st.warning(
                selected_row[
                    "risk_reason"
                ]
            )

        else:

            st.info(
                selected_row[
                    "risk_reason"
                ]
            )


        st.subheader(
            "Auditor Review Checklist"
        )

        st.caption(
            "Use these actions as review prompts. "
            "They are not automated claim decisions."
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


    # --------------------------------------------------------
    # AI REVIEW
    # --------------------------------------------------------

    with ai_tab:

        st.subheader(
            "Gemini-Assisted Audit Review"
        )


        st.caption(
            "Generate an auditor-friendly explanation "
            "from findings already produced by the audit engine."
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
                "• Risk score and risk classification"
            )

            st.write(
                "• Detected rule and anomaly signals"
            )

            st.write(
                "• Current audit recommendation"
            )


            st.info(
                "Gemini explains existing findings. "
                "It does not approve, deny or adjudicate claims."
            )


            generate_button = st.button(
                "Generate AI Audit Summary",
                type="primary",
                use_container_width=True,
            )


        if generate_button:

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
                "AI-assisted review generated."
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


            st.caption(
                "Review AI-generated content before "
                "using it in an audit workflow."
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
        "providers, financial variance and audit risk patterns."
    )


    # --------------------------------------------------------
    # CONTEXT SUMMARY
    # --------------------------------------------------------

    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        with st.container(
            border=True
        ):

            st.metric(
                "Claims in Context",
                total_claims,
            )


    with c2:

        with st.container(
            border=True
        ):

            st.metric(
                "High Risk",
                high_risk_claims,
            )


    with c3:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Variance",
                f"${priority_variance:,.0f}",
            )


    with c4:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Provider",
                top_provider,
            )


    st.divider()


    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.subheader(
        "Quick Investigations"
    )

    st.caption(
        "Use a common audit question "
        "or enter your own below."
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
            "Which provider has the most elevated-risk claims and why?",
        ),
    )


    q4.button(
        "Review Priority",
        use_container_width=True,
        on_click=set_copilot_question,
        args=(
            "Which claims should receive audit attention first based on the current risk indicators?",
        ),
    )


    st.write("")


    # --------------------------------------------------------
    # COPILOT WORKSPACE
    # --------------------------------------------------------

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

            st.caption(
                "Questions are answered using the "
                "currently analyzed dataset."
            )


            question = st.text_area(
                "Audit Question",
                key="copilot_question",
                height=120,
                placeholder=(
                    "Example: Why was CLM004 flagged, "
                    "and what should the auditor review?"
                ),
            )


            button1, button2 = (
                st.columns(2)
            )


            with button1:

                ask_button = st.button(
                    "Analyze with Copilot",
                    type="primary",
                    use_container_width=True,
                )


            with button2:

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

            st.caption(
                "Current signals available "
                "to the Copilot."
            )


            st.write(
                "**Highest-variance claim**"
            )

            st.write(
                highest_variance_claim[
                    "claim_id"
                ]
            )

            st.caption(
                f"${highest_variance_claim['variance']:,.0f} variance"
            )


            st.divider()


            st.write(
                "**Provider with most elevated-risk claims**"
            )

            st.write(
                top_provider
            )

            st.caption(
                f"{top_provider_count} elevated-risk claim(s)"
            )


            st.divider()


            st.write(
                "**Claims with multiple anomaly signals**"
            )

            st.write(
                multi_signal_claims
            )


    # --------------------------------------------------------
    # ASK GEMINI
    # --------------------------------------------------------

    if ask_button:

        if not question.strip():

            st.warning(
                "Enter an audit question before "
                "running the Copilot."
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


    # --------------------------------------------------------
    # CONVERSATION + GUARDRAILS
    # --------------------------------------------------------

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

            st.subheader(
                "Investigation History"
            )


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

            with st.container(
                border=True
            ):

                st.info(
                    "No Copilot investigation "
                    "has been started yet."
                )

                st.write(
                    "Try one of the Quick Investigations "
                    "or ask a question about the dataset."
                )


    with guardrail_tab:

        with st.container(
            border=True
        ):

            st.subheader(
                "Copilot Scope"
            )


            can_col, cannot_col = (
                st.columns(2)
            )


            with can_col:

                st.write(
                    "**Can support**"
                )

                st.write(
                    "• Finding elevated-risk claims"
                )

                st.write(
                    "• Comparing providers"
                )

                st.write(
                    "• Explaining existing risk signals"
                )

                st.write(
                    "• Prioritizing records for review"
                )

                st.write(
                    "• Summarizing financial variance"
                )


            with cannot_col:

                st.write(
                    "**Does not**"
                )

                st.write(
                    "• Approve or deny claims"
                )

                st.write(
                    "• Confirm fraud or improper billing"
                )

                st.write(
                    "• Make clinical determinations"
                )

                st.write(
                    "• Replace auditor judgment"
                )

                st.write(
                    "• Invent missing claim information"
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
        "financial variance, provider patterns "
        "and priority findings."
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


    # --------------------------------------------------------
    # KPI SUMMARY
    # --------------------------------------------------------

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
                "Claims included in current analysis"
            )


    with k2:

        with st.container(
            border=True
        ):

            st.metric(
                "Elevated Risk",
                elevated_risk,
            )

            st.caption(
                f"{elevated_rate:.0f}% of analyzed claims"
            )


    with k3:

        with st.container(
            border=True
        ):

            st.metric(
                "Financial Variance",
                f"${total_variance:,.0f}",
            )

            st.caption(
                "Total billed amount minus allowed amount"
            )


    with k4:

        with st.container(
            border=True
        ):

            st.metric(
                "Priority Provider",
                top_provider,
            )

            st.caption(
                f"{top_provider_count} elevated-risk claim(s)"
            )


    st.write("")


    # --------------------------------------------------------
    # EXECUTIVE SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        "Executive Snapshot"
    )

    st.caption(
        "Risk concentration and financial variance "
        "across the current audit dataset."
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

            st.caption(
                "Distribution of claims "
                "by audit risk level."
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


            risk_chart = px.pie(
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


            risk_chart.update_traces(
                textinfo="label+value",
                textposition="inside",
            )


            risk_chart.update_layout(
                height=320,
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=10,
                ),
                legend_title_text="Risk",
            )


            st.plotly_chart(
                risk_chart,
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

            st.caption(
                "Total claim variance grouped "
                "by provider."
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
                    claims=(
                        "claim_id",
                        "count",
                    ),
                )
                .reset_index()
                .sort_values(
                    by="variance",
                    ascending=True,
                )
            )


            exposure_chart = px.bar(
                provider_exposure,
                x="variance",
                y="provider",
                orientation="h",
                text="variance",
                hover_data=[
                    "claims"
                ],
                labels={
                    "variance":
                        "Financial Variance",

                    "provider":
                        "Provider",
                },
            )


            exposure_chart.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside",
            )


            exposure_chart.update_layout(
                height=320,
                margin=dict(
                    l=10,
                    r=60,
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


    # --------------------------------------------------------
    # MANAGEMENT ATTENTION
    # --------------------------------------------------------

    st.subheader(
        "Management Attention"
    )

    st.caption(
        "Key indicators to review before "
        "generating the executive summary."
    )


    attention1, attention2, attention3 = (
        st.columns(3)
    )


    with attention1:

        with st.container(
            border=True
        ):

            st.caption(
                "HIGHEST FINANCIAL VARIANCE"
            )


            st.subheader(
                highest_variance_claim[
                    "claim_id"
                ]
            )


            st.metric(
                "Variance",
                f"${highest_variance_claim['variance']:,.0f}",
            )


            st.write(
                f"**{highest_variance_claim['provider']}**"
            )


            st.caption(
                f"Procedure "
                f"{highest_variance_claim['procedure_code']}"
            )


    with attention2:

        with st.container(
            border=True
        ):

            st.caption(
                "HIGH-RISK CLAIMS"
            )


            st.subheader(
                high_risk_claims
            )


            st.metric(
                "Priority Variance",
                f"${priority_variance:,.0f}",
            )


            st.caption(
                "Variance associated with "
                "the elevated-risk review queue"
            )


    with attention3:

        with st.container(
            border=True
        ):

            st.caption(
                "PROVIDER WITH MOST ELEVATED RISK"
            )


            st.subheader(
                top_provider
            )


            st.metric(
                "Elevated-Risk Claims",
                top_provider_count,
            )


            st.caption(
                "Based on the current analyzed dataset"
            )


    # --------------------------------------------------------
    # PRIORITY FINDINGS
    # --------------------------------------------------------

    st.subheader(
        "Priority Findings"
    )

    st.caption(
        "Highest-priority claims based on "
        "risk score and financial variance."
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
        .copy()
    )


    if not priority_findings.empty:

        st.dataframe(
            priority_findings[
                [
                    "claim_id",
                    "provider",
                    "procedure_code",
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
                        format="%d",
                    ),

                "final_risk":
                    "Risk",
            },
        )

    else:

        st.success(
            "No elevated-risk priority findings "
            "were identified."
        )


    st.divider()


    # --------------------------------------------------------
    # REPORT GENERATION
    # --------------------------------------------------------

    st.subheader(
        "Generate Executive Audit Report"
    )

    st.caption(
        "Create a concise management summary "
        "using the analyzed dataset and Gemini."
    )


    contents_col, readiness_col = (
        st.columns(
            [
                1.4,
                1,
            ]
        )
    )


    with contents_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Report Contents"
            )


            left_section, right_section = (
                st.columns(2)
            )


            with left_section:

                st.write(
                    "✓ Executive Summary"
                )

                st.write(
                    "✓ Audit Risk Overview"
                )

                st.write(
                    "✓ Highest Priority Findings"
                )


            with right_section:

                st.write(
                    "✓ Provider-Level Observations"
                )

                st.write(
                    "✓ Financial Variance Analysis"
                )

                st.write(
                    "✓ Recommended Audit Focus"
                )


            st.info(
                "The report summarizes existing audit signals. "
                "It does not make final claim determinations."
            )


    with readiness_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "Report Readiness"
            )


            st.write(
                "**Dataset**"
            )

            st.write(
                f"{total_claims} claims analyzed"
            )


            st.write(
                "**Risk analysis**"
            )

            st.write(
                f"{elevated_risk} elevated-risk claims"
            )


            st.write(
                "**Financial analysis**"
            )

            st.write(
                f"${total_variance:,.0f} total variance analyzed"
            )


            st.success(
                "Ready to generate"
            )


            generate_report = st.button(
                "Generate Executive Audit Report",
                type="primary",
                use_container_width=True,
            )


    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    if generate_report:

        report_prompt = f"""
Create a concise professional executive audit report
for the current healthcare claims audit dataset.

DATASET SUMMARY

Claims analyzed: {total_claims}

High-risk claims: {high_risk_claims}

Medium-risk claims: {medium_risk_claims}

Low-risk claims: {low_risk_claims}

Elevated-risk percentage: {elevated_rate:.1f}%

Total billed amount: ${total_billed:,.2f}

Total allowed amount: ${total_allowed:,.2f}

Total financial variance: ${total_variance:,.2f}

Priority-queue variance: ${priority_variance:,.2f}

Provider with most elevated-risk claims:
{top_provider}

Elevated-risk claims for that provider:
{top_provider_count}


Create the report using these sections:

1. Executive Summary

2. Audit Risk Overview

3. Highest Priority Findings

4. Provider-Level Observations

5. Financial Variance Analysis

6. Recommended Audit Focus


Requirements:

Mention specific claim IDs where relevant.

Use only the supplied audit dataset.

Clearly separate observations from recommendations.

Do not claim that fraud has occurred.

Do not claim that overbilling has been confirmed.

Do not state that coding errors or improper claims
have been confirmed.

Describe unusual records only as indicators
requiring auditor review.

Do not approve or deny claims.

Write for an executive or audit-management audience.

Keep the report concise, professional and actionable.
"""


        with st.spinner(
            "Gemini is preparing the executive audit report..."
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


    # --------------------------------------------------------
    # GENERATED REPORT
    # --------------------------------------------------------

    if st.session_state[
        "executive_report"
    ]:

        st.divider()


        report_tab, data_tab = (
            st.tabs(
                [
                    "Executive Report",
                    "Supporting Data",
                ]
            )
        )


        with report_tab:

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


            download_col, notice_col = (
                st.columns(
                    [
                        1,
                        2,
                    ]
                )
            )


            with download_col:

                st.download_button(
                    "Download Executive Report",
                    data=st.session_state[
                        "executive_report"
                    ],
                    file_name=(
                        "ai_audit_executive_report.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True,
                )


            with notice_col:

                st.info(
                    "AI-generated executive content should "
                    "be reviewed by a qualified auditor "
                    "before broader use."
                )


        with data_tab:

            st.subheader(
                "Supporting Audit Data"
            )

            st.caption(
                "Structured findings provided as context "
                "for the executive report."
            )


            supporting_df = (
                df[
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
            )


            st.dataframe(
                supporting_df,
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
    "Rule-based and statistical risk indicators are decision-support signals only. "
    "AI-generated content requires human review. "
    "Use synthetic or properly de-identified data."
)