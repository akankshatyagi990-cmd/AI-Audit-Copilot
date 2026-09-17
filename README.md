# AI Audit Copilot

AI-powered healthcare claims audit analysis, risk intelligence, anomaly detection, and auditor decision-support prototype.

AI Audit Copilot analyzes structured claim data, identifies potentially unusual claims, prioritizes records for review, explains detected risk indicators, and uses Google Gemini to generate auditor-friendly summaries and answer questions about the analyzed dataset.

> This project is a prototype built using synthetic or de-identified data. It is intended for audit decision support and does not make final claim determinations.

---

## Overview

Healthcare auditors often need to review large volumes of claims and determine which records require closer investigation.

AI Audit Copilot helps simplify this process by combining:

- Rule-based audit checks
- Statistical anomaly detection
- Explainable risk scoring
- Audit recommendations
- Provider-level analytics
- Priority claim ranking
- Generative AI summaries
- Natural-language audit Q&A
- Executive audit reporting

The goal is to help auditors focus attention on potentially important claims instead of manually reviewing every record with the same priority.

---

## Key Features

### 1. CSV Claims Upload

Auditors can upload structured claim data directly through the application.

The system validates required fields before performing analysis.

Expected fields include:

- `claim_id`
- `provider`
- `procedure_code`
- `billed_amount`
- `allowed_amount`
- `units`
- `status`

If no file is uploaded, the application uses the included synthetic sample dataset.

---

### 2. Rule-Based Risk Detection

The prototype applies configurable audit rules to identify claims that may require review.

Current demonstration rules include:

- Billed amount greater than 2x the allowed amount
- Units greater than 3

These rules are for prototype demonstration only and should not be treated as production healthcare billing rules.

---

### 3. Statistical Anomaly Detection

The application uses the Interquartile Range (IQR) method to identify unusual values across:

- Billed amount
- Allowed amount
- Number of units
- Billed-to-allowed variance
- Billed-to-allowed ratio

Each claim receives an anomaly score based on the number of unusual features detected.

---

### 4. Explainable Risk Scoring

Rule-based findings and statistical anomalies are combined into a final risk score.

Claims are categorized as:

- **High Risk**
- **Medium Risk**
- **Low Risk**

The system also explains why each claim received its risk classification.

Example:

```text
Risk Level: High

Risk Score: 4

Reason:
Billed amount is more than 2x the allowed amount;
multiple statistical anomalies detected in billed amount,
variance and billed-to-allowed ratio.