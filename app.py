import streamlit as st
import pandas as pd
import joblib
st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    font-size:20px;
    border-radius:12px;
    height:55px;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #ddd;
}

</style>
""",unsafe_allow_html=True)

# ===========================
# Load Model
# ===========================

model = joblib.load("model/creditguard_model.pkl")

# ===========================
# Page Config
# ===========================

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="💳",
    layout="wide"
)

st.title("💳 CreditGuard AI")
st.markdown("### AI Powered Credit Risk Assessment")

st.divider()

st.header("Applicant Information")

col1, col2 = st.columns(2)

# ===========================
# LEFT COLUMN
# ===========================

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=75,
        value=30
    )

    duration = st.number_input(
        "Loan Duration (Months)",
        min_value=4,
        max_value=72,
        value=24
    )

    amount = st.number_input(
        "Loan Amount",
        min_value=250,
        max_value=20000,
        value=5000
    )

    installment_rate = st.selectbox(
        "Installment Rate",
        [1, 2, 3, 4]
    )

    present_residence = st.selectbox(
        "Years at Current Residence",
        [1, 2, 3, 4]
    )

    number_credits = st.selectbox(
        "Existing Credits",
        [1, 2, 3, 4]
    )

    people_liable = st.selectbox(
        "People Liable",
        [1, 2]
    )

    status = st.selectbox(
        "Account Status",
        [
            "... < 100 DM",
            "0 <= ... < 200 DM",
            "no checking account",
            "... >= 200 DM / salary for at least 1 year"
        ]
    )

    savings = st.selectbox(
        "Savings",
        [
            "unknown/no savings account",
            "... < 100 DM",
            "100 <= ... < 500 DM",
            "500 <= ... < 1000 DM",
            "... >= 1000 DM"
        ]
    )

    employment_duration = st.selectbox(
        "Employment Duration",
        [
            "... < 1 year",
            "1 <= ... < 4 years",
            "4 <= ... < 7 years",
            "... >= 7 years",
            "unemployed"
        ]
    )

# ===========================
# RIGHT COLUMN
# ===========================

with col2:

    credit_history = st.selectbox(
        "Credit History",
        [
            "critical account/other credits existing",
            "existing credits paid back duly till now",
            "delay in paying off in the past",
            "all credits at this bank paid back duly",
            "no credits taken/all credits paid back duly"
        ]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "car (new)",
            "car (used)",
            "furniture/equipment",
            "radio/television",
            "domestic appliances",
            "repairs",
            "education",
            "retraining",
            "business",
            "others"
        ]
    )

    personal_status_sex = st.selectbox(
        "Personal Status",
        [
            "male : single",
            "female : divorced/separated/married",
            "male : divorced/separated",
            "male : married/widowed"
        ]
    )

    other_debtors = st.selectbox(
        "Other Debtors",
        [
            "none",
            "co-applicant",
            "guarantor"
        ]
    )

    property = st.selectbox(
        "Property",
        [
            "real estate",
            "building society savings agreement/life insurance",
            "car or other",
            "unknown/no property"
        ]
    )

    other_installment_plans = st.selectbox(
        "Other Installment Plans",
        [
            "none",
            "bank",
            "stores"
        ]
    )

    housing = st.selectbox(
        "Housing",
        [
            "own",
            "rent",
            "for free"
        ]
    )

    job = st.selectbox(
        "Job",
        [
            "unemployed/unskilled - non-resident",
            "unskilled - resident",
            "skilled employee/official",
            "management/self-employed/highly qualified employee/officer"
        ]
    )

    telephone = st.selectbox(
        "Telephone",
        [
            "no",
            "yes"
        ]
    )

    foreign_worker = st.selectbox(
        "Foreign Worker",
        [
            "yes",
            "no"
        ]
    )
st.divider()

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    input_data = pd.DataFrame({

        "status":[status],
        "duration":[duration],
        "credit_history":[credit_history],
        "purpose":[purpose],
        "amount":[amount],
        "savings":[savings],
        "employment_duration":[employment_duration],
        "installment_rate":[installment_rate],
        "personal_status_sex":[personal_status_sex],
        "other_debtors":[other_debtors],
        "present_residence":[present_residence],
        "property":[property],
        "age":[age],
        "other_installment_plans":[other_installment_plans],
        "housing":[housing],
        "number_credits":[number_credits],
        "job":[job],
        "people_liable":[people_liable],
        "telephone":[telephone],
        "foreign_worker":[foreign_worker]

    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.header("Prediction Result")
    st.info(
    f"""
    Prediction generated using the trained
    LightGBM Credit Risk Model.
    """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        if prediction == 1:
            st.success("🟢 LOW RISK")
        else:
            st.error("🔴 HIGH RISK")

    with c2:

        st.metric(
    label="Approval Probability",
    value=f"{probability*100:.2f}%"
)

        st.progress(float(probability))

    with c3:

        if prediction == 1:
            st.success("✅ APPROVE")
        else:
            st.warning("⚠ REVIEW MANUALLY")

    st.divider()

    if prediction == 1:

        st.success("""
### Recommendation

✔ Applicant appears to have **Low Credit Risk**

**Suggested Action**

- Approve Loan
- Standard Verification
- Normal Interest Rate
""")

    else:

        st.error("""
### Recommendation

⚠ Applicant appears to have **High Credit Risk**

**Suggested Action**

- Manual Review
- Request Additional Documents
- Consider Higher Interest Rate
""")
st.divider()

st.caption("© 2026 CreditGuard AI | Developed by Arshit Goyal")