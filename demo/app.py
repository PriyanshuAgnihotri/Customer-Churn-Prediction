
import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

API_URL = "https://customer-churn-api-qmev.onrender.com/predict"

st.title("📊 Customer Churn Prediction")
st.caption("AI-powered customer churn risk assessment")
st.info(
    "This demonstration uses a machine learning model trained "
    "on synthetic B2B SaaS customer data."
)

with st.sidebar:
    st.header("API Configuration")
    api_url = st.text_input(
        "Prediction API URL",
        value=API_URL,
    )
    st.caption("The demo sends customer inputs to your FastAPI backend.")

st.subheader("Customer Information")

with st.form("customer_form"):

    st.markdown("### Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure_months = st.number_input(
            "Tenure (months)", min_value=0, value=24
        )
        country = st.selectbox(
            "Country",
            ["United States", "Australia", "Canada", "Germany",
             "India", "United Kingdom"],
        )
        industry = st.selectbox(
            "Industry",
            ["Technology", "Finance", "Healthcare",
             "Manufacturing", "Professional Services", "Retail"],
        )

    with col2:
        company_size = st.selectbox(
            "Company Size", ["Small", "Mid-Market", "Enterprise"]
        )
        subscription_plan = st.selectbox(
            "Subscription Plan", ["Basic", "Business", "Enterprise", "Pro"]
        )
        contract_type = st.selectbox(
            "Contract Type", ["Monthly", "Annual"]
        )

    with col3:
        monthly_revenue = st.number_input(
            "Monthly Revenue ($)", min_value=0.01,
            value=250.0, step=25.0
        )
        auto_renew_label = st.selectbox(
            "Auto Renew", ["Enabled", "Disabled"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Credit Card", "ACH", "Wire Transfer"],
        )

    st.markdown("### Customer Engagement")

    col4, col5, col6 = st.columns(3)

    with col4:
        monthly_logins = st.number_input(
            "Monthly Logins", min_value=0, value=12
        )
        active_days = st.number_input(
            "Active Days (last 30)", min_value=0,
            max_value=30, value=20
        )
        avg_session = st.number_input(
            "Average Session Minutes", min_value=0.0,
            value=15.0
        )

    with col5:
        feature_usage = st.number_input(
            "Feature Usage Score", min_value=0.0,
            max_value=100.0, value=65.0
        )
        days_since_login = st.number_input(
            "Days Since Last Login", min_value=0, value=5
        )
        usage_change = st.number_input(
            "Usage Change (30 days)", value=5.0
        )

    with col6:
        support_tickets = st.number_input(
            "Support Tickets (90 days)", min_value=0, value=1
        )
        resolution_hours = st.number_input(
            "Average Resolution Hours", min_value=0.0,
            value=8.0
        )
        satisfaction = st.number_input(
            "Customer Satisfaction Score", min_value=0.0,
            max_value=100.0, value=80.0
        )

    st.markdown("### Payment Information")

    failed_payments = st.number_input(
        "Failed Payments (last 6 months)",
        min_value=0,
        value=0,
    )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:

    payload = {
        "tenure_months": tenure_months,
        "country": country,
        "industry": industry,
        "company_size": company_size,
        "subscription_plan": subscription_plan,
        "contract_type": contract_type,
        "monthly_revenue": monthly_revenue,
        "auto_renew": 1 if auto_renew_label == "Enabled" else 0,
        "monthly_logins": monthly_logins,
        "active_days_last_30": active_days,
        "avg_session_minutes": avg_session,
        "feature_usage_score": feature_usage,
        "days_since_last_login": days_since_login,
        "usage_change_30d": usage_change,
        "support_tickets_last_90d": support_tickets,
        "avg_resolution_hours": resolution_hours,
        "customer_satisfaction_score": satisfaction,
        "failed_payments_last_6m": failed_payments,
        "payment_method": payment_method,
    }

    try:
        with st.spinner("Analyzing customer churn risk..."):
            response = requests.post(
                api_url,
                json=payload,
                timeout=120,
            )

        if response.status_code == 200:
            result = response.json()

            probability = result["churn_probability"]
            risk = result["risk_level"]

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}",
                )

            with col2:
                st.metric("Risk Classification", risk)

            st.progress(probability)

            if risk == "High":
                st.error("High churn risk detected.")
            elif risk == "Medium":
                st.warning("Moderate churn risk detected.")
            else:
                st.success("Low churn risk detected.")

            st.caption(
                "Prediction is for demonstration purposes and "
                "should not be treated as a guaranteed outcome."
            )

        else:
            st.error(f"API error: HTTP {response.status_code}")
            st.code(response.text)

    except requests.exceptions.Timeout:
        st.error("The API request timed out. Please try again.")

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to connect to the prediction API: {error}")
