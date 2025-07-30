import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Disaster Impact Predictor - Group 2", layout="wide")

# ------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ------------------------------------------------
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fef3c7"},
            "icon": {"color": "#92400e", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "--hover-color": "#fde68a"},
            "nav-link-selected": {"background-color": "#fcd34d", "color": "#000"},
        }
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Disaster Impact Predictor - Group 2")
    st.write("This tool predicts the number of people potentially affected by a natural disaster using historical patterns.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
    input_data = {}

    # Ask the user for input for each expected feature
    for col in X_columns:
        if col.lower() in ['total_deaths', 'number_injured', 'number_affected', 'number_homeless']:
            input_data[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, value=100.0)
        elif col.lower() in ['country', 'region', 'disaster_group', 'disaster_type']:
            input_data[col] = st.number_input(f"Encoded: {col.replace('_', ' ').title()}", min_value=0, step=1)
        elif col.lower() == 'year':
            input_data[col] = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
        else:
            input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)

    # Create DataFrame and reorder columns to match training data
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X_columns, fill_value=0)

    try:
        # Impute & Scale
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        if st.button("Predict Affected People"):
            prediction = model.predict(input_scaled)[0]
            st.success(f"📌 Estimated Number of People Affected: {prediction:,.0f}")
    except Exception as e:
        st.error("🚫 Prediction Failed.")
        st.code(str(e))
        st.write("Expected columns:", X_columns)
        st.write("Current input shape:", input_df.shape)

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This app was developed by *Group 2* to estimate the number of people affected by natural disasters.

        *Model Used*: Decision Tree Regressor  
        *Tools*: Python, Scikit-learn, Streamlit  
        *Dataset*: Natural Disaster Records (1993–2023)  
        *Goal*: Provide insights for emergency preparedness and resource planning.
    """)
