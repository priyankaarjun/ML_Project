import streamlit as st
import joblib 
import numpy as np

try:
    regression_model = joblib.load('regression_model.pkl')
    classification_model = joblib.load('classification_model.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'regression_model.pkl' and 'classification_model.pkl' are in the same directory.")
    st.stop()

st.title("Copper Industry Prediction App")

st.markdown("""
This app helps predict:
1. **Selling Price** of copper-related products (using a regression model).
2. **Lead Status**: Whether a lead is WON or LOST (using a classification model).
""")

st.sidebar.header("Enter Input Parameters")
feature_1 = st.sidebar.number_input('Feature 1 (e.g., Quantity)', min_value=0.0, format="%.2f")
feature_2 = st.sidebar.number_input('Feature 2 (e.g., Quality Index)', min_value=0.0, format="%.2f")
feature_3 = st.sidebar.number_input('Feature 3 (e.g., Market Demand Index)', min_value=0.0, format="%.2f")

features = np.array([[feature_1, feature_2, feature_3]])

if st.sidebar.button("Predict Selling Price"):
    try:
        predicted_price = regression_model.predict(features)[0]
        st.success(f"Predicted Selling Price: ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error in regression prediction: {e}")

if st.sidebar.button("Predict Lead Status (WON/LOST)"):
    try:
        predicted_status = classification_model.predict(features)[0]
        status_label = "WON" if predicted_status == 1 else "LOST"
        st.success(f"Predicted Lead Status: {status_label}")
    except Exception as e:
        st.error(f"Error in classification prediction: {e}")

st.markdown("### About")
st.markdown("This app uses machine learning models to assist the copper industry with sales predictions and lead classifications.")
