import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")

MODEL_PATH = "improved_house_price_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    p = Path(path)
    if not p.exists():
        st.error(f"Model file not found at {path}. Place your pipeline .pkl in the app folder.")
        return None
    model = joblib.load(path)
    return model

model = load_model()
if model is None:
    st.stop()

try:
    # Attempt to load df if it's not already in the environment (won't work directly in app.py)
    # A better approach is to load and preprocess data within the app.py or pass it
    pass # Placeholder - in a real app, handle data loading properly
except NameError:
    st.error("DataFrame 'df' not found. Please ensure your data loading and preprocessing steps are included in app.py or run before generating app.py")
    st.stop()


data = pd.read_csv("house_prices.csv")
PROPERTY_TYPES = sorted(data["property_type"].dropna().unique().tolist())
LOCATIONS = sorted(data["location"].dropna().unique().tolist())
CITIES = sorted(data["city"].dropna().unique().tolist())
AREA_CATEGORIES = ["Small", "Medium", "Large"]




with st.form(key="predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        property_type = st.selectbox("Property type", PROPERTY_TYPES)
        location = st.selectbox("Location", LOCATIONS)
        city = st.selectbox("City", CITIES)
        area = st.number_input("Area (in Marla)", min_value=0.0, step=0.1, value=5.0)
    with col2:
        baths = st.number_input("Bathrooms", min_value=0, step=1, value=2)
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=2)
        area_category = st.selectbox("Area category", AREA_CATEGORIES, index=0 if AREA_CATEGORIES else 0) # Handle empty list case
        price_per_marla_input = st.text_input("Price per Marla", 
            value="7000000")
    submit = st.form_submit_button("Predict Price")

if submit:
    sample = pd.DataFrame([{
        "property_type": property_type,
        "location": location,
        "city": city,
        "baths": baths,
        "bedrooms": bedrooms,
        "Area_in_Marla": area,
        "area_category": area_category,
        "price_per_marla": price_per_marla_input }])
    try:
        pred = model.predict(sample)
        pred_value = float(pred[0])

        if pred_value < 20:
            predicted_price = np.exp(pred_value)
            used_scale = "exponentiated from log"
        else:
            predicted_price = pred_value
            used_scale = "direct"

        st.success("Prediction ready")
        st.write(f"**Predicted Price ({used_scale}):** Rs {predicted_price:,.0f}")
        st.write(f"**Raw model output:** {pred_value:.4f}")

        st.subheader("Input summary")
        st.table(sample.T)

    except Exception as e:
        st.error("Prediction failed. See error below:")
        st.exception(e)

