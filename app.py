import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

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

# Assuming df is loaded and processed earlier in the notebook
# If not, you need to load and preprocess your data here in the app.py file
# For this example, I'll use the processed df from the notebook
# In a real Streamlit app, you'd load and process the data within app.py
# For demonstration purposes, let's assume df is available or load it again
try:
    # Attempt to load df if it's not already in the environment (won't work directly in app.py)
    # A better approach is to load and preprocess data within the app.py or pass it
    pass # Placeholder - in a real app, handle data loading properly
except NameError:
    st.error("DataFrame 'df' not found. Please ensure your data loading and preprocessing steps are included in app.py or run before generating app.py")
    st.stop()

# Define these based on your data's unique values.
# In a real app, you might load these from a config or preprocess your data to get them.
# For this example, let's use dummy lists or load from a saved file if available
# If you ran the notebook sequentially, these variables would be defined before writing app.py
# Assuming you have executed the cell defining these lists:
try:
    PROPERTY_TYPES = PROPERTY_TYPES # Accessing from notebook environment (for demonstration)
    LOCATIONS = LOCATIONS
    CITIES = CITIES
    AREA_CATEGORIES = AREA_CATEGORIES
except NameError:
     st.error("Categorical lists (PROPERTY_TYPES, LOCATIONS, CITIES, AREA_CATEGORIES) not found. Please ensure you define these before writing app.py, or load them within app.py")
     st.stop()


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
    submit = st.form_submit_button("Predict Price")

if submit:
    # Build a DataFrame in the same shape expected by pipeline
    sample = pd.DataFrame([{
        "property_type": property_type,
        "location": location,
        "city": city,
        "baths": baths,
        "bedrooms": bedrooms,
        "Area_in_Marla": area,
        "area_category": area_category
    }])
    try:
        pred = model.predict(sample)
        pred_value = float(pred[0])

        if pred_value < 20:
            predicted_price = np.exp(pred_value)
            used_scale = "exponentiated from log"
        else:
            predicted_price = pred_value
            used_scale = "direct"

        st.success("✅ Prediction ready")
        st.write(f"**Predicted Price ({used_scale}):** Rs {predicted_price:,.0f}")
        st.write(f"**Raw model output:** {pred_value:.4f}")

        st.subheader("Input summary")
        st.table(sample.T)

    except Exception as e:
        st.error("Prediction failed. See error below:")
        st.exception(e)

st.markdown("---")
st.markdown(
    """
    **Tips:**
    - This app expects the model pipeline `.pkl` to contain the preprocessor (onehot/scaler) and the regressor.
    - If you trained the model with `log(price)` as target, the app attempts to detect that and exponentiate the output.
    - Update LOCATION / CITY lists above for better UX or change the widgets to `st.text_input` if you prefer free text.
    """
)
