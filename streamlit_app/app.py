import streamlit as st
import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Voyage Analytics",
    layout="wide",
    page_icon="✈️"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "app", "models")

# --------------------------------------------------
# LOAD MODELS (NO LAG)
# --------------------------------------------------
@st.cache_resource
def load_models():
    flight_model = joblib.load(os.path.join(MODEL_DIR, "flight_price_model.pkl"))
    flight_encoders = joblib.load(os.path.join(MODEL_DIR, "flight_encoders.pkl"))
    gender_model = joblib.load(os.path.join(MODEL_DIR, "gender_classification_model.pkl"))
    hotel_features = joblib.load(os.path.join(MODEL_DIR, "hotel_features.pkl"))
    hotel_scaler = joblib.load(os.path.join(MODEL_DIR, "hotel_scaler.pkl"))
    return flight_model, flight_encoders, gender_model, hotel_features, hotel_scaler

flight_model, flight_encoders, gender_model, hotel_features, hotel_scaler = load_models()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🧠 Voyage Analytics")
module = st.sidebar.selectbox(
    "Choose Module",
    ["Gender Prediction", "Flight Price Prediction", "Hotel Recommendation"]
)

# ==================================================
# 👤 GENDER PREDICTION (FIXED)
# ==================================================
if module == "Gender Prediction":
    st.title("👤 Gender Prediction")

    age = st.slider("Age", 10, 80, 30)

    if st.button("Predict Gender"):
        X = np.array([[age]])  # ✅ ONLY ONE FEATURE
        pred = gender_model.predict(X)[0]
        gender = "Female" if pred == 1 else "Male"
        st.success(f"Predicted Gender: **{gender}**")

# ==================================================
# ✈️ FLIGHT PRICE PREDICTION (FIXED)
# ==================================================
elif module == "Flight Price Prediction":
    st.title("✈️ Flight Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        agency = st.selectbox("Agency", flight_encoders["agency"].classes_)
        flight_type = st.selectbox("Flight Type", flight_encoders["flightType"].classes_)
        from_city = st.selectbox("From", flight_encoders["from"].classes_)

    with col2:
        to_city = st.selectbox("To", flight_encoders["to"].classes_)
        user_code = st.selectbox("User Code", flight_encoders["userCode"].classes_)
        distance = st.slider("Distance (km)", 100, 5000, 1200)

    if st.button("Predict Flight Price"):
        X = [[
            flight_encoders["agency"].transform([agency])[0],
            flight_encoders["flightType"].transform([flight_type])[0],
            flight_encoders["from"].transform([from_city])[0],
            flight_encoders["to"].transform([to_city])[0],
            flight_encoders["userCode"].transform([user_code])[0],
            distance
        ]]

        price = flight_model.predict(X)[0]
        st.success(f"Estimated Flight Price: **₹{int(price):,}**")

# ==================================================
# 🏨 HOTEL RECOMMENDATION (REAL)
# ==================================================
elif module == "Hotel Recommendation":
    st.title("🏨 Hotel Recommendation")

    hotel_index = st.number_input(
        "Hotel Index",
        min_value=0,
        max_value=len(hotel_features) - 1,
        value=5
    )

    k = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Recommend Hotels"):
        scaled = hotel_scaler.transform(hotel_features)
        similarity = cosine_similarity(scaled)

        scores = list(enumerate(similarity[hotel_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        recommended = [i[0] for i in scores[1:k+1]]
        st.success(f"Recommended Hotel Indices: **{recommended}**")
