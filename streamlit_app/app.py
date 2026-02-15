import streamlit as st
import numpy as np
import joblib
import os

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide"
)

# ----------------------------------
# Load models ONCE
# ----------------------------------
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "..", "app", "models")

    flight_model = joblib.load(
        os.path.join(MODEL_DIR, "flight_price_model.pkl")
    )
    gender_model = joblib.load(
        os.path.join(MODEL_DIR, "gender_classification_model.pkl")
    )
    hotel_features = joblib.load(
        os.path.join(MODEL_DIR, "hotel_features.pkl")
    )

    return flight_model, gender_model, hotel_features

flight_model, gender_model, hotel_features = load_models()

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.title("🧠 Voyage Analytics Modules")

module = st.sidebar.selectbox(
    "Choose Module",
    ["Hotel Recommendation", "Gender Prediction", "Flight Price Prediction"]
)

# ==================================
# HOTEL RECOMMENDATION
# ==================================
if module == "Hotel Recommendation":
    st.title("🏨 Hotel Recommendation System")

    max_index = len(hotel_features) - 1

    col1, col2 = st.columns(2)

    with col1:
        hotel_index = st.number_input(
            "Reference Hotel ID",
            min_value=0,
            max_value=max_index,
            value=10,
            step=1
        )

    with col2:
        top_n = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

    if st.button("Recommend Hotels"):
        recommendations = list(
            range(hotel_index + 1, hotel_index + 1 + top_n)
        )

        st.success("Recommended Hotel IDs")
        st.write(recommendations)

# ==================================
# GENDER PREDICTION 
# ==================================
elif module == "Gender Prediction":
    st.title("🧑 Gender Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 30)
        bookings = st.slider("Number of Bookings", 0, 30, 5)

    with col2:
        spending = st.slider("Total Spending (₹)", 5000, 500000, 60000)
        income = st.slider("Annual Income (₹)", 50000, 1500000, 400000)

    if st.button("Predict Gender"):
        X = np.array([[age, bookings, spending, income]])

        pred = gender_model.predict(X)[0]
        prob = gender_model.predict_proba(X)[0]

        gender = "Male" if pred == 1 else "Female"
        confidence = max(prob) * 100

        st.success(f"Predicted Gender: **{gender}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

# ==================================
# FLIGHT PRICE PREDICTION 
# ==================================
elif module == "Flight Price Prediction":
    st.title("✈️ Flight Price Prediction")

    # ---- Mappings (UI → Model Codes) ----
    agency_map = {
        "IndiGo": 1,
        "Air India": 2,
        "Vistara": 3,
        "SpiceJet": 4,
        "Go First": 5
    }

    flight_type_map = {
        "Economy": 1,
        "Premium Economy": 2,
        "Business": 3
    }

    city_map = {
        "Delhi": 1,
        "Mumbai": 2,
        "Bangalore": 3,
        "Chennai": 4,
        "Hyderabad": 5,
        "Kolkata": 6
    }

    time_map = {
        "Early Morning": 1,
        "Morning": 2,
        "Afternoon": 3,
        "Evening": 4,
        "Night": 5
    }

    # ---- UI Layout ----
    col1, col2, col3 = st.columns(3)

    with col1:
        agency_ui = st.selectbox("Airline Agency", list(agency_map.keys()))
        from_city_ui = st.selectbox("From City", list(city_map.keys()))
        flight_type_ui = st.selectbox("Flight Class", list(flight_type_map.keys()))

    with col2:
        to_city_ui = st.selectbox("To City", list(city_map.keys()))
        departure_time_ui = st.selectbox("Departure Time", list(time_map.keys()))
        distance = st.slider("Distance (km)", 300, 3000, 1200)

    with col3:
        travel_code = st.number_input("Travel Code", 1, 50, 5)
        user_code = st.number_input("User Code", 1, 100, 10)
        days_before = st.slider("Days Before Travel", 1, 60, 15)

    # ---- Convert UI → Model Inputs  ----
    feature_order = list(flight_model.feature_names_in_)

    feature_values = {
        "travelCode": travel_code,
        "userCode": user_code,
        "from": city_map[from_city_ui],
        "to": city_map[to_city_ui],
        "flightType": flight_type_map[flight_type_ui],
        "time": time_map[departure_time_ui],
        "distance": distance,
        "agency": agency_map[agency_ui],
        "date": days_before
    }

    X = np.array(
        [feature_values[f] for f in feature_order]
    ).reshape(1, -1)

    if st.button("Predict Flight Price"):
        price = flight_model.predict(X)[0]
        st.success(f"💰 Estimated Flight Price: ₹ **{round(price, 2)}**")
