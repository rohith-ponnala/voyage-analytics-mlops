# Voyage Analytics
Machine Learning | MLOps | Streamlit | Docker | MLflow

Voyage Analytics – Travel Intelligence System is an end-to-end Machine Learning project designed to solve real-world problems in the travel and tourism industry.

This system integrates three intelligent modules:

1. ✈️ Flight Price Prediction  
   Predicts the estimated flight ticket price based on travel details such as airline agency, source city, destination city, travel class, distance, travel timing, and booking parameters.

2. 👤 Gender Prediction Model  
   Predicts customer gender based on behavioral and financial attributes such as age, number of bookings, total spending, and annual income.

3. 🏨 Hotel Recommendation System  
   Recommends similar hotels based on feature similarity using scaling and similarity computation techniques.

The project follows a complete Machine Learning lifecycle:
- Data Preprocessing
- Feature Engineering
- Model Training (Random Forest Regressor & Classifier)
- Model Evaluation
- Model Serialization using Joblib
- Deployment using Streamlit
- Containerization using Docker
- Kubernetes Enablement (MLOps integration)

This project demonstrates how ML models can be productionized and deployed as an interactive web application.
## How to Run Locally

1. Install dependencies
#pip install -r requirements.txt

2. Run Streamlit
#cd streamlit_app
#streamlit run app.py

