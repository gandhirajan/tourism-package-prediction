import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="gandhirajan/tourism-package-prediction", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction...
st.title("Tour Package Prediction App")
st.write("""
This application predicts the likelihood of a customer selecting the package parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
st.header("User Input")

Age  =  st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
TypeofContact  =  st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("CityTier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of person visiting", min_value=1, max_value=10, value=2, step=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=2, max_value=5, value=3, step=1)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of trips", min_value=1, max_value=10, value=2, step=1)
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children visiting", min_value=0, max_value=5, value=0, step=1)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=50000, step=100)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
NumberOfFollowups = st.number_input("Number of Followups", min_value=1, max_value=10, value=2, step=1)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=100, value=30, step=1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard","Super Deluxe", "King"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Package selected" if prediction == 1 else "Package not selected"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
