import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Diabetes Prediction App")

@st.cache_resource
def get_model():
    return load_model('classification_model')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['prediction_label'][0]

model = get_model()

st.title("Diabetes Prediction App") # Corrected "Diabtes" to "Diabetes"
st.markdown("Choose the values for each attribute of the Diabetes Type that you want to Predict.")

with st.form("Causality Feature"): # Use 'with' statement for the form
    # Sliders you already have and want to be dynamic
    BMI = st.slider('BMI', min_value=0.0, max_value=50.0, value=25.0, step = 0.1, format = '%f')
    Blood_Glucose_Levels = st.slider('Blood Glucose Levels', min_value=0.0, max_value=300.0, value=120.0, step = 0.1, format = '%f')
    Waist_Circumference = st.slider('Waist Circumference', min_value=0.0, max_value=60.0, value=35.0, step = 0.1, format = '%f')
    Cholesterol_Levels = st.slider('Cholesterol Levels', min_value=0.0, max_value=300.0, value=200.0, step = 0.1, format = '%f')
    Insulin_Levels = st.slider('Insulin Levels', min_value=0.0, max_value=60.0, value=15.0, step = 0.1, format = '%f')
    Weight_Gain_During_Pregnancy = st.slider('Weight Gain During Pregnancy', min_value=0.0, max_value=60.0, value=10.0, step = 0.1, format = '%f')

    # You MUST add input widgets for ALL features your model expects
    # Based on your previous hardcoded input_dict, you're missing these from the UI:
    Genetic_Markers = st.number_input('Genetic Markers', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    Family_History = st.selectbox('Family History', [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No") # Example: 0 for No, 1 for Yes
    Glucose_Tolerance_Test = st.slider('Glucose Tolerance Test', min_value=0.0, max_value=300.0, value=140.0, step=0.1)
    Physical_Activity = st.slider('Physical Activity', min_value=0.0, max_value=10.0, value=3.0, step=0.1) # Assuming a numeric scale
    Dietary_Habits = st.selectbox('Dietary Habits', ['Good', 'Average', 'Bad'])
    Smoking_Status = st.selectbox('Smoking Status', ['Yes', 'No'])
    Alcohol_Consumption = st.selectbox('Alcohol Consumption', ['Low', 'Moderate', 'High'])
    Blood_Pressure = st.slider('Blood Pressure', min_value=0.0, max_value=200.0, value=120.0, step=0.1)
    Liver_Function_Tests = st.slider('Liver Function Tests', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    Previous_Gestational_Diabetes = st.selectbox('Previous Gestational Diabetes', [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No") # Example: 0 for No, 1 for Yes
    Pregnancy_History = st.number_input('Pregnancy History (Number of Pregnancies)', min_value=0, max_value=10, value=1)
    Ethnicity = st.selectbox('Ethnicity', ['A', 'B', 'C', 'D']) # Replace with actual ethnicities
    Socioeconomic_Factors = st.selectbox('Socioeconomic Factors', ['Low', 'Mid', 'High'])


    predict_button = st.form_submit_button('Predict')

# IMPORTANT: Construct input_dict using the *variables* from the sliders/selectboxes
input_dict = {
    'BMI': BMI,
    'Blood Glucose Levels': Blood_Glucose_Levels,
    'Waist Circumference': Waist_Circumference,
    'Cholesterol Levels': Cholesterol_Levels,
    'Insulin Levels': Insulin_Levels,
    'Weight Gain During Pregnancy': Weight_Gain_During_Pregnancy,
    'Genetic Markers': Genetic_Markers,
    'Family History': Family_History,
    'Glucose Tolerance Test': Glucose_Tolerance_Test,
    'Physical Activity': Physical_Activity,
    'Dietary Habits': Dietary_Habits,
    'Smoking Status': Smoking_Status,
    'Alcohol Consumption': Alcohol_Consumption,
    'Blood Pressure': Blood_Pressure,
    'Liver Function Tests': Liver_Function_Tests,
    'Previous Gestational Diabetes': Previous_Gestational_Diabetes,
    'Pregnancy History': Pregnancy_History,
    'Ethnicity': Ethnicity,
    'Socioeconomic Factors': Socioeconomic_Factors
}

input_df = pd.DataFrame([input_dict])

if predict_button:
    # Print the DataFrame being passed to the model for debugging
    st.write("DataFrame being sent to prediction:")
    st.dataframe(input_df) # This will show you the exact values and column names

    try:
        out = predict(model, input_df)
        st.success(f'The predicted Diabetes Type is {out}.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all required input features are provided and match the model's training data.")
        st.write("Full Error Details:")
        st.exception(e) # This will print the full traceback in the Streamlit app
