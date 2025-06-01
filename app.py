import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Diabetes Prediction App")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('classification_model')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['prediction_label'][0]

model = get_model()


st.title("Diabtes Classification App")
st.markdown("Choose the values for each attribute of the Diabetes Type that you\
        want to Predict.")

form = st.form("Causality Feature")
BMI = form.slider('BMI', min_value=0.0, max_value=50.0, 
                           value=0.0, step = 0.1, format = '%f')
Blood_Glucose_Levels = form.slider('Blood Glucose Levels', min_value=0.0, max_value=300.0,
                           value=0.0, step = 0.1, format = '%f')
Waist_Circumference = form.slider('Waist Circumference', min_value=0.0, max_value=60.0,
                           value=0.0, step = 0.1, format = '%f')
Cholesterol_Levels = form.slider('Cholesterol Levels', min_value=0.0, max_value=300.0,
                           value=0.0, step = 0.1, format = '%f')
Insulin_Levels = form.slider('Insulin Levels', min_value=0.0, max_value=60.0,
                           value=0.0, step = 0.1, format = '%f')
Weight_Gain_During_Pregnancy = form.slider('Weight Gain During Pregnancy', min_value=0.0, max_value=60.0,
                           value=0.0, step = 0.1, format = '%f')
    
predict_button = form.form_submit_button('Predict')

input_dict = {
    'Genetic Markers': 0.5,
    'Family History': 1,
    'Insulin Levels': 150.0,
    'Blood Glucose Levels': 120.0,  # This will be overridden or cleaned
    'Glucose Tolerance Test': 140.0,
    'Waist Circumference': 90.0,
    'Physical Activity': 3.0,      # This will be overridden or cleaned
    'Dietary Habits': 'Good',
    'Smoking Status': 'No',
    'Alcohol Consumption': 'Low',
    'Blood Pressure': 120.0,       # This will be overridden or cleaned
    'Cholesterol Levels': 200.0,
    'Liver Function Tests': 50.0,
    'Previous Gestational Diabetes': 0, # This will be overridden or cleaned
    'Pregnancy History': 1,
    'Weight Gain During Pregnancy': 10.0,
    'Ethnicity': 'A',
    'Socioeconomic Factors': 'Mid',
    'BMI': 50
}
input_df = pd.DataFrame([input_dict])

if predict_button:
    out = predict(model, input_df)
    st.success(f'The predicted Diabetes Type is {out}.')
    
