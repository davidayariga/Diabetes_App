import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Diabetes Classification App")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('classification_model')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['prediction_label'][0]

model = get_model()


st.title("Diabtes Classification App")
st.markdown("Choose the values for each attribute of the Diabetes Type that you\
        want to be classified.")

form = st.form("Causality Feature")
BMI = form.slider('BMI', min_value=0.0, max_value=50.0, 
                           value=0.0, step = 0.1, format = '%f')
Blood_Glucose_Levels = form.slider('Blood_Glucose_Levels', min_value=0.0, max_value=300.0,
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

input_dict = {'BMI' : BMI, 'Blood_Glucose_Levels' : Blood_Glucose_Levels,
              'Waist_Circumference' : Waist_Circumference, 'Cholesterol_Levels' : Cholesterol_Levels, 
              'Insulin_Levels': Insulin_Levels, 'Weight_Gain_During_Pregnancy': Weight_Gain_During_Pregnancy}
            
input_df = pd.DataFrame([input_dict])

if predict_button:
    out = predict(model, input_df)
    st.success(f'The predicted Diabetes Type is {out}.')
    
