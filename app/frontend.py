import streamlit as st
import numpy as np
import pickle
import joblib
from joblib import load
import requests

def display_iamge(prediction):
    file_path = f"images/{prediction}.jpg"
    st.image(file_path)

def predict_flower(data_inference):

    response = requests.post("http://127.0.0.1:8000/predict", json=data_inference)
    if response.status_code == 200:
        predictions = response.json()
        st.write("The predicition form Pickle:",predictions["Prediction_Pickle"])
        display_iamge(prediction=predictions["Prediction_Pickle"])
        st.write("The prediction from JobLib:",predictions["Prediction_Joblib"])
        display_iamge(prediction=predictions["Prediction_Joblib"])


def main():
    st.title("Iris Classification")

    sepalLength = st.number_input("Enter Sepal Length in Cm", min_value=0.0,max_value=10.0, value=0.0)
    sepalWidth =  st.number_input("Enter Sepal Width in Cm", min_value=0.0,max_value=10.0, value=0.0)

    petalLength = st.number_input("Enter Petal Length in Cm", min_value=0.0,max_value=10.0, value=0.0)
    petalWidth = st.number_input("Enter Petal Width in Cm", min_value=0.0,max_value=10.0, value=0.0)

    data_inference = {
                    "sepalLength": sepalLength,
                    "sepalWidth": sepalWidth,
                    "petalLength": petalLength,
                    "petalWidth": petalLength
                }

    if(st.button("Predict",type="primary")):
        predict_flower(data_inference=data_inference)
    

if __name__ == "__main__":
    main()