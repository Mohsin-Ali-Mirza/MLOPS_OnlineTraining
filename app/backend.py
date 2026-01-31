from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from joblib import load
import pickle
import numpy as np

app = FastAPI()

class Flower(BaseModel):
    sepalLength : float
    sepalWidth: float
    petalLength: float
    petalWidth: float

def load_models():
    pickle_file_path = "xgb_model.pkl"
    joblib_file_path = "xgb_model.joblib"

    with open(pickle_file_path,"rb") as f:
        pickle_model = pickle.load(f)
    
    joblib_model = load(joblib_file_path)

    return [pickle_model,joblib_model]

@app.post("/predict")
async def predict(item: Flower):
    data_inference = np.array([[item.sepalLength,item.sepalWidth,item.petalLength,item.petalWidth]])
    pickle_model, joblib_model = load_models()


    predict_pickle = pickle_model.predict(data_inference)[0]
    predict_joblib = joblib_model.predict(data_inference)[0]
    return {"Prediction_Pickle":int(predict_pickle),
            "Prediction_Joblib":int(predict_joblib)}


@app.get("/")
async def root():
    return {"message":"hello world"}