from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load, dump
import pickle
import numpy as np
import redis
import uvicorn
import os
import logging
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("app")

app = FastAPI()

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
PREDICTION_COUNTER_KEY = "prediction_count"

PICKLE_MODEL_PATH = "./models/rf_model.pkl"
JOBLIB_MODEL_PATH = "./models/rf_model.joblib"

class Flower(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float

class TrainRequest(BaseModel):
    test_size: float = 0.3
    random_state: int = 42

def load_models():
    logger.debug("Loading models from disk")
    with open(PICKLE_MODEL_PATH, "rb") as f:
        pickle_model = pickle.load(f)
    joblib_model = load(JOBLIB_MODEL_PATH)
    logger.info("Models loaded successfully")
    return [pickle_model, joblib_model]

@app.post("/predict")
async def predict(item: Flower):
    logger.info("Prediction endpoint hit")
    data_inference = np.array([[item.sepalLength, item.sepalWidth, item.petalLength, item.petalWidth]])
    pickle_model, joblib_model = load_models()

    redis_client.incr(PREDICTION_COUNTER_KEY)

    predict_pickle = pickle_model.predict(data_inference)[0]
    predict_joblib = joblib_model.predict(data_inference)[0]

    prediction_count = int(redis_client.get(PREDICTION_COUNTER_KEY))
    logger.info("Prediction successful")

    return {
        "Prediction_Pickle": int(predict_pickle),
        "Prediction_Joblib": int(predict_joblib),
        "prediction_count": prediction_count,
    }
@app.post("/train")
async def train(request: TrainRequest):
    logger.info("Train endpoint hit")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    model = RandomForestClassifier(random_state=request.random_state)
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")

    os.makedirs("./models", exist_ok=True)

    with open(PICKLE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    dump(model, JOBLIB_MODEL_PATH)
    logger.info("Models saved to disk")

    raw_count_before = redis_client.get(PREDICTION_COUNTER_KEY)
    verified_count_before = int(raw_count_before) if raw_count_before else 0

    redis_client.set(PREDICTION_COUNTER_KEY, 0)
    logger.info("Redis counter reset to 0")

    raw_count_after = redis_client.get(PREDICTION_COUNTER_KEY)
    verified_count_after = int(raw_count_after) if raw_count_after else 0

    return {
        "message": "Model trained and saved successfully.",
        "dataset": "sklearn built-in Iris dataset",
        "target_names": list(iris.target_names),
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "pickle_path": PICKLE_MODEL_PATH,
        "joblib_path": JOBLIB_MODEL_PATH,
        "prediction_count_in_redis_before_training": verified_count_before,
        "prediction_count_in_redis_after_training": verified_count_after,
    }
@app.get("/prediction-count")
async def get_prediction_count():
    logger.info("Prediction count endpoint hit")
    count = redis_client.get(PREDICTION_COUNTER_KEY)
    return {"prediction_count_in_redis": int(count) if count else 0}

@app.get("/")
async def root():
    logger.info("Root endpoint hit")
    return {"message": "hello world"}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)