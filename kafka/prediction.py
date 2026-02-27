from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load, dump
import pickle
import numpy as np
import redis
import uvicorn
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    try:
        with open(PICKLE_MODEL_PATH, "rb") as f:
            pickle_model = pickle.load(f)
        joblib_model = load(JOBLIB_MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Models not found. Call POST /train first.",
        )
    return [pickle_model, joblib_model]


@app.post("/predict")
async def predict(item: Flower):
    data_inference = np.array([[item.sepalLength, item.sepalWidth, item.petalLength, item.petalWidth]])
    pickle_model, joblib_model = load_models()

    redis_client.incr(PREDICTION_COUNTER_KEY)

    predict_pickle = pickle_model.predict(data_inference)[0]
    predict_joblib = joblib_model.predict(data_inference)[0]

    # Read the current count from Redis after incrementing
    prediction_count = int(redis_client.get(PREDICTION_COUNTER_KEY))

    return {
        "Prediction_Pickle": int(predict_pickle),
        "Prediction_Joblib": int(predict_joblib),
        "prediction_count": prediction_count,
    }


@app.post("/train")
async def train(request: TrainRequest = TrainRequest()):
    # 1. Load the built-in sklearn Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Split into train / test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    # 3. Train the RandomForest model
    model = RandomForestClassifier(random_state=request.random_state)
    model.fit(X_train, y_train)

    # 4. Save both model formats
    os.makedirs("./models", exist_ok=True)

    with open(PICKLE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    dump(model, JOBLIB_MODEL_PATH)

    # 5. Fetch latest prediction count from Redis
    count = redis_client.get(PREDICTION_COUNTER_KEY)
    prediction_count = int(count) if count else 0

    return {
        "message": "Model trained and saved successfully.",
        "dataset": "sklearn built-in Iris dataset",
        "target_names": list(iris.target_names),
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "pickle_path": PICKLE_MODEL_PATH,
        "joblib_path": JOBLIB_MODEL_PATH,
        "prediction_count": prediction_count,
    }


@app.get("/prediction-count")
async def get_prediction_count():
    """Return the total number of predictions made so far."""
    count = redis_client.get(PREDICTION_COUNTER_KEY)
    return {"prediction_count": int(count) if count else 0}


@app.get("/")
async def root():
    return {"message": "hello world"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)