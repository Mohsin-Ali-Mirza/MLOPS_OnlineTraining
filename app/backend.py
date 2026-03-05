from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load, dump
import pickle
import numpy as np
import redis
import uvicorn
import os
import logging
import threading
from confluent_kafka import Producer, Consumer
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
RETRAIN_THRESHOLD = 10
KAFKA_BROKER = "localhost:9092"
RETRAIN_TOPIC = "model_retrain_topic"

PICKLE_MODEL_PATH = "./models/rf_model.pkl"
JOBLIB_MODEL_PATH = "./models/rf_model.joblib"

kafka_producer = Producer({"bootstrap.servers": KAFKA_BROKER})

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

def execute_training(test_size: float = 0.3, random_state: int = 42):
    logger.info("Starting training process")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")

    os.makedirs("./models", exist_ok=True)

    with open(PICKLE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    dump(model, JOBLIB_MODEL_PATH)
    logger.info("Models saved to disk")

    redis_client.set(PREDICTION_COUNTER_KEY, 0)
    logger.info("Redis counter reset to 0 after training")
    
    return len(X_train), len(X_test), list(iris.target_names)

def kafka_consumer_worker():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "training_group",
        "auto.offset.reset": "latest"
    })
    consumer.subscribe([RETRAIN_TOPIC])
    logger.info("Kafka consumer started and listening")

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        logger.info("Received retrain event from Kafka")
        execute_training()

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=kafka_consumer_worker, daemon=True)
    thread.start()

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

    if prediction_count >= RETRAIN_THRESHOLD:
        logger.info("Threshold reached. Publishing retrain event to Kafka.")
        kafka_producer.produce(RETRAIN_TOPIC, value="trigger_retrain")
        kafka_producer.flush()
        redis_client.set(PREDICTION_COUNTER_KEY, 0)
        prediction_count = 0

    return {
        "Prediction_Pickle": int(predict_pickle),
        "Prediction_Joblib": int(predict_joblib),
        "prediction_count": prediction_count,
    }

@app.post("/train")
async def train(request: TrainRequest):
    logger.info("Manual train endpoint hit")
    
    raw_count_before = redis_client.get(PREDICTION_COUNTER_KEY)
    verified_count_before = int(raw_count_before) if raw_count_before else 0

    train_samples, test_samples, target_names = execute_training(request.test_size, request.random_state)

    raw_count_after = redis_client.get(PREDICTION_COUNTER_KEY)
    verified_count_after = int(raw_count_after) if raw_count_after else 0

    return {
        "message": "Model trained and saved successfully.",
        "dataset": "sklearn built-in Iris dataset",
        "target_names": target_names,
        "num_train_samples": train_samples,
        "num_test_samples": test_samples,
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)