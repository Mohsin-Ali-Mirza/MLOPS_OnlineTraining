from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load, dump
import pickle
import numpy as np
import redis.asyncio as aioredis
import redis as sync_redis
import uvicorn
import os
import logging
import threading
import asyncio
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

redis_client_async = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
redis_client_sync = sync_redis.Redis(host="localhost", port=6379, decode_responses=True)

PREDICTION_COUNTER_KEY = "prediction_count"
RETRAIN_THRESHOLD = 10
KAFKA_BROKER = "localhost:9092"
RETRAIN_TOPIC = "model_retrain_topic"

PICKLE_MODEL_PATH = "./models/rf_model.pkl"
JOBLIB_MODEL_PATH = "./models/rf_model.joblib"

kafka_producer = Producer({"bootstrap.servers": KAFKA_BROKER})

global_models = {
    "pickle_model": None,
    "joblib_model": None
}

class Flower(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float

class TrainRequest(BaseModel):
    test_size: float = 0.3
    random_state: int = 42

def load_models():
    logger.debug("Loading models from disk into memory cache")
    with open(PICKLE_MODEL_PATH, "rb") as f:
        global_models["pickle_model"] = pickle.load(f)
    global_models["joblib_model"] = load(JOBLIB_MODEL_PATH)
    logger.info("Models loaded successfully into memory")

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

    load_models()
    
    return len(X_train), len(X_test), list(iris.target_names)

def _do_inference(data_inference):
    predict_pickle = global_models["pickle_model"].predict(data_inference)[0]
    predict_joblib = global_models["joblib_model"].predict(data_inference)[0]
    return int(predict_pickle), int(predict_joblib)

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
        redis_client_sync.set(PREDICTION_COUNTER_KEY, 0)
        logger.info("Redis counter reset to 0 by background worker")

@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(load_models)
    thread = threading.Thread(target=kafka_consumer_worker, daemon=True)
    thread.start()

@app.post("/predict")
async def predict(item: Flower):
    logger.info("Prediction endpoint hit")
    data_inference = np.array([[item.sepalLength, item.sepalWidth, item.petalLength, item.petalWidth]])
    
    predict_pickle, predict_joblib = await asyncio.to_thread(_do_inference, data_inference)

    await redis_client_async.incr(PREDICTION_COUNTER_KEY)
    prediction_count = int(await redis_client_async.get(PREDICTION_COUNTER_KEY))
    logger.info("Prediction successful")

    if prediction_count >= RETRAIN_THRESHOLD:
        logger.info("Threshold reached. Publishing retrain event to Kafka.")
        await asyncio.to_thread(kafka_producer.produce, RETRAIN_TOPIC, value="trigger_retrain")
        await asyncio.to_thread(kafka_producer.flush)
        await redis_client_async.set(PREDICTION_COUNTER_KEY, 0)
        prediction_count = 0

    return {
        "Prediction_Pickle": predict_pickle,
        "Prediction_Joblib": predict_joblib,
        "prediction_count": prediction_count,
    }

@app.post("/train")
async def train(request: TrainRequest):
    logger.info("Manual train endpoint hit")
    
    raw_count_before = await redis_client_async.get(PREDICTION_COUNTER_KEY)
    verified_count_before = int(raw_count_before) if raw_count_before else 0

    train_samples, test_samples, target_names = await asyncio.to_thread(
        execute_training, request.test_size, request.random_state
    )

    await redis_client_async.set(PREDICTION_COUNTER_KEY, 0)
    logger.info("Redis counter reset to 0 after manual training")

    raw_count_after = await redis_client_async.get(PREDICTION_COUNTER_KEY)
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
    count = await redis_client_async.get(PREDICTION_COUNTER_KEY)
    return {"prediction_count_in_redis": int(count) if count else 0}

@app.get("/")
async def root():
    logger.info("Root endpoint hit")
    return {"message": "hello world"}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)