from fastapi import FastAPI
import numpy as np
import uvicorn
import threading
import asyncio

from config import logger, PREDICTION_COUNTER_KEY, RETRAIN_THRESHOLD, RETRAIN_TOPIC
from schemas import Flower, TrainRequest
from database import redis_client_async, kafka_producer
from ml_core import load_models, execute_training, _do_inference
from worker import kafka_consumer_worker

app = FastAPI()

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

    train_samples, test_samples, target_names, accuracy = await asyncio.to_thread(
        execute_training, request.test_size, request.random_state
    )

    raw_count_after = await redis_client_async.get(PREDICTION_COUNTER_KEY)
    verified_count_after = int(raw_count_after) if raw_count_after else 0

    return {
        "message": "Model trained, registered to MLflow, and models pruned successfully.",
        "dataset": "sklearn built-in Iris dataset",
        "target_names": target_names,
        "num_train_samples": train_samples,
        "num_test_samples": test_samples,
        "new_model_accuracy": accuracy,
        "prediction_count_in_redis_before_training": verified_count_before,
        "prediction_count_in_redis_after_training": verified_count_after,
    }

@app.get("/prediction-count")
async def get_prediction_count():
    count = await redis_client_async.get(PREDICTION_COUNTER_KEY)
    return {"prediction_count_in_redis": int(count) if count else 0}

@app.get("/")
async def root():
    return {"message": "hello world"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)