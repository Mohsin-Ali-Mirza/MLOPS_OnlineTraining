import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("app")

PREDICTION_COUNTER_KEY = "prediction_count"
RETRAIN_THRESHOLD = 10
KAFKA_BROKER = "localhost:9092"
RETRAIN_TOPIC = "model_retrain_topic"

PICKLE_MODEL_PATH = "./models/rf_model.pkl"
JOBLIB_MODEL_PATH = "./models/rf_model.joblib"