import numpy as np
from confluent_kafka import Consumer
from config import logger, KAFKA_BROKER, RETRAIN_TOPIC
from ml_core import execute_training

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
        execute_training(0.3, np.random.randint(0, 1000))