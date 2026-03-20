import os
import pickle
from joblib import load, dump
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import logger, PICKLE_MODEL_PATH, JOBLIB_MODEL_PATH, PREDICTION_COUNTER_KEY
from database import redis_client_sync

mlflow.set_tracking_uri("sqlite:///mlflow.db")

global_models = {
    "pickle_model": None,
    "joblib_model": None
}

def prune_model_registry(model_name: str):
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    version_scores = []

    for version in all_versions:
        acc = version.tags.get("accuracy")
        acc_val = float(acc) if acc else -float("inf")
        version_scores.append((version.version, acc_val))

    version_scores.sort(key=lambda x: x[1], reverse=True)
    versions_to_delete = version_scores[2:]

    for version_num, acc_val in versions_to_delete:
        logger.info(f"Deleting model version {version_num} with accuracy {acc_val}")
        client.delete_model_version(name=model_name, version=str(version_num))

def load_models():
    logger.info("Loading local models from disk into memory cache")
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
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("./models", exist_ok=True)
    with open(PICKLE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    dump(model, JOBLIB_MODEL_PATH)

    mlflow.set_experiment("Iris_Experiment")
    with mlflow.start_run() as run:
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            registered_model_name="IrisModel"
        )
        
        client = MlflowClient()
        client.set_model_version_tag(
            name="IrisModel", 
            version=model_info.registered_model_version, 
            key="accuracy", 
            value=str(acc)
        )

    logger.info("Model logged and registered in MLflow")

    prune_model_registry("IrisModel")
    load_models()

    redis_client_sync.set(PREDICTION_COUNTER_KEY, 0)
    logger.info("Redis counter reset to 0 after training")
    
    return len(X_train), len(X_test), list(iris.target_names), acc

def _do_inference(data_inference):
    predict_pickle = global_models["pickle_model"].predict(data_inference)[0]
    predict_joblib = global_models["joblib_model"].predict(data_inference)[0]
    return int(predict_pickle), int(predict_joblib)