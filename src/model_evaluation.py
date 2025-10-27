import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import json
import pandas as pd
import logging
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import Live
from config.get_params import load_params

# Configure logging
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# data loader function
def load_data(data_path: str) -> pd.DataFrame:
    """This function returns all test data for evaluation"""
    try:
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
        logger.debug("Load the test data!!")
        return X_test, y_test.iloc[:, 0]
    except Exception as e:
        logger.error("Failed to load the test data %s", e)
        raise

# model loader function
def load_model(model_path: str):
    """This function load and return the pickle file"""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model is loaded successfully!!")
        return model
    except FileNotFoundError:
        logger.error("Model is not found at %s", model_path)
        raise
    except Exception as e:
        logger.error("Unexpected error while model loading %s", e)
        raise

# model evaluation function
def evaluate_model(model, xtest, ytest) -> dict:
    """This function calculates the evaluation metrics and return a dictionary"""
    try:
        y_pred = model.predict(xtest)
        accuracy = accuracy_score(ytest, y_pred)
        precision = precision_score(ytest, y_pred, average="weighted")
        recall = recall_score(ytest, y_pred, average="weighted")
        f1 = f1_score(ytest, y_pred, average="weighted")
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1 }
        logger.debug("Model is evaluated successfully!!")

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy)
            live.log_metric('precision', precision)
            live.log_metric('recall', recall)
            live.log_metric('f1_score', f1)
            live.log_params(load_params(r"config\config.yaml"))
        return metrics
    except Exception as e:
        logger.error("Filed to evaluate the model %s", e)
        raise

# save evaluation metrics
def save_metrics(metrics: dict) -> None:
    """Store the evaluation metrics inside the model folder"""
    try:
        eval_path = os.path.join("model", "evaluation_metrics")
        os.makedirs(eval_path, exist_ok=True)
        with open(os.path.join(eval_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Evaluation metrics are saved successfully!!")
    except Exception as e:
        logger.error("Failed to save the evaluation metrics %s", e)
        raise

# main function
def main():
    try:
        logger.info("Start the model evaluation process!!")
        test_data_path = r"data\preprocess_data"
        X_test, y_test = load_data(test_data_path)
        model = load_model(r"model\rf_model.pkl")
        evaluation_metrics = evaluate_model(model, X_test, y_test)
        save_metrics(evaluation_metrics)
    except Exception as e:
        logger.error("Unexpected error while model evaluation process %s", e)
        raise
    
if __name__ == "__main__":
    main()