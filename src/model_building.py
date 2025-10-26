import pickle
import pandas as pd
import os
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier

# Configure the logging
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# data loader function
def load_data(data_path: str) -> pd.DataFrame:
    """This function returns all my pre-processed data for training"""
    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
        logger.debug("Load the pre-processed data!!")
        return X_train, y_train.iloc[:, 0]  # or y_train.squeeze()
    except Exception as e:
        logger.error("Failed to load the pre-processed data %s", e)
        raise

# model training function
def train_model(xtrain, ytrain) -> RandomForestClassifier:
    """Train the model and return the model"""
    try:
        rfc = RandomForestClassifier()
        rfc.fit(xtrain, ytrain)
        logger.debug("Model is trained successfully!!")
        return rfc
    except Exception as e:
        logger.error("Failed to train the model %s", e)
        raise

# function for save model
def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the given model as a pickle file on the given path"""
    try:
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "rf_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model is saved successfully!!")
    except Exception as e:
        logger.error("Failed to save the model %s", e)
        raise

# main function
def main():
    try:
        logger.debug("Start the model building process!!")
        preprocess_data_path = r"data\preprocess_data"
        X_train, y_train = load_data(preprocess_data_path)
        model = train_model(X_train, y_train)
        # save the model
        save_model(model, "model")
    except Exception as e:
        logger.error("Unexpected error while model training process")
        raise

if __name__ == "__main__":
    main()
        