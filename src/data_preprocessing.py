import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Configure the logging
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# preprocess data function
"""This function select the important features and transform the columns"""
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # split the dependent and independent attributes
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # filter the top 20 attributes
        model = RandomForestClassifier()
        rfe = RFE(model, n_features_to_select=20)
        X_new = rfe.fit_transform(X, y)
        # split the dataset into train test and split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=47)
        # standardize the attributes
        scaler = StandardScaler()
        X_train[:, 0:4] = scaler.fit_transform(X_train[:, 0:4])
        X_test[:, 0:4] = scaler.transform(X_test[:, 0:4])
        logger.info("Data pre-processed successfully!!")
        # return the spit dataset
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Failed to pre-process the data %s", e)
        raise

# function for save data
def save_data(xtrain, xtest, ytrain, ytest, data_path:str) -> None:
    """This function save the dataset inside the given data path"""
    try:
        preprocess_data_path = os.path.join(data_path, "preprocess_data")
        os.makedirs(preprocess_data_path, exist_ok=True)
        # Convert numpy arrays to DataFrame/Series before saving
        pd.DataFrame(xtrain).to_csv(os.path.join(preprocess_data_path, "X_train.csv"), index=False)
        pd.DataFrame(xtest).to_csv(os.path.join(preprocess_data_path, "X_test.csv"), index=False)
        pd.Series(ytrain).to_csv(os.path.join(preprocess_data_path, "y_train.csv"), index=False, header=['prognosis'])
        pd.Series(ytest).to_csv(os.path.join(preprocess_data_path, "y_test.csv"), index=False, header=['prognosis'])
        logger.info("Pre-processed data is saved successfully!!")
    except Exception as e:
        logger.error("Failed to save data after pre-processing %s", e)
        raise

# define main function
def main():
    try:
        logger.info("Data pre-processing is started!!")
        data = pd.read_csv(r"data\raw_data\data.csv")
        X_train, X_test, y_train, y_test = preprocess_data(data)
        # save data
        save_data(X_train, X_test, y_train, y_test, "data")
    except Exception as e:
        logger.error("Failed to complete the data pre-processing %s", e)
        raise

if __name__ == "__main__":
    main()

