import pandas as pd
import os
import logging

# Configure the logging

log_dir = 'logs' # Create a log directory
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load data function
def load_data(data_link: str) -> pd.DataFrame:
    """This function load the dataset from the given link"""
    try:
        data = pd.read_csv(data_link)
        logger.info("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

# Save data function
def save_data(data: pd.DataFrame, data_path: str) -> None:
    """This function save the dataset to the given file path"""
    try:
        raw_data_path = os.path.join(data_path, 'raw_data')
        os.makedirs(raw_data_path, exist_ok=True)
        data.to_csv(os.path.join(raw_data_path, "data.csv"), index=False)
        logger.info("Dataset is saved successfully!!")
    except Exception as e:
        logger.error("Failed to save data %s", e)
        raise

def main():
    try:
        data_link = "https://raw.githubusercontent.com/Jeet-047/ML-training-with-DVC/refs/heads/main/Weather-related%20disease%20prediction.csv"
        data = load_data(data_link)
        save_data(data, "data")
        logger.info("Data Ingestion is completed")
    except Exception as e:
        logger.error("Failed to complete the Data Ingestion %s", e)
        raise

if __name__ == "__main__":
    main()