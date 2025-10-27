from venv import logger
import yaml
import logging

# Config the logging
logger = logging.getLogger("model_config")
logger.setLevel("DEBUG")

# function that retrieve data from yaml file
def load_params(params_path: str) -> yaml:
    """This function returns all the necessary parameters stored in config.yaml file"""
    try:
        with open(params_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
        return params
    except FileNotFoundError:
        logger.error("Config file not found at %s", params_path)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading config file %s", e)
        raise
