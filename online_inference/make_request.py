import sys
import logging
import requests
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PATH", dest="data_path", type=str, default="../data/raw/heart_cleveland_upload_no_target.csv")
parser.add_argument("-i", dest="ip", type=str, default="0.0.0.0")
parser.add_argument("-p", dest="port", type=str, default="8000")
parser.add_argument("-n", dest="num_requests", type=int)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    data_path = vars(parser.parse_args())["data_path"]
    ip = vars(parser.parse_args())["ip"]
    port = vars(parser.parse_args())["port"]
    num_requests = vars(parser.parse_args())["num_requests"]

    data = pd.read_csv(data_path)
    request_features = list(data.columns)
    
    num_requests = data.shape[0] if not num_requests else num_requests
    url = f"http://{ip}:{port}/predict/"

    logger.info(f"Num requests: {num_requests}")
    logger.info(f"Url: {url}")

    for i in range(num_requests):
        request_data = data.iloc[i].tolist()

        response = requests.get(
            url,
            json={"data": [request_data], "features": request_features} 
        )

        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response data: {response.json()}")