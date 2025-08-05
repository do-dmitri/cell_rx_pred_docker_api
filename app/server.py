from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import logging

model = joblib.load('app/cell_RX_model.joblib')

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Cell Rx Run'}

@app.post('/predict')
def predict(data: dict):
    """
    Runs prediction. Takes data as dict {'features': (-74.9, -69.5, -9.9)}
    Returns: predicted NB Rx value as dict {'predicted_rx': -106.092384}
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    prediction = {'predicted_rx': prediction[0]}
    logger.info(prediction)

    return prediction

















