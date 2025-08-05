'''
Script for inferencing the deployed model
Runs prediction for every row in a logfile.
Adds 'Predicted NB_RX' column and exports to work_NB_log_predicted.csv
'''

import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

url = 'http://localhost:8000/predict/'

log_df = pd.read_csv('work_NB_log.csv')

log_df = log_df[['RSRP', 'RSSI', 'UE_TX_Power', 'NodeB_RX']]
data_df = log_df[['RSRP', 'RSSI', 'UE_TX_Power']]
predicted_values = []

for row in data_df.itertuples(index=False, name=None):
    feat_dict = {'features': list(row)}
    feat_dict = json.dumps(feat_dict)
    try:
        response = requests.post(url, data=feat_dict)
        predicted_values.append(response.json()['predicted_rx'])
    except Exception as e:
        print("Prediction failed for:", row)
        predicted_values.append(None)

log_df['Predicted NB_RX'] = predicted_values
log_df.to_csv('work_NB_log_predicted.csv', index=False)

plt.figure()
axis_x = range(len(log_df['NodeB_RX']))
plt.plot(axis_x, log_df['NodeB_RX'], c='navy', label='ground_truth')
plt.plot(axis_x, log_df['Predicted NB_RX'], c='orange', label='prediction')
plt.legend()
plt.show()
