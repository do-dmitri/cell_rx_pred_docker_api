PROBLEM DESCRIPTION:

Full scale drive test of mobile network cluster requires logging both Uplink and Downlink for the full analysis,
however in reality Uplink measurements are mush harder to perform, as they are logged at infrastructure side which is 
not in control of RAN engineer, and require extra logistics, licensing, time, data processing. 
As a result, UL analysis is often ignored.


While some uplink high level characteristics leave a trace in mob <-> cell communication, for example we can assume
that mobile UL BER at cell is still good as retransmission rate is low, other low-level parameters like cell RX at cell
antenna may not be seen without UL logging. At the same time cell RX is linked to UE Tx power which is linked to
how well mobile receives cell through different flavours of mob power control algorithms.
Good thing is we don't have to know all bits of those algorithms, we may learn its pattern and predict it with ML
for given RF propagation morphology (basically for a given goe area or cluster). 
There is a good chance high correlation can be found.


TASK:

To predict cell Rx (UL) based on mobile drive test (DL) log having 
- Mobile_rsrp, 
- Mobile_rssi, 
- Mobile_Tx_power


SOLUTION:
UL and DL are logged once for creation of prediction model. The model is chosen from optimum config based on model 
metrics from
- Linear Regression, 
- Decision Tree, 
- Random Forest, 
- Gradient Boosting 

Next model is used for other drive logs for the same cluster, where we need to predict NodeB Rx.
In this code the drivetest log where we need to predict NodeB Rx has values measured, so we can compare 
measured vs predicted at the end for demo purpose. 


MODULES AND FILES:

- make_NB_Rx_model.py         - runs predictions and exports 'app/cell_RX_model.joblib' model
- tune_NB_log.csv             - (UL+DL) combined logfile with mobile_rsrp, mobile_rssi, mobile_Tx_power, and cell_Rx
- work_NB_log.csv             - a logfile in the same cluster where we predict NodeB Rx  
- requirements.txt            - virtual environment config
- app/server.py               - FastAPI Docker server for predictions
- app/cell_RX_model.joblib    - prediction model 


DEPLOYMENT HOW-TO:

- prepare 'tune_NB_log.csv' (for model creation) and work_NB_log.csv (for prediction) log files.
- run make_NB_Rx_model.py to create 'app/cell_RX_model.joblib' model
- setup Docker engine on local client machine (https://docs.docker.com/engine/install)
When Docker is setup run in term: 
- $ sudo docker build -t <container_image> .      
- $ sudo docker run --name <container_name> -p 8000:8000 <container_image> 
- run client.py on local machine to predict on work_NB_log.csv data and populate UL prediction.

This will generate 'work_NB_log_predicted.csv' file with measured (truth) and predicted NodeB Rx both, 
and you will see a plot comparing them.


