PROBLEM DESCRIPTION:

Full scale drive test of mobile network cluster requires logging both Uplink and Downlink for the full analysis,
however in reality Uplink measurements are mush harder to perform, as they are logged at infrastructure side which is 
not in cotrol of cell engineer, and require extra logistics,licensing, time, scope of work, coordination,
data processing, etc. As a result, UL analysis is often ignored, leading to a blind spot in evaluating uplink quality.


While some uplink high level characteristics leave a trace mob <-> cell communication, for example we can assume
that mobile UL BER at cell is still good as retransmission rate is low, other low-level parameters cell RX at cell
antenna may not be seen without UL logging. At the same time cell RX comes from mobile Tx power which is linked to
how well mobile receives cell in the first place through different flavours of mob power control algorithms.
Good thing is we don't have to know all bits of those algorithms, we may learn its pattern and predict it with ML
for given RF propagation morphology (basically for a given goe area or cluster). 
There is a good chance high correlation will be found.


TASK:

To predict cell Rx (UL) based on mobile drive test (DL) log having 
- Mobile_rsrp, 
- Mobile_rssi, 
- Mobile_Tx_power



SOLUTION:

A prediction model is chosen from optimum config based on model metrics from
- Linear Regression, 
- Decision Tree, 
- Random Forest, 
- Gradient Boosting 

The model obtained at this step is stored and used for other DL-only drive logs for the same cluster, no need for UL logging in majority of cases.


MODULES AND FILES:

- make_NB_Rx_model.py         - runs predictions and exports 'app/cell_RX_model.joblib' model
- tune_NB_log.csv             - (UL+DL) combined logfile with mobile_rsrp, mobile_rssi, mobile_Tx_power, and cell_Rx
- analysis_NB.csv             - DL-only logfile in the same cluster which require adding UL Rx info. 
                                  DL+UL logfile can be used for benchmarking when prediction is added. 
- app/server.py               - FastAPI Docker server for predictions
- app/requirements.txt        - Docker container vertual environment config
- app/cell_RX_model.joblib    - prediction model 


DEPLOYMENT HOW-TO:

- prepare tune_NB_log.csv (for model creation) and analysis_NB.csv (for prediction) log files.
- run make_NB_Rx_model.py to create model
- setup Docker engine on local client machine (https://docs.docker.com/engine/install)
when Docker is setup 
- $ sudo docker build -t <container_image> .      
- $ sudo docker run --name <container_name> -p 8000:8000 <container_image> 
- run client.py to predict on analysis.csv and populate UL data.
This will generate a file work_NB_log_predicted.csv and display a plot comparing true vs. predicted NodeB Rx values in case of benchmarking.


