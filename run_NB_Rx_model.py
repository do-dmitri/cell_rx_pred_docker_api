
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt


pd.set_option('display.width', None) # disable scroll


def try_NB_rx_predict_model(logname):
    diag_message = ''

    df = pd.read_csv(logname, sep=',')
    df = df[['RSRP', 'RSSI', 'UE_TX_Power', 'NodeB_RX']]

    X = df.drop('NodeB_RX', axis=1).copy()
    y = df['NodeB_RX'].copy()

    model = joblib.load('cell_RX_model.joblib')

    diag_message += '\n\nTrying model on new logfile ---------------------------------------------'

    model_pred = model.predict(X)

    diag_message += '\nScore: {:.4f}'.format(model.score(X, y))
    diag_message += '\nR2   : {:.4f}'.format(r2_score(y, model_pred))
    diag_message += '\nMSE  : {:.4f}'.format(mean_squared_error(y, model_pred))

    cv_score_mean = cross_val_score(LinearRegression(), X, y, cv=3, scoring='r2').mean()
    diag_message += '\nCV mean: {:.4f}'.format(cv_score_mean)

    df['Prediction'] = pd.DataFrame(model_pred)
    return diag_message, df


if __name__ == '__main__':
    def main():
        diag_line = 'model was not made'
        logfile_name = 'tune_NB_log.csv'
        diag_line, result_df = try_NB_rx_predict_model(logfile_name)
        print(diag_line)
        print(result_df)
        plt.figure()
        plt.plot(range(len(result_df['NodeB_RX'])), result_df['NodeB_RX'])
        plt.plot(range(len(result_df['NodeB_RX'])), result_df['Prediction'])
        plt.show()


    main()
