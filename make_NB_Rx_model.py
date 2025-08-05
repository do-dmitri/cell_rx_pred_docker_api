# ------------------------------------------------------------------
# NodeB mobile RX prediction based on
# 'RSRP', 'RSSI', 'UE_TX_Power'
#
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

pd.set_option('display.width', None) # disable scroll


def make_NB_rx_predict_model(logname) -> tuple:
    """
    Creates UL Rx prediction model
    :param logname: *.csv logfile name with 'RSRP', 'RSSI', 'UE_TX_Power', 'NodeB_RX'
    :return: diagnostic message, prediction model
    """
    diag_message = ''
    df = pd.read_csv(logname, sep=',')
    df = df[['RSRP', 'RSSI', 'UE_TX_Power', 'NodeB_RX']]
    print(df)
    X = np.array(df.drop('NodeB_RX', axis=1).copy())
    y = np.array(df['NodeB_RX'].copy())
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



    # ------------------------------------------------------------------
    # Linear Regression with Cross Validation check
    # ------------------------------------------------------------------
    diag_message += '\n\nLinear Regression ---------------------------------------------'

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_model_pred = lr_model.predict(X_test)

    diag_message += '\nScore: {:.4f}'.format(lr_model.score(X_test, y_test))
    diag_message += '\nR2   : {:.4f}'.format(r2_score(y_test, lr_model_pred))
    diag_message += '\nMSE  : {:.4f}'.format(mean_squared_error(y_test, lr_model_pred))

    diag_message += f'\ncoef : {lr_model.coef_}, {lr_model.intercept_}'

    cv_score_mean = cross_val_score(LinearRegression(), X, y, cv=3, scoring='r2').mean()
    diag_message += '\nCV mean: {:.4f}'.format(cv_score_mean)



    # ------------------------------------------------------------------
    # Decision Tree Regression + CV
    # ------------------------------------------------------------------
    diag_message += '\n\nDT Regression -------------------------------------------------'

    dtr_model = DecisionTreeRegressor(max_depth=3)
    dtr_model.fit(X_train, y_train)
    dtr_model_pred = dtr_model.predict(X_test)

    diag_message += '\nScore: {:.4f}'.format(dtr_model.score(X_test, y_test))
    diag_message += '\nR2   : {:.4f}'.format(r2_score(y_test, dtr_model_pred))
    diag_message += '\nMSE  : {:.4f}'.format(mean_squared_error(y_test, dtr_model_pred))

    cv_score_mean = cross_val_score(DecisionTreeRegressor(max_depth=3), X, y, cv=3, scoring='r2').mean()
    diag_message += '\nCV R2 mean: {:.4f}'.format(cv_score_mean)



    # ------------------------------------------------------------------
    # Decision Tree GridSearch
    # ------------------------------------------------------------------
    diag_message += '\n\nGridSearchCV DTR-------------------------------------------------'
    gs_params = {'max_depth': range(2, 20),
                 'criterion': ['squared_error']}
    gs_model = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=gs_params, cv=3, scoring='r2', n_jobs=4)
    gs_model.fit(X_train, y_train)
    diag_message += f'\nbest estimator : {gs_model.best_estimator_}'
    diag_message += f'\nbest params : {gs_model.best_params_}'  # {'criterion': 'squared_error', 'max_depth': 6}
    diag_message += '\nBest score R2 : {:.4f}'.format(gs_model.best_score_)  # Best R2 0.8319

    dtr1_model = DecisionTreeRegressor(criterion='squared_error', max_depth=6, random_state=42)
    dtr1_model.fit(X_train, y_train)
    diag_message += '\nDTR R2 with X_test: {:.4f}'.format(r2_score(y_test, dtr1_model.predict(X_test)))



    # ------------------------------------------------------------------
    # Random Forest Regressor GridSearch
    # ------------------------------------------------------------------
    diag_message += '\n\nGridSearchCV RFR-------------------------------------------------'
    rfr_gs_params = {'n_estimators': range(2, 20, 3),
                     'max_depth': range(2, 20, 3),
                     'max_features': ['log2', 'sqrt']}

    rfr_gs_model = GridSearchCV(RandomForestRegressor(random_state=42), rfr_gs_params, cv=3, scoring='r2', n_jobs=4)
    rfr_gs_model.fit(X_train, y_train)

    diag_message += f'\nbest estimator : {rfr_gs_model.best_estimator_}'
    diag_message += f'\nbest params : {rfr_gs_model.best_params_}'  # {'max_depth': 5, 'max_features': 'log2', 'n_estimators': 17}
    diag_message += '\nBest score R2 : {:.4f}'.format(rfr_gs_model.best_score_)

    rfc_model = RandomForestRegressor(max_depth=8, max_features='log2', n_estimators=117, random_state=42)
    rfc_model.fit(X_train, y_train)
    diag_message += '\nRFR R2 with X_test: {:.4f}'.format(r2_score(y_test, rfc_model.predict(X_test)))



    # ------------------------------------------------------------------
    # Gradient Boosting GridSearch
    # ------------------------------------------------------------------
    diag_message += '\n\nGridSearchCV GBR-------------------------------------------------'
    gbr_gs_params = {'n_estimators': range(2, 20, 3),
                     'max_depth': range(2, 20, 3),
                     'max_features': ['log2', 'sqrt']}

    gbr_gs_model = GridSearchCV(GradientBoostingRegressor(random_state=42), gbr_gs_params, cv=3, scoring='r2', n_jobs=4)
    gbr_gs_model.fit(X_train, y_train)

    diag_message += f'\nbest estimator: {gbr_gs_model.best_estimator_}'
    diag_message += '\nBest score R2 : {:.4f}'.format(gbr_gs_model.best_score_)  # Best R2: 0.8389

    # learning_rate 100 less, x100 n_estimators. Improved performance:
    gbr_model = GradientBoostingRegressor(max_depth=5, max_features='log2', learning_rate=0.01, n_estimators=1700,
                                          random_state=42)
    gbr_model.fit(X_train, y_train)
    diag_message += '\nx100 n_est GBR R2 : {:.4f}'.format(r2_score(y_test, gbr_model.predict(X_test)))

    importances = gbr_model.feature_importances_
    features = ['RSRP', 'RSSI', 'UE_TX_Power']
    plt.bar(features, importances, color='orange')
    plt.title('Feature Importance')
    plt.show()

    return diag_message, lr_model


if __name__ == '__main__':
    def main():
        diag_line = 'model was not made'
        logfile_name = 'tune_NB_log.csv'
        diag_line, cellrx_model = make_NB_rx_predict_model(logfile_name)
        joblib.dump(cellrx_model, 'app/cell_RX_model.joblib')
        print(diag_line)



    main()
