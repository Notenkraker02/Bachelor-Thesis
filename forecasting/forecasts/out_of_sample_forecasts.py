#%pip install model-confidence-set
import numpy as np
import pandas as pd
from tqdm import tqdm, notebook
from forecasting.hypertuning.hypertuning import hypertune_model
from forecasting.Models.local_linear_forest import LocalLinearForestRegressor
from forecasting.Models.random_forest import train_test_rf
from forecasting.Models.GARCH import predict_GARCH
from forecasting.Models.GJR_GARCH import predict_GJR
from forecasting.Models.HAR_RV import predict_har
from forecasting.utils.feature_importance import plot_feature_importance

def forecast(coin, X_complete, Y_complete, X_ridge, initial_train_size, feature = True):
    predictions = {}
    LLF_pred = []
    RF_pred = []
    GJR_pred = []
    GARCH_pred = []
    HAR_pred = []

    #Get Dates
    har_Y = Y_complete
    Y_complete = Y_complete.iloc[21:]
    X_ridge = X_ridge.iloc[21:]
    X_complete = X_complete.iloc[21:]

    train_size = int(len(X_complete) * initial_train_size)
    test_size = len(X_complete) - train_size
    Y_dates = Y_complete.iloc[train_size:].index
    feature_names =  X_complete.columns
    feature_importance = np.zeros(len(feature_names))

    #Hyperparameter Tuning
    X_tune = X_complete.iloc[:train_size].to_numpy()
    Y_tune = Y_complete.iloc[:train_size].to_numpy().ravel()
    X_ridge_tune = X_ridge.iloc[:train_size].to_numpy()
    LLF_parameters = hypertune_model("LocalLinearForest", X_tune, Y_tune, X_ridge_tune, n_trials = 50)
    RF_parameters = hypertune_model("RandomForest", X_tune, Y_tune, n_trials = 50)

    for i in notebook.tqdm(range(test_size)):
        X_train_split = X_complete.iloc[:train_size + i]
        X_train_ridge = X_ridge.iloc[:train_size + i]
        returns = np.log(X_train_split['Close'] / X_train_split['Close'].shift(1)).dropna()
        returns = returns * 100
        X_train_split = X_train_split.to_numpy()
        X_train_ridge = X_train_ridge.to_numpy()
        X_test_split = X_complete.iloc[train_size + i : train_size + i + 1].to_numpy()
        X_test_ridge = X_ridge.iloc[train_size + i : train_size + i + 1].to_numpy()
        Y_train = Y_complete.iloc[:train_size + i].to_numpy().ravel()
        Y_test =  Y_complete.iloc[train_size + i : train_size + i + 1].to_numpy().ravel()
        
        # Local Linear Forest
        LLF = LocalLinearForestRegressor(**LLF_parameters)
        LLF.fit(X_train_split, Y_train, X_train_ridge)
        LLF_pred.append(max(LLF.predict_LLF(X_test_split, X_test_ridge)[0], 0.1))
        feature_importance = feature_importance + LLF.feature_importances_

        # Random Forest
        RF_pred.append(train_test_rf(X_train_split, Y_train, X_test_split, **RF_parameters)[0])

        # GJR-GARCH
        GJR_pred.append(predict_GJR(returns, True))

        #GARCH(1,1)
        GARCH_pred.append(predict_GARCH(returns, True))

        # HAR-RV
        har_data = har_Y.iloc[:train_size + i + 21]
        HAR_pred.append(predict_har(har_data))

    if feature: 
       feature_importance = feature_importance / test_size
       plot_feature_importance(feature_importance, feature_names, coin, threshold = 0.005)

    predictions['LLF'] = pd.Series(LLF_pred, index=Y_dates)
    predictions['RF'] = pd.Series(RF_pred, index=Y_dates)
    predictions['GARCH'] = pd.Series(GARCH_pred, index=Y_dates)
    predictions['GJR'] = pd.Series(GJR_pred, index=Y_dates)
    predictions['HAR-RV'] = pd.Series(HAR_pred, index = Y_dates)
    Y_test = Y_complete.iloc[train_size:]

    return predictions, Y_test