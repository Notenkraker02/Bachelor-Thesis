import numpy as np
import pandas as pd
from forecasting.hypertuning.hypertuning import hypertune_model
from forecasting.Models.local_linear_forest import LocalLinearForestRegressor
from forecasting.Models.random_forest import train_test_rf
from forecasting.Models.GARCH import predict_GARCH
from forecasting.Models.GJR_GARCH import predict_GJR
from forecasting.Models.HAR_RV import predict_har

def in_sample_forecast(X_complete, Y_complete, X_ridge):
    predictions = {}
    LLF_pred = []
    RF_pred = []
    GJR_pred = []
    GARCH_pred = []

    #Get Dates
    har_Y = Y_complete
    Y_complete = Y_complete.iloc[21:]
    Y_complete_dates = Y_complete.index
    X_ridge = X_ridge.iloc[21:]
    X_complete = X_complete.iloc[20:]
    
    #Get returns
    returns = np.log(X_complete['Close'] / X_complete['Close'].shift(1)).dropna()
    returns = returns * 100

    X_complete = X_complete.iloc[1:]

    #Set to correct format
    X_complete = X_complete.to_numpy()
    X_ridge = X_ridge.to_numpy()
    Y_complete = Y_complete.to_numpy().ravel()

    # Local Linear Forest
    LLF_parameters = hypertune_model("LocalLinearForest", X_complete, Y_complete, X_ridge, n_trials = 100)
    LLF = LocalLinearForestRegressor(**LLF_parameters)
    LLF.fit(X_complete, Y_complete, X_ridge)
    LLF_pred = LLF.predict_LLF(X_complete, X_ridge)
    predictions['LLF'] = pd.Series(LLF_pred, index=Y_complete_dates)

    # Random Forest
    RF_parameters = hypertune_model("RandomForest", X_complete, Y_complete, n_trials = 100)    
    RF_pred = train_test_rf(X_complete, Y_complete, X_complete, **RF_parameters)
    predictions['RF'] = pd.Series(RF_pred, index=Y_complete_dates)

    # GARCH(1,1)
    GARCH_pred = predict_GARCH(returns, False)
    predictions['GARCH'] = pd.Series(GARCH_pred, index=Y_complete_dates)

    # GJR-GARCH
    GJR_pred = predict_GJR(returns, False)
    predictions['GJR'] = pd.Series(GJR_pred, index=Y_complete_dates)

    #HAR-RV
    har_pred = predict_har(har_Y)
    predictions['HAR-RV'] = pd.Series(har_pred, index=Y_complete_dates)

    return predictions, Y_complete
