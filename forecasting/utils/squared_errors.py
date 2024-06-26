import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_errors(predictions, Y_test):
    mse = {}
    mae = {}
    rmse = {}
    for model, pred in predictions.items():
        Y_test = Y_test.squeeze()
        mse_pred = mean_squared_error(Y_test, pred)
        mae_pred = mean_absolute_error(Y_test, pred)

        mse[model] = mse_pred
        mae[model] = mae_pred
        rmse[model] = np.sqrt(mse_pred)

    return mse, mae, rmse