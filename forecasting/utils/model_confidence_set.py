import numpy as np
import pandas as pd
from model_confidence_set import ModelConfidenceSet

def get_mcs(predictions, Y_test, error):
    errors = {}
    for model, pred in predictions.items():
        Y_test = Y_test.squeeze()
        if error == "RMSE":
            errors[model] = (Y_test - pred)  ** 2
        elif error =="MAE":
            errors[model] = (np.abs(Y_test - pred))
        elif error == "QLIKE":
            errors[model] = (Y_test/pred) - np.log(Y_test/pred) - 1
        elif error == "Utility":
            errors[model] = - (0.08 * (np.sqrt(Y_test/pred)) - 0.04 * (Y_test/pred))
    errors = pd.DataFrame(errors)

    ## Use block_len = 7 from the rule of thumb (cube root of number of observations)
    mcs = ModelConfidenceSet(errors, n_boot = 5000, alpha = 0.05, block_len= 7)
    mcs.compute()
    mcs_results = mcs.results()
    included_models = {model: (mcs_results.loc[model, 'status'] == 'included') for model in predictions.keys()}
    return included_models

def update_mcs_count(predictions, Y_test, mcs_counts_rmse=None, mcs_counts_mae = None, mcs_counts_qlike=None, mcs_counts_utility=None):
    included_models_rmse = get_mcs(predictions, Y_test, "RMSE")
    included_models_mae = get_mcs(predictions, Y_test, "MAE")
    included_models_qlike = get_mcs(predictions, Y_test, 'QLIKE')
    included_models_utility = get_mcs(predictions, Y_test, "Utility")

    for model in predictions.keys():
        if included_models_rmse[model] and mcs_counts_rmse is not None:
            mcs_counts_rmse[model] += 1
        if included_models_mae[model] and mcs_counts_mae is not None:
            mcs_counts_mae[model] += 1            
        if included_models_qlike[model] and mcs_counts_qlike is not None:
            mcs_counts_qlike[model] += 1
        if included_models_utility[model] and mcs_counts_utility is not None:
            mcs_counts_utility[model] += 1

    return mcs_counts_rmse, mcs_counts_mae, mcs_counts_qlike, mcs_counts_utility