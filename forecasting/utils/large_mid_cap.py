import pandas as pd
import numpy as np
from scipy.stats import skew

def normalized_errors(predictions, Y_test):
    Y_test = Y_test.squeeze()

    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.iloc[:, 0]
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.iloc[:, 0]
    return (np.log(predictions / Y_test)) ** 2

def initialize_errors():
    return {model: [] for model in ["LLF", "RF", "GJR", "GARCH", "HAR-RV"]}

def collect_errors(predictions, Y_test, coin, large_cap_coins, mid_cap_coins, large_cap_errors, mid_cap_errors):
    for model, model_predictions in predictions.items():
        norm_errors = normalized_errors(model_predictions, Y_test)
        if coin in large_cap_coins:
            large_cap_errors[model].append(norm_errors)
        elif coin in mid_cap_coins:
            mid_cap_errors[model].append(norm_errors)

def compute_statistics(errors):
    statistics = {}
    for model, error_list in errors.items():
        combined_errors = pd.concat(error_list, axis=0, ignore_index=True)
        statistics[model] = {
            'RMSLE': np.sqrt(combined_errors.mean()),
            'std': combined_errors.std(),
            'skew': skew(combined_errors.dropna())
        }
    return statistics

def get_grouped_errors(large_cap_coins, mid_cap_coins, coins, predictions, Y_test):
    large_cap_errors = initialize_errors()
    mid_cap_errors = initialize_errors()

    for coin in coins:
        collect_errors(predictions, Y_test, coin, large_cap_coins, mid_cap_coins, large_cap_errors, mid_cap_errors)

    large_cap_statistics = compute_statistics(large_cap_errors)
    mid_cap_statistics = compute_statistics(mid_cap_errors)

    return large_cap_statistics, mid_cap_statistics
