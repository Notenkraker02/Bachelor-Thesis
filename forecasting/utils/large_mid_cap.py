import pandas as pd
import numpy as np
from scipy.stats import skew

def normalized_errors(predictions, Y_test):
    Y_test = Y_test.squeeze()
    return (np.log(predictions / Y_test)) ** 2

def initialize_errors():
    return {model: [] for model in ["LLF", "RF", "GJR", "GARCH", "HAR-RV"]}

def collect_errors(predictions, Y_test, coin, large_cap_coins, mid_cap_coins, large_cap_errors, mid_cap_errors):
    for model, model_predictions in predictions.items():
        norm_errors = normalized_errors(model_predictions, Y_test)
        if coin in large_cap_coins:
            large_cap_errors[model].append(np.array(norm_errors))  
        elif coin in mid_cap_coins:
            mid_cap_errors[model].append(np.array(norm_errors))    
    return large_cap_errors, mid_cap_errors

def compute_statistics(errors):
    statistics = {}
    for model, error_list in errors.items():
        error_list = np.concatenate(error_list)
        error_list = np.array(error_list)  # Ensure error_list is a NumPy array
        statistics[model] = {
            'RMSLE': np.sqrt(error_list.mean()),
            'std': error_list.std(),
            'skew': skew(error_list)
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
