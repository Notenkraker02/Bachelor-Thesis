import pandas as pd
import numpy as np
from scipy.stats import f_oneway, skew, kruskal
import scikit_posthocs as sp
from IPython.utils import io

def define_market_phases(prices, window=30, threshold=0.05):
    moving_avg = prices.rolling(window).mean()
    change = (prices - moving_avg) / moving_avg
    market_phases = np.where(change > threshold, 'bull', np.where(change < -threshold, 'bear', 'consolidating'))
    return pd.Series(market_phases, index=prices.index)

def segment_data_by_phases(errors, market_phases):
    phases_data = {model: {'bull': [], 'bear': [], 'consolidating': []} for model in errors.keys()}
    phase_counts = {'bull': 0, 'bear': 0, 'consolidating': 0}

    for phase in ['bull', 'bear', 'consolidating']:
        phase_indices = market_phases[market_phases == phase].index
        phase_counts[phase] = len(phase_indices)
        for model, model_errors in errors.items():
            phase_indices_model = phase_indices.intersection(errors[model].index)
            phase_data = model_errors.loc[phase_indices_model]
            phases_data[model][phase] = phase_data.tolist()

    return phases_data, phase_counts

def perform_kruskal(errors_by_phase):
    results = {}
    for model, phase_errors in errors_by_phase.items():
        stat, p_value = kruskal(phase_errors['bull'], phase_errors['bear'], phase_errors['consolidating'])
        results[model] = {'stat': stat, 'p_value': p_value}
    return results

def perform_dunn_test(errors_by_phase):
    results = {}
    for model, phase_errors in errors_by_phase.items():
        dunn = sp.posthoc_dunn([phase_errors['bull'], phase_errors['bear'], phase_errors['consolidating']], p_adjust='bonferroni')
        results[model] = dunn
    return results

def cycle_errors(predictions, Y_test):
    errors = {}
    for model, pred in predictions.items():
        Y_test = Y_test.squeeze()
        errors[model] = (pred - Y_test) ** 2
    return errors

