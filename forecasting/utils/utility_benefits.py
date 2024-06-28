import numpy as np

def expected_utility(forecast, realized_volatility):
  utility = {}
  for model, pred in forecast.items():
    realized_volatility = realized_volatility.squeeze()
    utility[model] = 100* (np.mean(0.08 * (np.sqrt(realized_volatility/pred)) - 0.04 * (realized_volatility/pred)))
  return utility

def calculate_weight(sharpe, gamma, forecast):
    x_t = (sharpe/gamma) /np.sqrt(forecast)
    return x_t