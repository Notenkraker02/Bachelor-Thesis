import numpy as np

def utility(forecast, realized_volatility):
  realized_volatility = realized_volatility.squeeze()
  return 0.08 * (np.sqrt(realized_volatility/forecast)) - 0.04 * (realized_volatility/forecast)

def calculate_weight(sharpe, gamma, forecast):
    x_t = (sharpe/gamma) /np.sqrt(forecast)
    return x_t

def expected_utility(forecasts, realized_volatilities):
    total_utility = 0
    for date, forecast in forecasts.items():
        realized_volatility = realized_volatilities.loc[date].iloc[0]

        utility_i = utility(forecast, realized_volatility)
        total_utility += utility_i

    expected_utility = total_utility / len(forecasts)

    return expected_utility