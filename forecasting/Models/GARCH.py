from arch import arch_model
import numpy as np

def predict_GARCH(returns, oos):
    # Create GARCH(1,1) model
    model = arch_model(returns, vol='Garch', p=1, q=1)

    # Fit the model
    model_fit = model.fit(disp='off')

    if(oos):
      # Make one-step-ahead forecast
      forecast = model_fit.forecast(horizon=1)
      # Get the forecasted conditional volatility
      forecasted_volatility = np.sqrt(forecast.variance.values[-1][-1])

      return forecasted_volatility

    else:
      in_sample_fitted_values = np.sqrt(model_fit.conditional_volatility)
      print("GARCH estimates:", model_fit.params)

      return in_sample_fitted_values