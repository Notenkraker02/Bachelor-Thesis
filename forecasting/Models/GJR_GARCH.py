from arch import arch_model
import numpy as np

def predict_GJR(Y_train, oos):
    # Specify GJR-GARCH(1,1) model
    model = arch_model(Y_train, vol='GARCH', p=1, o=1, q=1)

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
      print("GJR estimates:", model_fit.params)

      return in_sample_fitted_values