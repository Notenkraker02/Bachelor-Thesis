from sklearn.linear_model import LinearRegression
import pandas as pd

def predict_har(realized_volatility):
    realized_volatility = pd.DataFrame(realized_volatility, columns=['Realized Volatility'])
    rv = realized_volatility['Realized Volatility']
    df = pd.DataFrame({
        'Constant': 1,
        'RV': rv,
        'RV_day': rv.shift(1),
        'RV_weekly': rv.rolling(window=5).mean(),
        'RV_monthly': rv.rolling(window=22).mean()
    }).dropna()

    X = df[['Constant', 'RV_day', 'RV_weekly', 'RV_monthly']]
    y = df['RV']

    model = LinearRegression()
    model.fit(X, y)

    # Prepare the most recent values for prediction
    latest_Y_lagged = rv.iloc[-1]
    latest_Y_weekly = rv.rolling(window=5).mean().iloc[-1]
    latest_Y_monthly = rv.rolling(window=22).mean().iloc[-1]

    # Create a DataFrame for the most recent values
    latest_X = pd.DataFrame({
        'Constant': [1],
        'RV_day': [latest_Y_lagged],
        'RV_weekly': [latest_Y_weekly],
        'RV_monthly': [latest_Y_monthly]
    })

    # One-step-ahead prediction
    one_step_ahead_prediction = model.predict(latest_X)
    return one_step_ahead_prediction[0]