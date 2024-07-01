import ta
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from forecasting.data_preprocessing.get_filepaths import get_filepaths
import numpy as np

def load_csvs(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date'] + ['close']].dropna()
    df = df.iloc[::-1].reset_index(drop=True)
    df.set_index('date', inplace = True)
    df = df.sort_index()
    return df

def test_stationarity(feature_set):
    stationary_features = []
    non_stationary_features = []

    for column in feature_set.columns:
        # Perform Augmented Dickey-Fuller test
        if feature_set[column].min() != feature_set[column].max():
            result = adfuller(feature_set[column])
            p_value = result[1]
            if p_value <= 0.05:
                stationary_features.append(column)
            else:
                non_stationary_features.append(column)
    return non_stationary_features

def handle_non_stationarity(feature_set):
    # Test for stationarity
    non_stationary_features = test_stationarity(feature_set)
    differencing = 0
    # Take differences until all features are stationary
    while non_stationary_features:
        differencing = differencing + 1
        for feature in non_stationary_features:
            feature_set[feature] = feature_set[feature].diff().dropna()

        feature_set = feature_set.iloc[1:]

        # Test again for stationarity
        non_stationary_features = test_stationarity(feature_set)

    return feature_set, differencing

def obtainData(coin):
    print(coin)
    X_complete = []
    Y_complete = []

    #Get Data
    data_path, minute_data_paths = get_filepaths(coin)
    data = pd.read_csv(data_path)
    data.columns = [col.lower() for col in data.columns]
    data.columns = [col.capitalize() for col in data.columns]
    data['Date'] = pd.to_datetime(data['Date'])
    features = ['Open', 'High', 'Low', 'Close', 'Volume eur']
    data = data[['Date'] + features].dropna()
    data = data.iloc[::-1].reset_index(drop=True)
    data.set_index('Date', inplace = True)

    #Get Intraday Data
    intraday_data = load_csvs(minute_data_paths)
    intraday_data.columns = [col.lower() for col in intraday_data.columns]
    intraday_data.columns = [col.capitalize() for col in intraday_data.columns]

    # High-Low
    data['High Minus Low'] = data['High'] - data['Low']

    #Realized Volatility
    realized_volatility = calculate_daily_realized_volatility(intraday_data)
    data["Realized Volatility Lag 1"]= realized_volatility.shift(1)

    for lag in range(1,5):
      data[f'Realized Volatility Lag {lag + 1}'] = data["Realized Volatility Lag 1"].shift(lag)

    data['Mean Realized Volatility Last 5'] = realized_volatility.rolling(window=5).mean()
    data['Mean Realized Volatility'] = realized_volatility.expanding().mean()

    # Lagged Returns
    data['Return Lag 1'] = np.log(data['Close']/data['Close'].shift(1))*100

    for lag in range(1, 5):
      data[f'Return Lag {lag+1}'] = data['Return Lag 1'].shift(lag)*100

    # Moving Average
    data['MA 5'] = data['Close'].rolling(window=5).mean()

    #Correlation MA and Lagged Return
    data['MA Close Correlation'] = data['Return Lag 1'].rolling(window=5).corr(data['MA 5'])

    #Relative Strength Index
    data['RSI 6'] = ta.momentum.RSIIndicator(close=data['Close'], window=6).rsi()
    data['RSI 14'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['RSI > 80'] = ((data['RSI 6'] > 80) & (data['RSI 14'] > 80)).astype(int)
    data['RSI < 20'] = ((data['RSI 6'] < 20) & (data['RSI 14'] < 20)).astype(int)

    #EWMA
    data['EWMA'] = ta.trend.EMAIndicator(close=data['Close'], window=1 / (1 - 0.9)).ema_indicator()

    # Momentum
    data['Momentum'] = data['Close'].diff(5)

    # Calculate Aroon Up and Aroon Down
    data['Aroon Up'] = data['Close'].rolling(window=14).apply(lambda x: (14 - (x.argmax() + 1)) / 14 * 100)
    data['Aroon Down'] = data['Close'].rolling(window=14).apply(lambda x: (14 - (x.argmin() + 1)) / 14 * 100)

    # Calculate Aroon Stochastic Oscillator
    data['Aroon Oscillator'] = data['Aroon Up'] - data['Aroon Down']

    # Calculate Commodity Channel Index (CCI) with window = 14
    data['CCI'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14).cci()

    # Calculate Double Exponential Moving Average (DEMA) with window = 10
    data['DEMA'] = ta.trend.EMAIndicator(close=data['Close'], window=10).ema_indicator()

    # Rate of Change (ROC)
    data['ROC 9'] = ta.momentum.ROCIndicator(close=data['Close'], window=9).roc()
    data['ROC 14'] = ta.momentum.ROCIndicator(close=data['Close'], window=14).roc()

    # Average True Range (ATR)
    data['ATR 5'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=5).average_true_range()
    data['ATR 10'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=10).average_true_range()

    # Williams' %R
    data['Williams %R'] = ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'], lbp=14).williams_r()

    # MACD
    macd = ta.trend.MACD(close=data['Close'], window_slow=10, window_fast=5, window_sign=5)
    data['MACD'] = macd.macd()

    # Sum past returns
    data['Sum Past Returns 3'] = data['Return Lag 1'].rolling(window=3).sum()
    data['Sum Past Returns 5'] = data['Return Lag 1'].rolling(window=5).sum()

    # Difference past returns
    data['Diff Past Returns 3'] = data['Return Lag 1'] - data['Return Lag 1'].shift(3)
    data['Diff Past Returns 5'] = data['Return Lag 1'] - data['Return Lag 1'].shift(5)

    # Add Bollinger Bands
    data['BB High'] = ta.volatility.bollinger_hband(data['Close'], window=20, window_dev=2)
    data['BB Low'] = ta.volatility.bollinger_lband(data['Close'], window=20, window_dev=2)

    # Add On-Balance Volume (OBV)
    data['On-Balance Volume'] = ta.volume.on_balance_volume(data['Close'], data['Volume eur'])

    # Add Stochastic Oscillator
    data['Stochastic Oscillator'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)

    # Add Keltner Channels
    data['Keltner High'] = ta.volatility.keltner_channel_hband(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
    data['Keltner Low'] = ta.volatility.keltner_channel_lband(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
    data['Keltner Middle'] = ta.volatility.keltner_channel_mband(data['High'], data['Low'], data['Close'], window=20, window_atr=10)

    # Add Volume-Weighted Average Price (VWAP)
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume eur'], window=14)

    # Add Volatility Index (VIX) proxy using ATR
    data['VIX Proxy'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

    # Ichimoku Cloud
    ichimoku_cloud = ta.trend.IchimokuIndicator(high=data['High'], low=data['Low'])
    data['Tenkan-sen'] = ichimoku_cloud.ichimoku_a()
    data['Kijun-sen'] = ichimoku_cloud.ichimoku_b()
    data['Senkou Span A'] = ichimoku_cloud.ichimoku_a()
    data['Senkou Span B'] = ichimoku_cloud.ichimoku_b()

    # Volume-Price Trend (VPT)
    data['Volume-Price Trend'] = ta.volume.VolumePriceTrendIndicator(close=data['Close'], volume=data['Volume eur']).volume_price_trend()

    #Start data from 2021
    data = data[data.index >= "2021-07-01"]
    data = data[data.index <= "2024-06-01"]

    initial_features = ['Open', 'High', 'Low', 'Close', 'Volume eur']
    ridge_data = data.drop(columns=initial_features)

    stationary_data, differencing = handle_non_stationarity(ridge_data)
    ##Subtract 22 days for the HAR-RV model
    start_time = pd.to_datetime('2021-08-01') + dt.timedelta(days = differencing) - dt.timedelta(days =22)
    stationary_data = stationary_data[stationary_data.index >= start_time]
    data = data[data.index >= start_time]
    realized_volatility = realized_volatility[realized_volatility.index >= start_time]
    realized_volatility = realized_volatility[realized_volatility.index <= "2024-06-01"]


    X_complete = data
    X_ridge = stationary_data
    Y_complete = realized_volatility

    return X_complete, Y_complete, X_ridge

def calculate_daily_realized_volatility(intraday_data):
    realized_volatility = pd.DataFrame()
    df_hourly = intraday_data.copy()

    # Calculate the returns
    df_hourly['Return'] = np.log(df_hourly['Close'] / df_hourly['Close'].shift(1))
    df_hourly['Squared Return'] = df_hourly['Return'] ** 2

    # Calculate daily realized volatility
    realized_volatility["Realized Volatility"] = df_hourly.groupby(df_hourly.index.date)['Squared Return'].sum().apply(np.sqrt) * 100
    realized_volatility.index = pd.to_datetime(realized_volatility.index)

    return realized_volatility

def plot_volatility(Y_complete):
    plt.figure(figsize=(14, 7))
    plt.plot(Y_complete.index, Y_complete.values, label='Realized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Realized Volatility')
    plt.title('Daily Realized Volatility')
    plt.legend()
    plt.show()