import numpy as np
import pandas as pd
import yfinance as yf
import itertools
import matplotlib.pyplot as plt

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from loguru import logger
def train_test_forecast(stock, experiment, forecast_days):
    
    # Download stock data
    ticker = f"{stock}.JK"        
    stock_data = yf.download(ticker, start='2006-01-01', end='2024-06-15')
    stock_data.reset_index(inplace=True)

    # Check if DataFrame is empty
    if stock_data.empty:
        raise ValueError(f"'{stock}' historical data might not be available in yahoo Finance. Please recheck.")
    # Check if DataFrame contains any NaN values
    if stock_data.isna().sum().sum() > 0:
        raise ValueError("DataFrame contains NaN values")

    # Ensure the date column is in datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Define the window size for volatility calculation
    window_size = 21  # You can change this to any other number if needed

    # Calculate additional features
    stock_data['Volatility'] = stock_data['Adj Close'].pct_change().rolling(window=window_size).std() * np.sqrt(252)
    stock_data['SMA_50'] = stock_data['Adj Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    stock_data['SMA_200'] = stock_data['Adj Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
    stock_data['Momentum'] = stock_data['Adj Close'] - stock_data['Adj Close'].shift(10)  # 10-day Momentum

    # Add more technical indicators
    stock_data = add_technical_indicators(stock_data)

    # Prepare the data for Prophet with additional features
    prophet_data = stock_data[['Date', 'Adj Close', 'Volatility', 'SMA_50', 'SMA_200', 'Momentum', 'EMA_12', 'EMA_26', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal', '%K', '%D', 'OBV', 'ADL', 'Log_Returns', 'Skewness', 'Kurtosis']].dropna()
    prophet_data.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

    prophet_data.tail()

    if experiment:
        # Split data into training and test sets
        train_data, test_data = train_test_split(prophet_data, test_size=0.2, shuffle=False, stratify=None)

    else:
        train_data = prophet_data

    logger.info(f"Training raw model")
    # Initialize and fit the model without additional features
    model_no_features = Prophet()
    model_no_features.fit(train_data[['ds', 'y']])
    
    logger.info(f"Training Featured model")
    # Initialize and fit the model with additional features
    model_with_features = Prophet()
    model_with_features.add_regressor('Volatility')
    model_with_features.add_regressor('SMA_50')
    model_with_features.add_regressor('SMA_200')
    model_with_features.add_regressor('Momentum')
    model_with_features.add_regressor('Log_Returns')
    model_with_features.add_regressor('Skewness')
    model_with_features.add_regressor('Kurtosis')
    model_with_features.fit(train_data)

    # Hyperparameter tuning
    params_grid = {
    'seasonality_mode': ['multiplicative', 'additive'],
    'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
    'holidays_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
    'n_changepoints': [100, 150, 200]
}

    all_params = [dict(zip(params_grid.keys(), v)) for v in itertools.product(*params_grid.values())]
    best_params = None
    best_mae = float('inf')

    for index, params in enumerate(all_params):
        model = Prophet(**params)
        model.add_regressor('Volatility')
        model.add_regressor('SMA_50')
        model.add_regressor('SMA_200')
        model.add_regressor('Momentum')
        model.add_regressor('Log_Returns')
        model.add_regressor('Skewness')
        model.add_regressor('Kurtosis')
        model.fit(train_data)
        
        future = model.make_future_dataframe(periods=forecast_days, freq='B')
        future = future.merge(stock_data[['Date', 'Volatility', 'SMA_50', 'SMA_200', 'Momentum', 'EMA_12', 'EMA_26', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal', '%K', '%D', 'OBV', 'ADL', 'Log_Returns', 'Skewness', 'Kurtosis']], left_on='ds', right_on='Date', how='left')
        future.drop(columns=['Date'], inplace=True)
        future.fillna(method='ffill', inplace=True)
        
        forecast = model.predict(future)
        
        if experiment:
            test_data_aligned = test_data[test_data['ds'].isin(forecast['ds'])]
            mae = mean_absolute_error(test_data_aligned['y'], forecast.loc[forecast['ds'].isin(test_data_aligned['ds']), 'yhat'])
            if mae < best_mae:
                best_mae = mae
                best_params = params

    print(f"Best MAE: {best_mae} with parameters: {best_params}")

    # Initialize and fit the final model with best parameters
    model_with_features_optimized = Prophet(**best_params)
    model_with_features_optimized.add_regressor('Volatility')
    model_with_features_optimized.add_regressor('SMA_50')
    model_with_features_optimized.add_regressor('SMA_200')
    model_with_features_optimized.add_regressor('Momentum')
    model_with_features_optimized.add_regressor('Log_Returns')
    model_with_features_optimized.add_regressor('Skewness')
    model_with_features_optimized.add_regressor('Kurtosis')
    model_with_features_optimized.fit(train_data)

    # Make predictions
    forecast_with_features = model_with_features.predict(future_with_features)

    # Specify the number of additional days to forecast
    additional_days = forecast_days  # Example: 700 days into the future
    windows = -((additional_days * 2))

    if experiment:
        # Create a future dataframe for the entire dataset and the forecast period
        future_no_features = pd.DataFrame(test_data['ds'])
        futures_with_volatility = pd.DataFrame(test_data['ds'])
        future_with_features = pd.DataFrame(test_data['ds'])
        

    else:
        # Create a future dataframe for the entire dataset and the forecast period
        future_no_features = model_no_features.make_future_dataframe(periods=additional_days, freq='B').iloc[windows:]
        future_with_features = model_with_features.make_future_dataframe(periods=additional_days, freq='B').iloc[windows:]
        future_with_features_optimized = model_with_features_optimized.make_future_dataframe(periods=additional_days, freq='B').iloc[windows:]

    # Make predictions for the entire dataset including future dates
    forecast_no_features = model_no_features.predict(future_no_features)
    forecast_with_features = model_with_features.predict(future_with_features)
    forecast_with_features_optimized = model_with_features_optimized.predict(future_with_features_optimized)

    # Align the future dataframe with stock_data and fill in additional features for the model with features
    future_with_features = future_with_features.merge(stock_data[['Date', 'Volatility', 'SMA_50', 'SMA_200', 'Momentum', 'EMA_12', 'EMA_26', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal', '%K', '%D', 'OBV', 'ADL', 'Log_Returns', 'Skewness', 'Kurtosis']], left_on='ds', right_on='Date', how='left')
    future_with_features.drop(columns=['Date'], inplace=True)
    future_with_features.fillna(method='ffill', inplace=True)

    # Create an optimized future dataframe for the forecast period
    future_with_features_optimized = future_with_features_optimized.merge(stock_data[['Date', 'Volatility', 'SMA_50', 'SMA_200', 'Momentum', 'EMA_12', 'EMA_26', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD', 'Signal', '%K', '%D', 'OBV', 'ADL', 'Log_Returns', 'Skewness', 'Kurtosis']], left_on='ds', right_on='Date', how='left')
    future_with_features_optimized.drop(columns=['Date'], inplace=True)
    future_with_features_optimized.fillna(method='ffill', inplace=True)
    
    if experiment:
        # Evaluate performance on the test set
        test_data_aligned = test_data[test_data['ds'].isin(forecast_no_features['ds'])]
        mae_no_features = mean_absolute_error(test_data_aligned['y'], forecast_no_features.loc[forecast_no_features['ds'].isin(test_data_aligned['ds']), 'yhat'])
        mse_no_features = mean_squared_error(test_data_aligned['y'], forecast_no_features.loc[forecast_no_features['ds'].isin(test_data_aligned['ds']), 'yhat'])

        test_data_aligned_with_features = test_data[test_data['ds'].isin(forecast_with_features['ds'])]
        mae_with_features = mean_absolute_error(test_data_aligned_with_features['y'], forecast_with_features.loc[forecast_with_features['ds'].isin(test_data_aligned_with_features['ds']), 'yhat'])
        mse_with_features = mean_squared_error(test_data_aligned_with_features['y'], forecast_with_features.loc[forecast_with_features['ds'].isin(test_data_aligned_with_features['ds']), 'yhat'])

        test_data_aligned_with_features_optimized = test_data[test_data['ds'].isin(forecast_with_features_optimized['ds'])]
        mae_with_features_optimized = mean_absolute_error(test_data_aligned_with_features_optimized['y'], forecast_with_features_optimized.loc[forecast_with_features_optimized['ds'].isin(test_data_aligned_with_features_optimized['ds']), 'yhat'])
        mse_with_features_optimized = mean_squared_error(test_data_aligned_with_features_optimized['y'], forecast_with_features_optimized.loc[forecast_with_features_optimized['ds'].isin(test_data_aligned_with_features_optimized['ds']), 'yhat'])

        print(f"Model without Additional Features - MAE: {mae_no_features}, MSE: {mse_no_features}")
        print(f"Model with Additional Features - MAE: {mae_with_features}, MSE: {mse_with_features}")
        print(f"Model with Optimized Additional Features - MAE: {mae_with_features_optimized}, MSE: {mse_with_features_optimized}")
        

    # Plotting the entire time series with forecast including future dates
    plt.figure(figsize=(14, 7))

    if experiment:
        # Plot actual prices from the start
        plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual', color='blue')

    else:
        # Plot actual prices from the start
        plt.plot(prophet_data['ds'][windows:], prophet_data['y'][windows:], label='Actual', color='blue')
        
    # Plot forecast without additional features
    plt.plot(forecast_no_features['ds'], forecast_no_features['yhat'], label='Forecast without Features', color='orange')
    # Plot forecast with additional features
    plt.plot(forecast_with_features['ds'], forecast_with_features['yhat'], label='Forecast with Volatility + Statistical Features', color='green')
    # Plot forecast with Optimized features
    plt.plot(forecast_with_features_optimized['ds'], forecast_with_features_optimized['yhat'], label='Forecast with Optimized Model', color='red')
    
    # Fill between forecast with features
    plt.fill_between(forecast_with_features_optimized['ds'], forecast_with_features_optimized['yhat_lower'], forecast_with_features_optimized['yhat_upper'], color='gray', alpha=0.2)
    plt.title(f'{stock} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    if not experiment:
        plt.xlim(prophet_data['ds'][windows:].min(), future_no_features['ds'][windows:].max())
    else:
        plt.xlim(prophet_data['ds'][:].min(), future_no_features['ds'][:].max())
    plt.show()

    return forecast_no_features, forecast_with_features, forecast_with_features_optimized