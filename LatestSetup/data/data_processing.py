import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.data_fetching import DataFetcher
from data.add_features import add_features
from utils.utils import check_data_integrity, clean_dataframe, ensure_adj_close

def fetch_and_prepare_data(config):
    data_fetcher = DataFetcher()
    historical_data = data_fetcher.fetch_data(ticker=config['ticker'], start_date=config['start_date'], end_date=config['end_date'])

    # Ensure 'Adj Close' exists
    historical_data = ensure_adj_close(historical_data)

    # Check and preprocess data
    historical_data = check_data_integrity(historical_data)

    # Clean the dataframe
    historical_data = clean_dataframe(historical_data)

    # Add features to the data
    historical_data = add_features(historical_data)

    # Normalize the data except for the 'Close' column
    columns_to_normalize = [col for col in historical_data.columns if col != 'Close']
    scaler = MinMaxScaler()
    historical_data[columns_to_normalize] = scaler.fit_transform(historical_data[columns_to_normalize])

    expected_features = historical_data.columns.tolist()
    
    return historical_data, scaler, expected_features
