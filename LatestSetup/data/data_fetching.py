import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import ccxt
import json
import time
from coinbase.websocket import WSClient  # Ensure this is the correct import
import yfinance as yf
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        pass

    def fetch_data(self, ticker="BTC-USD", start_date=None, end_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # 5 years ago
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        if data.isna().sum().sum() > 0:
            print("Data contains NaN values. Filling NaNs with 0.")
            data.fillna(0, inplace=True)
        return data

    def preprocess_data(self, data):
        data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data.dropna(inplace=True)

        majority_class = data[data['target'] == 0]
        minority_class = data[data['target'] == 1]
        minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        data_balanced = pd.concat([majority_class, minority_class_upsampled])

        features = data_balanced.drop(columns=['Date', 'target'])
        target = data_balanced['target']
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features, target

    def fetch_historical_data_coinbase(self, symbol, timeframe, retries=5):
        exchange = ccxt.coinbase()
        for i in range(retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
                init_df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                init_df['timestamp'] = pd.to_datetime(init_df['timestamp'], unit='ms')
                return init_df
            except ccxt.RateLimitExceeded:
                if i < retries - 1:
                    wait_time = 2 ** i
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

    def on_message(self, msg):
        print(json.dumps(msg, indent=2))

    class MyWebsocketClient(WSClient):
        def __init__(self, api_key, api_secret, on_message=None, channels=["ticker"], products=["BTC-USD"], **kwargs):
            self.api_key = api_key
            self.api_secret = api_secret
            self.on_message = on_message if on_message else self.default_on_message
            self.channels = channels
            self.products = products
            super().__init__(**kwargs)

        def on_open(self):
            print("WebSocket connection opened.")
            self.subscribe({"name": "ticker", "product_ids": self.products})

        def on_message(self, msg):
            if self.on_message:
                self.on_message(msg)

        def on_close(self):
            print("WebSocket connection closed.")

        def default_on_message(self, msg):
            print(json.dumps(msg, indent=2))

    def start_live_data(self, api_key, api_secret, symbols, duration=10):
        ws_client = self.MyWebsocketClient(api_key=api_key, api_secret=api_secret, products=symbols)
        ws_client.start()
        time.sleep(duration)
        ws_client.close()  # Ensure the close method is called correctly

    def fetch_historical_data_yahoo(self, symbol, interval, start_date=None):
        ticker = yf.Ticker(symbol)
        if start_date:
            hist = ticker.history(start=start_date, interval=interval)
        else:
            hist = ticker.history(period="max", interval=interval)
        hist.reset_index(inplace=True)
        return hist

# Example usage:
if __name__ == "__main__":
    import config as config  # Assuming you have your API keys in config.py
    data_fetcher = DataFetcher()
    
    # Fetch historical data
    historical_data = data_fetcher.fetch_data(ticker="BTC-USD")
    print(historical_data.head())
    
    # Preprocess data
    features, target = data_fetcher.preprocess_data(historical_data)
    print(features.shape, target.shape)
    
    # Start live data fetching
    symbols = ["BTC-USD"]
    data_fetcher.start_live_data(config.coinbase_api_key, config.coinbase_api_secret, symbols)
