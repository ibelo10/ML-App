import sys
import os
import pandas as pd
from backtesting import Backtest, Strategy
from tensorflow.keras.models import load_model
from data.add_features import add_features

# Ensure correct paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

class MyStrategy(Strategy):
    def init(self):
        # Load the pre-trained model (ensure the path is correct)
        self.model = load_model(os.path.join(os.path.dirname(__file__), 'tensormodel', 'best_model.keras'))
        
        # Add features to the data and store them
        self.features = add_features(self.data.df)
        
        # Handle NaN values in features by forward filling, then back filling
        self.features = self.features.fillna(method='ffill').fillna(method='bfill')
        
        # Recheck for any remaining NaN values and handle them
        if self.features.isna().any().any():
            print("Warning: NaN values detected even after fillna. Filling with 0.")
            self.features = self.features.fillna(0)
        
        # Initialize a list to store predictions for later analysis
        self.predictions = []

    def next(self):
        # Define the columns that are required for making predictions
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD', 
                            'Bollinger_Mid', 'Bollinger_High', 'Bollinger_Low', 'ADX', 'Stochastic', 
                            'ATR', 'CCI', 'Williams_%R', 'VWAP', 'OBV', 'CMF', 'MFI']
        
        # Extract the latest features for prediction
        latest_features = self.features[required_columns].iloc[-1:].copy()

        # Convert any datetime columns to numeric (if they exist)
        for col in latest_features.columns:
            if pd.api.types.is_datetime64_any_dtype(latest_features[col]):
                latest_features[col] = latest_features[col].apply(lambda x: x.timestamp())

        # Ensure there are no NaNs in the latest features
        latest_features = latest_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure all features are numeric and of correct dtype
        print("Feature data types before prediction:")
        print(latest_features.dtypes)
        
        # Convert features to a numpy array and ensure it's the correct dtype
        latest_features_array = latest_features.values.astype('float32')

        # Make a prediction using the model
        prediction = self.model.predict(latest_features_array)
        self.predictions.append(prediction[0][0])
        print(f"Prediction: {prediction[0][0]}")

        # Adjusted thresholds for testing
        buy_threshold = 0.48  # Slightly below constant prediction
        sell_threshold = 0.50  # Slightly above constant prediction

        # Execute trading logic based on the prediction
        if prediction > buy_threshold and not self.position:
            print("Buy signal generated")
            self.buy()
        elif prediction < sell_threshold and self.position:
            print("Sell signal generated")
            self.sell()

        # Print the current position for debugging purposes
        print(f"Current position: {self.position}")

    def analyze_predictions(self):
        # Print all collected predictions for analysis after backtesting
        print(f"All predictions: {self.predictions}")

# Example usage in a backtest
if __name__ == "__main__":
    # Load historical data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'path_to_your_data.csv'))
    
    # Ensure the data is in the correct format (e.g., setting the Date column as the index)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Run the backtest
    bt = Backtest(data, MyStrategy, cash=10000, commission=.002)
    stats = bt.run()
    print(stats)

    # Optionally, analyze predictions after the test
    bt.strategy_instance.analyze_predictions()
