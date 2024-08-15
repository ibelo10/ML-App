import unittest
import pandas as pd
import sys
import os

# Adjust the path to import the add_features function
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'LatestSetup'))

from data.add_features import add_features

class TestAddFeatures(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
            'Open': range(100, 200),
            'High': range(110, 210),
            'Low': range(90, 190),
            'Close': range(95, 195),
            'Volume': range(1000, 1100)
        })

    def test_add_features(self):
        df = add_features(self.df)
        
        # Test that all expected features are added
        expected_features = [
            'SMA', 'EMA', 'RSI', 'MACD', 'Bollinger_Mid', 'Bollinger_High', 'Bollinger_Low', 'ADX',
            'Stochastic', 'ATR', 'CCI', 'Williams_%R', 'VWAP', 'OBV', 'CMF', 'MFI'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df.columns)
            # Test if columns are not empty or have unexpected values
            self.assertFalse(df[feature].isna().all(), f"{feature} column is entirely NaN")
        
        # Test specific indicators' mean is close to zero if normalized
        for column in df.columns:
            if column not in ['Date', 'Open', 'High', 'Low', 'Close', 'SMA', 'EMA', 'MACD', 'VWAP']:  # Exclude SMA, EMA, MACD, and VWAP from this check
                mean_value = df[column].mean()
                print(f"Testing column: {column}, Mean: {mean_value}")
                if abs(mean_value) > 0.1:  # If the mean is not within the expected range
                    print(f"Column {column} has a mean of {mean_value}, which is outside the expected range.")
                    print(df[column].describe())  # Print summary statistics for debugging
                self.assertAlmostEqual(mean_value, 0, delta=1e-1)

        # For SMA, EMA, MACD, and VWAP, allow a higher mean since they are moving averages or weighted averages
        for indicator in ['SMA', 'EMA', 'MACD', 'VWAP']:
            mean_value = df[indicator].mean()
            print(f"Testing {indicator} column, Mean: {mean_value}")
            self.assertTrue(0 <= mean_value <= 1, f"{indicator} mean is outside expected range [0, 1]")

if __name__ == '__main__':
    unittest.main()
