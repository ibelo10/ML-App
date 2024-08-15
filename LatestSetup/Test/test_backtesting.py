import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from backtesting import Backtest
from backtest_strategy import MyStrategy  # Ensure this import matches your file structure

# Adjust the path to the correct location of backtest_strategy.py

class TestBacktesting(unittest.TestCase):
    
    @patch('tensorflow.keras.models.load_model')
    def test_backtest(self, mock_load_model):
        # Mock the model's predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.6]]  # Example prediction value
        mock_load_model.return_value = mock_model

        # Create a sample DataFrame with historical data for testing
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'Open': [1.0] * 100,
            'High': [1.1] * 100,
            'Low': [0.9] * 100,
            'Close': [1.0 + i*0.01 for i in range(100)],
            'Volume': [1000] * 100
        })
        mock_data.set_index('Date', inplace=True)

        # Run the backtest using the mock data and strategy
        bt = Backtest(mock_data, MyStrategy, cash=10000, commission=.002)
        stats = bt.run()

        # Assertions to check the backtest results
        self.assertIsNotNone(stats)
        self.assertGreater(stats['Return [%]'], 0)  # Example assertion to ensure positive return
        print(stats)
        
        # Analyze predictions
        bt.strategy_instance.analyze_predictions()
        
if __name__ == '__main__':
    unittest.main()
