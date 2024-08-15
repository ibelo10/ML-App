import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

# Add the parent directory of the project to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tensorflow_model import train_model  # Ensure this path matches your file structure
from data.add_features import add_features  # Ensure this path matches your file structure
from data.data_fetching import DataFetcher  # Ensure this path matches your file structure

class TestTensorFlowModel(unittest.TestCase):

    def setUp(self):
        # Use DataFetcher to load real dataset
        data_fetcher = DataFetcher()
        self.df = data_fetcher.fetch_data(ticker="BTC-USD", start_date="2020-01-01", end_date="2023-12-31")
        # Add features to the real dataset
        self.df = add_features(self.df)

    @patch('models.tensorflow_model.logger')
    def test_train_model(self, mock_logger):
        # Ensure DataFrame has the required columns
        required_columns = {'Date', 'High', 'Low', 'Close'}
        self.assertTrue(required_columns.issubset(self.df.columns))

        # Print the DataFrame columns to ensure they are correct
        print("DataFrame columns before training:", self.df.columns)
        
        # Test train_model function with the real DataFrame
        model_path = train_model(self.df, mock_logger)
        
        # Check if the returned model path ends with '.keras'
        self.assertTrue(model_path.endswith('.keras'))
        
        # Optionally check if the path exists
        self.assertTrue(os.path.exists(model_path))

if __name__ == '__main__':
    unittest.main()
