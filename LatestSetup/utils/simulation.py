import streamlit as st
import logging
import time
import pandas as pd
import numpy as np
import ccxt
from models.model_training import run_training
from config import API_KEY, API_SECRET
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from data.data_fetching import DataFetcher
from data.add_features import add_features
from utils.utils import ensure_adj_close, normalize_data, align_features, validate_data_shape, clean_dataframe


def run_backtest(agent, historical_data, expected_features):
    st.write("Starting Backtesting...")  # Indicate that backtesting has started
    
    with st.spinner("Backtesting in progress..."):
        class DQNStrategy(Strategy):
            def init(self):
                self.agent = agent
                self.custom_position = 0
                self.expected_features = expected_features

            def next(self):
                # Log the selected feature columns for debugging
                feature_columns = [col for col in self.expected_features if col != 'Date']
                logging.debug(f"Selected feature columns for current_state: {feature_columns}")

                # Check if the number of features matches the expected count
                if len(feature_columns) != self.agent.state_size:
                    logging.error(f"Feature column count mismatch. Expected {self.agent.state_size}, but got {len(feature_columns)}")

                # Check for NaN values in the selected columns
                if self.data.df[feature_columns].isna().any().any():
                    logging.error(f"NaN values found in feature columns:\n{self.data.df[feature_columns].isna().sum()}")

                # Extract the current state from the last row of the data
                current_state = self.data.df[feature_columns].iloc[-1].values
                
                # Log the values and shape of current_state
                logging.debug(f"current_state values: {current_state}")
                logging.debug(f"current_state shape: {current_state.shape}, agent.state_size: {self.agent.state_size}")

                # Check if the shape matches the agent's expected input
                if current_state.shape[0] != self.agent.state_size:
                    missing_features = set(self.expected_features) - set(self.data.df.columns)
                    logging.error(f"Expected {self.agent.state_size} features but got {current_state.shape[0]}. Missing features: {missing_features}")
                    return
                
                # Reshape and proceed with the model
                current_state = np.reshape(current_state, (1, self.agent.state_size))
                action = self.agent.act(current_state, is_eval=True)
                if action == 1 and not self.custom_position:
                    self.buy()
                    self.custom_position = 1
                elif action == 2 and self.custom_position:
                    self.sell()
                    self.custom_position = 0





                # Log the shape of current_state and agent.state_size for debugging
                logging.debug(f"current_state shape: {current_state.shape}, agent.state_size: {self.agent.state_size}")
                
                # Check if the shape matches the agent's expected input
                if current_state.shape[0] != self.agent.state_size:
                    missing_features = set(self.expected_features) - set(self.data.df.columns)
                    st.error(f"Expected {self.agent.state_size} features but got {current_state.shape[0]}. Missing features: {missing_features}")
                    logging.error(f"Expected {self.agent.state_size} features but got {current_state.shape[0]}. Missing features: {missing_features}")
                    return
                
                # Reshape to match the expected input shape of the model
                current_state = np.reshape(current_state, (1, self.agent.state_size))
                
                action = self.agent.act(current_state, is_eval=True)
                if action == 1 and not self.custom_position:
                    self.buy()
                    self.custom_position = 1
                elif action == 2 and self.custom_position:
                    self.sell()
                    self.custom_position = 0

        # Ensure feature alignment with the model's expected features
        logging.debug(f"Expected features: {expected_features}")
        logging.debug(f"Available columns in historical data: {historical_data.columns.tolist()}")

        available_columns = historical_data.columns.intersection(expected_features)
        if len(available_columns) == 0:
            raise KeyError(f"None of the expected features are available in the historical data: {expected_features}")

        historical_data = historical_data[available_columns]  # Align columns

        # Log final columns after alignment for debugging
        logging.debug(f"Final columns after alignment: {historical_data.columns.tolist()}")

        bt = Backtest(historical_data, DQNStrategy, cash=10_000, commission=.002)
        stats = bt.run()

        # Ensure that the DataFrame only contains numeric data
        stats_df = stats.to_frame().select_dtypes(include=[np.number])

        # Clean the DataFrame
        stats_df = clean_dataframe(stats_df)

        st.session_state['backtest_results'] = stats_df  # Store the backtest results
        st.write(stats_df)
    
    st.success("Backtesting Completed!")  # Indicate that backtesting has finished
  # Indicate that backtesting has finished

def run_paper_trading(agent, ticker, scaler, initial_features):
    st.write("Starting Paper Trading...")

    with st.spinner("Paper trading in progress..."):
        data_fetcher = DataFetcher()
        symbol = ticker.replace("-", "/") + "C"  # Convert ticker to the format used by ccxt and add 'C'
        timeframe = '1m'  # Use 1-minute candles for paper trading
        paper_trading_duration = timedelta(days=1)  # Run for 1 day, adjust as needed
        end_time = datetime.now() + paper_trading_duration

        # Initialize portfolio with some cash (e.g., $10,000)
        cash = 10000.0
        position = 0
        portfolio_value = cash

        # Initialize an empty DataFrame for historical data
        historical_data_paper = pd.DataFrame()

        trades_log = st.empty()  # Placeholder for trade logs
        price_placeholder = st.empty()  # Placeholder for displaying the latest price
        chart_placeholder = st.empty()  # Placeholder for displaying the chart
        portfolio_value_placeholder = st.empty()  # Placeholder for displaying portfolio value

        prices = []  # List to store the latest prices for charting

        while datetime.now() < end_time:
            try:
                # Fetch the latest candle using DataFetcher method
                new_data = data_fetcher.fetch_historical_data_coinbase(symbol, timeframe)
                
                if new_data.empty:
                    logging.warning("Fetched data is empty. Skipping this iteration.")
                    continue

                # Verify the fetched data contains the expected fields
                if 'Close' not in new_data.columns:
                    logging.error("Close price is missing from the fetched data. Skipping this iteration.")
                    continue

                # Store the original 'Close' price before processing
                original_close = new_data['Close'].iloc[-1]

                # Log raw data for debugging
                logging.info(f"Raw data fetched: {new_data}")

                # Add the fetched data to the historical data
                historical_data_paper = pd.concat([historical_data_paper, new_data], ignore_index=True)
                historical_data_paper = ensure_adj_close(historical_data_paper)  # Ensure 'Adj Close' exists

                # Add features and normalize
                processed_data = add_features(historical_data_paper)
                columns_to_normalize = [col for col in processed_data.columns if col != 'Close']
                processed_data, _ = normalize_data(processed_data, columns_to_normalize)
                
                # Align features with the training data
                processed_data = align_features(processed_data, initial_features)

                # Check for realistic price values
                if not (100 <= original_close <= 100000):
                    logging.warning(f"Unusual price detected: {original_close}. Please verify data integrity.")
                    continue  # Skip this iteration if the price is unusual

                # Add the latest price to the list for charting
                prices.append(original_close)
                
                # Update the displayed price and chart
                price_placeholder.metric("Latest Price", f"${original_close:.2f}")
                chart_placeholder.line_chart(prices)

                # Proceed with the trading decision using the normalized state data
                latest_state = processed_data.iloc[-1].values.reshape(1, processed_data.shape[1])
                action = agent.act(latest_state, is_eval=True)
                
                action_str = 'Buy' if action == 1 else 'Sell' if action == 2 else 'Sit'
                timestamp = datetime.now()

                # Execute the trade based on the action
                if action == 1 and cash >= original_close:  # Buy action
                    position += 1
                    cash -= original_close
                    reason_for_trade = "Bought at market price"
                elif action == 2 and position > 0:  # Sell action
                    position -= 1
                    cash += original_close
                    reason_for_trade = "Sold at market price"
                else:
                    reason_for_trade = "No trade executed"
                    action_str = 'Sit'  # Change action to Sit if no trade is executed

                # Calculate the portfolio value
                portfolio_value = cash + (position * original_close)

                # Display the trade details
                trade_details = f"Executed Trade: {action_str} at ${original_close:.2f} | Reason: {reason_for_trade} | Portfolio Value: ${portfolio_value:.2f}"
                trades_log.text(trade_details)
                portfolio_value_placeholder.metric("Portfolio Value", f"${portfolio_value:.2f}")
                logging.info(trade_details)

            except ccxt.BaseError as e:
                st.error(f"Error fetching data: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                break  # Exit the loop if there's any unexpected error

            time.sleep(60)  # Wait for the next minute to fetch new data
    
    st.success("Paper Trading Completed!")

    # Re-display the backtest results after paper trading
    if 'backtest_results' in st.session_state:
        st.write("Previous Backtest Results:")
        st.write(st.session_state['backtest_results'])

# Function to fetch OHLCV data with retries
def fetch_data_with_retries(exchange, symbol, timeframe, max_retries=3, sleep_time=5):
    """Fetch OHLCV data with retries in case of temporary issues."""
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1)
            if ohlcv and len(ohlcv[0]) == 6:
                return ohlcv[-1]
            else:
                logging.warning("No or incomplete OHLCV data returned. Retrying...")
                time.sleep(sleep_time)
        except ccxt.BaseError as e:
            logging.error(f"Error fetching data: {str(e)}. Retrying...")
            time.sleep(sleep_time)
    logging.error("Max retries reached. Returning None.")
    return None

def run_live_trading(agent, ticker, scaler, initial_features):
    st.write("Starting Live Trading...")

    with st.spinner("Live trading in progress..."):
        exchange = ccxt.coinbase({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
        })

        symbol = ticker.replace("-", "/")  # Convert ticker to the format used by ccxt
        timeframe = '1m'  # Use 1-minute candles for live trading

        # Initialize portfolio with some cash (e.g., $10,000)
        cash = 10000.0
        position = 0
        portfolio_value = cash

        trades_log = st.empty()  # Placeholder for trade logs
        price_placeholder = st.empty()  # Placeholder for displaying the latest price
        chart_placeholder = st.empty()  # Placeholder for displaying the chart
        portfolio_value_placeholder = st.empty()  # Placeholder for displaying portfolio value

        prices = []  # List to store the latest prices for charting

        while True:  # Continuous loop for live trading
            try:
                ohlcv = fetch_data_with_retries(exchange, symbol, timeframe)
                if ohlcv is None:
                    logging.error("Failed to fetch valid OHLCV data after retries. Stopping trading.")
                    break

                timestamp, open, high, low, close, volume = ohlcv
                new_data = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(timestamp / 1000.0)],
                    'Open': [open],
                    'High': [high],
                    'Low': [low],
                    'Close': [close],
                    'Volume': [volume]
                })

                logging.info(f"Raw data fetched: {new_data}")

                # Store the original 'Close' price for trade execution
                original_close = new_data['Close'].iloc[-1]

                # Process and validate data
                new_data = ensure_adj_close(new_data)
                processed_data = add_features(new_data)
                processed_data = align_features(processed_data, initial_features)

                # Normalize features except 'Close'
                columns_to_normalize = [col for col in processed_data.columns if col != 'Close']
                processed_data[columns_to_normalize] = scaler.transform(processed_data[columns_to_normalize])

                validate_data_shape(processed_data, len(initial_features))

                # Reshape the latest state
                latest_state = processed_data.iloc[-1].values.reshape(1, -1)
                logging.debug(f"Latest state shape after reshaping: {latest_state.shape}")

                # Add the latest original price to the list for charting
                prices.append(original_close)
                price_placeholder.metric("Latest Price", f"${original_close:.2f}")
                chart_placeholder.line_chart(prices)

                # Trading decision
                action = agent.act(latest_state, is_eval=True)
                action_str = 'Buy' if action == 1 else 'Sell' if action == 2 else 'Sit'
                timestamp = datetime.now()

                if action == 1 and cash >= original_close:  # Buy action
                    order = exchange.create_market_buy_order(symbol, 1)
                    position += 1
                    cash -= original_close
                    reason_for_trade = "Bought at market price"
                elif action == 2 and position > 0:  # Sell action
                    order = exchange.create_market_sell_order(symbol, 1)
                    position -= 1
                    cash += original_close
                    reason_for_trade = "Sold at market price"
                else:
                    reason_for_trade = "No trade executed"
                    action_str = 'Sit'

                portfolio_value = cash + (position * original_close)
                trade_details = f"Executed Trade: {action_str} at ${original_close:.2f} | Reason: {reason_for_trade} | Portfolio Value: ${portfolio_value:.2f}"
                trades_log.text(trade_details)
                portfolio_value_placeholder.metric("Portfolio Value", f"${portfolio_value:.2f}")
                logging.info(trade_details)

            except ValueError as e:
                logging.error(f"Data validation error: {str(e)}")
                st.error(f"Data validation error: {str(e)}")
                break
            except ccxt.BaseError as e:
                logging.error(f"Error fetching data: {str(e)}")
                st.error(f"Error fetching data: {str(e)}")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                st.error(f"Unexpected error: {str(e)}")
                break

            time.sleep(60)  # Wait for the next minute to fetch new data
    
    st.success("Live Trading Completed!")


def run_all():
    st.write("Running Training, Backtesting, Paper Trading, and Live Trading sequentially...")
    
    with st.spinner("Running Training..."):
        run_training()

    with st.spinner("Running Backtesting..."):
        run_backtest()

    with st.spinner("Running Paper Trading..."):
        run_paper_trading()
        
    with st.spinner("Running Live Trading..."):
        run_live_trading()

    st.success("All simulations completed!")
