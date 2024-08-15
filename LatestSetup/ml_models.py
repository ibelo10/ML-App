import streamlit as st
import pandas as pd
import numpy as np
import psutil
import gc
import logging
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from models.model_dqn import Agent, TradingEnv, Worker  # Ensure correct import for TradingEnv
from data.data_fetching import DataFetcher
from data.add_features import add_features
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from config import COINBASE_API_KEY, COINBASE_API_SECRET, API_KEY, API_SECRET, OPENAI_API_KEY
import ccxt

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check data integrity
def check_data_integrity(data):
    if data.isnull().values.any():
        st.warning("Data contains NaN values. They will be filled with 0.")
        data.fillna(0, inplace=True)
    if np.isinf(data.values).any():
        st.warning("Data contains infinite values. They will be replaced with max finite values.")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(data.max().max(), inplace=True)
    return data

# Function to validate data shape before reshaping
def validate_data_shape(data, expected_shape):
    if data.shape[1] != expected_shape:
        raise ValueError(f"Expected {expected_shape} features, but got {data.shape[1]}.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)) or data.shape[0] == 0:
        raise ValueError("Data contains NaN, infinite values, or is empty.")
    return data

# Function to process data in chunks
def process_in_chunks(df, chunk_size=10000):
    chunks = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
    processed_chunks = [add_features(chunk) for chunk in chunks]
    return pd.concat(processed_chunks)

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


# Ensure no Timedelta objects or other incompatible types are in the DataFrame
def clean_dataframe(df):
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()  # Convert timedelta to seconds
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)  # Convert objects to strings
    return df

# Function to normalize data
def normalize_data(data, columns_to_normalize):
    scaler = MinMaxScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    return data, scaler  # Return the scaler to apply to new data later
# Function to ensure 'Adj Close' exists
def ensure_adj_close(df):
    if 'Adj Close' not in df.columns:
        logging.warning("'Adj Close' column missing. Adding 'Adj Close' by copying 'Close' column.")
        df['Adj Close'] = df['Close']
    return df

# Function to align features during paper trading
def align_features(df, expected_features):
    current_features = df.columns.tolist()
    missing_features = set(expected_features) - set(current_features)
    extra_features = set(current_features) - set(expected_features)
    
    # Add missing features with zeros
    for feature in missing_features:
        df[feature] = 0
    
    # Drop extra features
    df = df.drop(columns=extra_features)
    
    return df[expected_features]

# Initialize Streamlit app
st.set_page_config(layout="wide")
st.title("Trading Bot Progress Monitor")
st.subheader("Monitor the performance and progress of your trading bot.")

# Sidebar for user inputs
st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
episodes = st.sidebar.number_input("Episodes", value=1000, min_value=1)
batch_size = st.sidebar.number_input("Batch Size", value=32, min_value=1)

# Strategy selection
strategy = st.sidebar.selectbox("Choose Strategy", ("DQN", "Rainbow DQN", "A3C"))

# Dynamic configuration options
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.00001, max_value=0.001, value=0.0005, step=0.00001)
epsilon_decay = st.sidebar.slider("Epsilon Decay", min_value=0.995, max_value=0.9999, value=0.998, step=0.0001)
stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
risk_per_trade = st.sidebar.slider("Risk Per Trade Percentage", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

# Fetch historical data
st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")
data_fetcher = DataFetcher()
historical_data = data_fetcher.fetch_data(ticker=ticker, start_date=start_date, end_date=end_date)

# Ensure 'Adj Close' exists
historical_data = ensure_adj_close(historical_data)

# Ensure 'Date' is a valid column before setting it as the index
if 'Date' in historical_data.columns:
    historical_data.set_index('Date', inplace=True)
else:
    st.error("The data does not contain a 'Date' column. Please check the dataset.")

# Store the original 'Close' for non-training use
original_close = historical_data[['Close']].copy()

# Check and preprocess data
historical_data = check_data_integrity(historical_data)

# Clean the dataframe
historical_data = clean_dataframe(historical_data)

# Add features to the data
historical_data = add_features(historical_data)

# Normalize the data except for the 'Close' column
columns_to_normalize = [col for col in historical_data.columns if col != 'Close']
historical_data, scaler = normalize_data(historical_data, columns_to_normalize)

# Store expected features after training
expected_features = historical_data.columns.tolist()
state_size = len(expected_features)
st.session_state['initial_features'] = expected_features
st.session_state['state_size'] = state_size
st.session_state['scaler'] = scaler  # Store the scaler for later use
st.write(f"State size determined: {state_size}")

# Initialize the agent based on selected strategy
agent = Agent(state_size=state_size, action_size=3, strategy=strategy.lower(), learning_rate=learning_rate, epsilon_decay=epsilon_decay)

# Initialize the trading environment with the processed historical data
env = TradingEnv(data=historical_data, stop_loss_pct=stop_loss_pct, risk_per_trade=risk_per_trade)

# Use columns to display charts side by side
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Reward Over Episodes")
    score_chart_placeholder = st.empty()
    score_chart = score_chart_placeholder.line_chart([0.0])

with col2:
    st.write("Loss Over Episodes")
    loss_chart_placeholder = st.empty()
    loss_chart = loss_chart_placeholder.line_chart([0.0])

with col3:
    st.write("Average Reward (Smoothed)")
    avg_reward_placeholder = st.empty()
    avg_reward_chart = avg_reward_placeholder.line_chart([0.0])

# Memory and performance monitoring
memory_threshold = 40000
cpu_usage_display = st.sidebar.metric("CPU Usage (%)", 0)
memory_usage_display = st.sidebar.metric("Memory Usage (MB)", 0)

def run_dqn_training():
    log_text = ""
    total_rewards = []
    smoothed_reward = []
    smoothed_loss = []
    avg_reward_data = []
    trades = []
    score_data = []
    latest_loss_display = st.empty()
    total_reward_display = st.empty()
    progress_bar = st.progress(0)
    log_display = st.empty()

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (state_size,))  # Reshape for the model input

        # Apply normalization to the state
        state = transform_state(state, scaler, columns_to_normalize, expected_features)

        total_reward = 0
        episode_loss = 0
        episode_trades = []

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            if np.isnan(reward) or np.isinf(reward):
                logging.warning(f"Reward became nan or inf at step {time_step} in episode {e+1}, setting reward to 0.")
                reward = 0.0

            if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                logging.error(f"Next state contains NaN or inf values at step {time_step} in episode {e+1}.")

            next_state = np.reshape(next_state, (state_size,))  # Reshape next_state for model input

            # Apply normalization to the next state
            next_state = transform_state(next_state, scaler, columns_to_normalize, expected_features)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Use the original (unnormalized) 'Close' prices for logging
            original_close = env.data.iloc[env.current_step]['Close']
            episode_trades.append((env.current_step, action, original_close))

            if done:
                break

        trades.append(episode_trades)

        # Smooth functions
        def smooth(x, window=10):
            if len(x) < window:
                return x
            return np.convolve(x, np.ones(window) / window, mode='valid')

        smoothed_reward.append(smooth([total_reward])[-1])
        smoothed_reward_data = smooth(total_rewards)

        log_text += f"Episode {e + 1}:\n"
        log_text += f"  Total Reward: {total_reward}\n"
        log_text += f"  Episode Loss: {episode_loss}\n"
        log_text += "  Trades:\n"
        for step, action, price in episode_trades:
            action_str = "Buy" if action == 1 else "Sell" if action == 2 else "Sit"
            log_text += f"    Step {step}: {action_str} at price {price:.6f}\n"
        log_text += "\n"

        if e % 10 == 0:
            gc.collect()

        log_display.text_area("Training Log", log_text, height=150, key=f"log_text_area_{e}")

        memory_usage = psutil.virtual_memory().used / 1024 / 1024
        memory_total = psutil.virtual_memory().total / 1024 / 1024
        logging.info(f"Memory usage at episode {e}: {round(memory_usage, 2)} MB")

        if memory_usage > 0.9 * memory_total:
            st.warning(f"Memory usage exceeded 90% of total memory. Consider running garbage collection.")

        cpu_usage = psutil.cpu_percent()
        memory_usage_display.metric("Memory Usage (MB)", round(memory_usage, 2))
        cpu_usage_display.metric("CPU Usage (%)", round(cpu_usage, 2))

        logging.info(f"Memory usage at episode {e}: {round(memory_usage, 2)} MB")

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-50:])
        avg_reward_data.append(avg_reward)

        if len(agent.memory) > batch_size * 5:
            try:
                loss = agent.train_experience_replay()
                loss = 0.0 if np.isnan(loss) else loss
                episode_loss += loss
                smoothed_loss.append(smooth([loss])[-1])
                loss_chart.add_rows([smoothed_loss[-1]])
                latest_loss_display.metric("Latest Loss", round(smoothed_loss[-1], 4))
                log_text += f"  Loss: {loss}\n"
            except Exception as err:
                st.error(f"Error during training: {err}")
                break

        log_text += "\n"

        score_data.append(total_reward)
        score_chart.add_rows([total_reward])
        avg_reward_chart.add_rows([smoothed_reward[-1]])
        total_reward_display.metric("Total Reward", round(total_reward, 4))

        progress_bar.progress((e + 1) / episodes)

        if e % 50 == 0:
            agent.save(e)

    st.subheader("Training Log")
    st.expander("Training Log", expanded=True).text_area("Training Log", log_text, height=150, key="final_log_text")

    st.success("Training Completed!")


def transform_state(state, scaler, columns_to_normalize, all_columns):
    # Ensure state is reshaped correctly before converting it to a DataFrame
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    
    # Create a DataFrame with the correct columns
    state_df = pd.DataFrame(state, columns=all_columns)
    
    # Normalize only the specified columns
    state_df[columns_to_normalize] = scaler.transform(state_df[columns_to_normalize])

    # Ensure the transformed state maintains the correct number of features
    state_normalized = state_df.values.flatten()
    
    return state_normalized

def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')

def run_a3c_training():
    log_text = ""
    total_rewards = []
    smoothed_reward = []
    avg_reward_data = []
    score_data = []
    trades = []  # Define trades here
    latest_loss_display = st.empty()
    total_reward_display = st.empty()
    progress_bar = st.progress(0)
    log_display = st.empty()
    smoothed_loss = []  # Initialize for smoothed loss tracking
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, state_size))  # Add batch dimension

        # Apply normalization to the state
        state = scaler.transform(state)

        total_reward = 0
        episode_trades = []
        episode_actor_loss = 0.0  # Ensure it's a float
        episode_critic_loss = 0.0  # Ensure it's a float

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            if np.isnan(reward) or np.isinf(reward):
                logging.warning(f"Reward became nan or inf at step {time_step} in episode {e+1}, setting reward to 0.")
                reward = 0.0

            if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                logging.error(f"Next state contains NaN or inf values at step {time_step} in episode {e+1}.")
            
            next_state = np.reshape(next_state, (1, state_size))  # Add batch dimension
            
            # Apply normalization to the next state
            next_state = scaler.transform(next_state)

            discounted_reward = reward + agent.agent.gamma * agent.agent.critic.predict(next_state) if not done else reward
            advantage = discounted_reward - agent.agent.critic.predict(state)

            # Calculate the actor and critic losses
            action_one_hot = np.zeros(agent.action_size)
            action_one_hot[action] = 1

            # Update Actor and Critic
            actor_loss = agent.agent._actor_optimizer(state, action_one_hot, advantage)
            critic_loss = agent.agent._critic_optimizer(state, discounted_reward)

            episode_actor_loss += actor_loss.numpy()  # Convert Tensor to float
            episode_critic_loss += critic_loss.numpy()  # Convert Tensor to float

            state = next_state
            total_reward += reward

            price = original_close.iloc[env.current_step]['Close']
            episode_trades.append((env.current_step, action, price))

            if done:
                break

        trades.append(episode_trades)

        def smooth(x, window=10):
            if len(x) < window:
                return x
            return np.convolve(x, np.ones(window) / window, mode='valid')

        smoothed_reward.append(smooth([total_reward])[-1])
        smoothed_reward_data = smooth(total_rewards)

        # Calculate smoothed loss (averaging over the last few episodes)
        episode_loss = episode_actor_loss + episode_critic_loss
        smoothed_loss.append(smooth([episode_loss])[-1])

        log_text += f"Episode {e + 1}:\n"
        log_text += f"  Total Reward: {total_reward}\n"
        log_text += f"  Actor Loss: {episode_actor_loss}\n"
        log_text += f"  Critic Loss: {episode_critic_loss}\n"
        log_text += "  Trades:\n"
        for step, action, price in episode_trades:
            action_str = "Buy" if action == 1 else "Sell" if action == 2 else "Sit"
            log_text += f"    Step {step}: {action_str} at price {price:.6f}\n"
        log_text += "\n"

        if e % 10 == 0:
            gc.collect()

        log_display.text_area("Training Log", log_text, height=150, key=f"log_text_area_{e}")

        memory_usage = psutil.virtual_memory().used / 1024 / 1024
        memory_total = psutil.virtual_memory().total / 1024 / 1024
        logging.info(f"Memory usage at episode {e}: {round(memory_usage, 2)} MB")

        if memory_usage > 0.9 * memory_total:
            st.warning(f"Memory usage exceeded 90% of total memory. Consider running garbage collection.")

        cpu_usage = psutil.cpu_percent()
        memory_usage_display.metric("Memory Usage (MB)", round(memory_usage, 2))
        cpu_usage_display.metric("CPU Usage (%)", round(cpu_usage, 2))

        logging.info(f"Memory usage at episode {e}: {round(memory_usage, 2)} MB")

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-50:])
        avg_reward_data.append(avg_reward)

        # Display the loss
        loss_chart.add_rows([smoothed_loss[-1]])
        latest_loss_display.metric("Latest Loss", round(smoothed_loss[-1], 4))
        
        score_data.append(total_reward)
        score_chart.add_rows([total_reward])
        avg_reward_chart.add_rows([smoothed_reward[-1]])
        total_reward_display.metric("Total Reward", round(total_reward, 4))

        progress_bar.progress((e + 1) / episodes)

        if e % 50 == 0:
            agent.save(e)

    st.subheader("Training Log")
    st.expander("Training Log", expanded=True).text_area("Training Log", log_text, height=150, key="final_log_text")

    st.success("Training Completed!")

def run_training():
    strategy_lower = strategy.lower()  # Convert the strategy to lowercase
    logging.info(f"Selected strategy: {strategy_lower}")

    if strategy_lower == "dqn":
        logging.info("Running DQN training...")
        run_dqn_training()
    elif strategy_lower == "rainbow dqn":
        logging.info("Running Rainbow DQN training...")
        run_dqn_training()  # Rainbow DQN follows similar training as DQN
    elif strategy_lower == "a3c":
        logging.info("Running A3C training...")
        run_a3c_training()
    else:
        st.error(f"Unknown strategy: {strategy}")
        logging.error(f"Unknown strategy: {strategy}")


# Function to compare features and log missing ones
def compare_features(current_features):
    missing_features = set(st.session_state['initial_features']) - set(current_features)
    if missing_features:
        logging.error(f"Missing features: {missing_features}")
        st.error(f"Missing features: {missing_features}")
    return missing_features

def run_backtest():
    st.write("Starting Backtesting...")  # Indicate that backtesting has started
    
    # Optionally, you can add a spinner to show progress
    with st.spinner("Backtesting in progress..."):
        class DQNStrategy(Strategy):
            def init(self):
                self.agent = agent
                self.custom_position = 0

            def next(self):
                current_state = self.data.df.iloc[-1].values.reshape(1, -1)
                action = self.agent.act(current_state, is_eval=True)
                if action == 1 and not self.custom_position:
                    self.buy()
                    self.custom_position = 1
                elif action == 2 and self.custom_position:
                    self.sell()
                    self.custom_position = 0

        bt = Backtest(historical_data, DQNStrategy, cash=10_000, commission=.002)
        stats = bt.run()
        stats_df = clean_dataframe(stats.to_frame())
        st.session_state['backtest_results'] = stats_df  # Store the backtest results
        st.write(stats_df)
    
    st.success("Backtesting Completed!")  # Indicate that backtesting has finished

def run_paper_trading():
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
                processed_data = align_features(processed_data, st.session_state['initial_features'])

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

def run_live_trading():
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
                processed_data = align_features(processed_data, st.session_state['initial_features'])

                # Normalize features except 'Close'
                columns_to_normalize = [col for col in processed_data.columns if col != 'Close']
                processed_data[columns_to_normalize] = st.session_state['scaler'].transform(processed_data[columns_to_normalize])

                validate_data_shape(processed_data, st.session_state['state_size'])

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

# Sidebar for running backtesting or paper trading
st.sidebar.subheader("Run Simulation")
simulation_type = st.sidebar.radio("Choose Simulation Type", ("Training", "Backtest", "Paper Trading", "Live Trading", "All"))

# Add a button to start the selected process
if st.sidebar.button("Run"):
    if simulation_type == "Training":
        run_training()
    elif simulation_type == "Backtest":
        run_backtest()
    elif simulation_type == "Paper Trading":
        run_paper_trading()
    elif simulation_type == "Live Trading":
        run_live_trading()
    elif simulation_type == "All":
        run_all()
