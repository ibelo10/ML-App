import visualization as viz
# Configure the Streamlit page
viz.configure_page()

import os
import logging
import pandas as pd
import numpy as np
import streamlit as st
from models.model_dqn import Agent, TradingEnv
from data.data_fetching import DataFetcher
from data.add_features import add_features
from utils.simulation import run_backtest, run_paper_trading, run_live_trading, run_all
from utils.utils import ensure_adj_close, clean_dataframe, validate_data_shape
from models.model_training import run_training
from sklearn.preprocessing import RobustScaler

# Sidebar inputs
(ticker, start_date, end_date, episodes, batch_size, strategy, 
 learning_rate, epsilon_decay, stop_loss_pct, risk_per_trade, 
 actor_lr, critic_lr) = viz.sidebar_inputs()

# Normalize strategy to lowercase before passing it to the Agent class
normalized_strategy = strategy.lower()

# Logging setup
os.makedirs('data', exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('data/training_log.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)
logger.info("Logging configuration is set up correctly.")

def flush_logs():
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

# Fetch historical data
st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")
data_fetcher = DataFetcher()
historical_data = data_fetcher.fetch_data(ticker=ticker, start_date=start_date, end_date=end_date)

# Ensure 'Adj Close' exists
historical_data = ensure_adj_close(historical_data)
original_close = historical_data[['Close']].copy()

# Validate, clean, and preprocess data
historical_data = validate_data_shape(historical_data, len(historical_data.columns))
historical_data = clean_dataframe(historical_data)
historical_data = add_features(historical_data)

# Normalize data
columns_to_normalize = historical_data.select_dtypes(include=[np.number]).columns.tolist()
scaler = RobustScaler()
historical_data[columns_to_normalize] = scaler.fit_transform(historical_data[columns_to_normalize])
expected_features = historical_data.columns.tolist()

# Initialize the agent
if normalized_strategy in ["dqn", "rainbow dqn"]:
    agent = Agent(
        state_size=len(expected_features),
        action_size=3,
        strategy=normalized_strategy,
        learning_rate=learning_rate,
        epsilon_decay=epsilon_decay
    )
elif normalized_strategy == "a3c":
    agent = Agent(
        state_size=len(expected_features),
        action_size=3,
        strategy=normalized_strategy,
        actor_lr=actor_lr,
        critic_lr=critic_lr
    )
else:
    st.error(f"Unknown strategy: {strategy}")
    st.stop()

# Initialize the trading environment
env = TradingEnv(data=historical_data, stop_loss_pct=stop_loss_pct, risk_per_trade=risk_per_trade)

# Simulation controls
simulation_type = viz.sidebar_simulation_controls()

# Prevent duplicated button clicks
if "run_button_pressed" not in st.session_state:
    st.session_state.run_button_pressed = False

# Run simulation if the button is clicked and wasn't clicked before
if viz.show_run_button() and not st.session_state.run_button_pressed:
    st.session_state.run_button_pressed = True  # Mark button as pressed

    if simulation_type == "Training":
        run_training(strategy, env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size)
    elif simulation_type == "Backtest":
        run_backtest(agent, historical_data, expected_features)
    elif simulation_type == "Paper Trading":
        run_paper_trading(agent, ticker, scaler, expected_features)
    elif simulation_type == "Live Trading":
        run_live_trading(agent, ticker, scaler, expected_features)
    elif simulation_type == "All":
        run_all(agent, historical_data, ticker, scaler, expected_features)
    
    flush_logs()
    viz.show_logs()
