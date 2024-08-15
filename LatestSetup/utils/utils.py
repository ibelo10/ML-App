# visualization.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# utils/utils.py

def clean_dataframe(df):
    """
    Clean the DataFrame by removing any duplicate rows, handling missing values, and ensuring data integrity.

    Parameters:
    - df: pandas DataFrame to clean.

    Returns:
    - df: pandas DataFrame that has been cleaned.
    """
    # Drop duplicate rows if any
    df = df.drop_duplicates()

    # Handle missing values, for example, fill with the mean of the column
    df = df.fillna(df.mean())

    # Optionally, you can drop rows with missing values
    # df = df.dropna()

    return df


def validate_data_shape(df, expected_columns_count):
    """
    Validate the shape of the DataFrame to ensure it has the expected number of columns.

    Parameters:
    - df: pandas DataFrame to validate.
    - expected_columns_count: Integer, the expected number of columns in the DataFrame.

    Returns:
    - df: pandas DataFrame that is validated to have the correct number of columns.
    
    Raises:
    - ValueError: If the DataFrame does not have the expected number of columns.
    """
    if df.shape[1] != expected_columns_count:
        raise ValueError(f"Expected {expected_columns_count} columns, but got {df.shape[1]} columns.")
    
    return df

def align_features(df, reference_features):
    """
    Align the DataFrame columns with a reference set of features.

    Parameters:
    - df: pandas DataFrame whose features need to be aligned.
    - reference_features: List of feature names that the DataFrame should have, in the correct order.

    Returns:
    - df_aligned: pandas DataFrame with columns aligned to reference_features.
    """
    # Create a DataFrame with only the reference features, filling missing ones with 0
    df_aligned = pd.DataFrame(columns=reference_features)
    for feature in reference_features:
        if feature in df.columns:
            df_aligned[feature] = df[feature]
        else:
            # Fill missing columns with zeros
            df_aligned[feature] = 0

    return df_aligned

def normalize_data(df, scaler=None):
    """
    Normalize numerical columns in the dataframe using the provided scaler.
    
    Parameters:
    - df: pandas DataFrame containing the data to be normalized.
    - scaler: An instance of a Scikit-learn scaler (StandardScaler, MinMaxScaler, RobustScaler).
    
    Returns:
    - df: Normalized pandas DataFrame.
    """
    if scaler is None:
        # If no scaler is provided, use StandardScaler as a default.
        scaler = RobustScaler()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit and transform the data
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def configure_page():
    st.set_page_config(
        page_title="Trading Simulation",
        layout="wide",  # Use 'wide' for full-width layout
        initial_sidebar_state="expanded"
    )

def sidebar_inputs():
    st.sidebar.title("Configuration")
    ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    episodes = st.sidebar.number_input("Episodes", value=1000, min_value=1)
    batch_size = st.sidebar.number_input("Batch Size", value=32, min_value=1)
    strategy = st.sidebar.selectbox("Choose Strategy", ("DQN", "A3C", "Rainbow DQN"))
    
    # Dynamic configuration options
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.00001, max_value=0.001, value=0.0005, step=0.00001)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", min_value=0.995, max_value=0.9999, value=0.998, step=0.0001)
    stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
    risk_per_trade = st.sidebar.slider("Risk Per Trade Percentage", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

    # Additional inputs for A3C if selected
    actor_lr = None
    critic_lr = None
    if strategy.lower() == "a3c":
        actor_lr = st.sidebar.slider("Actor Learning Rate", min_value=0.00001, max_value=0.001, value=0.0001, step=0.00001)
        critic_lr = st.sidebar.slider("Critic Learning Rate", min_value=0.00001, max_value=0.001, value=0.001, step=0.00001)
    
    return (ticker, start_date, end_date, episodes, batch_size, strategy, learning_rate, 
            epsilon_decay, stop_loss_pct, risk_per_trade, actor_lr, critic_lr)

def sidebar_simulation_controls():
    st.sidebar.subheader("Run Simulation")
    simulation_type = st.sidebar.radio("Choose Simulation Type", ("Training", "Backtest", "Paper Trading", "Live Trading", "All"))
    return simulation_type

def show_logs():
    with open('data/training_log.log', 'r') as f:
        st.text_area("Logs", f.read(), height=300)

def show_run_button():
    return st.sidebar.button("Run")


def smooth(data, window_size=10):
    """
    Smooths the input data using a simple moving average.

    Args:
    - data (list or np.ndarray): The input data to smooth.
    - window_size (int): The window size for the moving average.

    Returns:
    - np.ndarray: The smoothed data.
    """
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def ensure_adj_close(df):
    """
    Ensure the dataframe contains an 'Adj Close' column.
    If not, it adds it by copying the 'Close' column.
    """
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    return df