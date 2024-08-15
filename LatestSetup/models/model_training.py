import numpy as np
import gc
import psutil
import streamlit as st
import pandas as pd
from models.model_dqn import TradingEnv
from utils.utils import smooth  # Import the smooth function
from data.add_features import add_features
import logging
import os

# Set up logging configuration
log_file = os.path.join('logs', 'training_log.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to capture all logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Log to file
        logging.StreamHandler()  # Optional: Also log to console/Streamlit
    ]
)

# Test logging to verify setup
logging.info("Logging configuration initialized.")

# Define metric placeholders once for reuse
memory_usage_display = st.sidebar.empty()
cpu_usage_display = st.sidebar.empty()

# Function to update and display memory and CPU usage
def update_metrics():
    memory_usage = psutil.virtual_memory().used / 1024 / 1024
    cpu_usage = psutil.cpu_percent()

    memory_usage_display.metric("Memory Usage (MB)", round(memory_usage, 2))
    cpu_usage_display.metric("CPU Usage (%)", round(cpu_usage, 2))

    if memory_usage > 0.9 * psutil.virtual_memory().total / 1024 / 1024:
        st.warning(f"Memory usage exceeded 90% of total memory. Consider running garbage collection.")

# Function to clip values before transformation
def clip_values(df, max_value=1e6):
    df = df.clip(-max_value, max_value)
    return df

# Example for A3C Loss Logging
def calculate_a3c_loss(actor_loss, critic_loss):
    # Log the actor and critic losses
    logging.info(f"Actor Loss: {actor_loss}")
    logging.info(f"Critic Loss: {critic_loss}")

    total_loss = actor_loss + critic_loss

    logging.info(f"Total Loss: {total_loss}")

    return total_loss

# Example for DQN Loss Logging
def calculate_dqn_loss(target_q_values, predicted_q_values):
    # Calculate the MSE loss
    loss = np.mean((target_q_values - predicted_q_values) ** 2)

    # Log the target and predicted Q-values and loss
    logging.info(f"Target Q-values: {target_q_values}")
    logging.info(f"Predicted Q-values: {predicted_q_values}")
    logging.info(f"Loss: {loss}")

    return loss

# Function to transform state for the model input
def transform_state(state, scaler, columns_to_normalize, all_columns):
    logging.debug(f"Initial state before transformation: {state}")
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    state_df = pd.DataFrame(state, columns=all_columns)
    
    # Convert datetime columns to timestamps
    for col in state_df.columns:
        if pd.api.types.is_datetime64_any_dtype(state_df[col]):
            state_df[col] = state_df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
    
    # Handle NaN and infinite values
    state_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    state_df.fillna(0, inplace=True)
    
    # Clip the state_df values before normalization
    state_df = clip_values(state_df)
    logging.debug(f"State after clipping values: {state_df.head()}")
    
    # Standardize the data (Z-score normalization)
    state_df[columns_to_normalize] = scaler.transform(state_df[columns_to_normalize])
    logging.debug(f"State after normalization: {state_df.head()}")
    
    state_normalized = state_df.values.flatten()
    logging.debug(f"Final transformed state: {state_normalized}")
    
    return state_normalized

# Function to preprocess data for training
def preprocess_data_for_training(historical_data, scaler, columns_to_normalize):
    logging.info("Preprocessing data for training...")
    historical_data = add_features(historical_data)
    
    # Log a few rows to inspect the data
    logging.debug(f"Data after adding features: {historical_data.head()}")
    
    # Handle NaN and infinite values
    historical_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    historical_data.fillna(0, inplace=True)
    
    historical_data = clip_values(historical_data)
    
    # Standardize the data (Z-score normalization)
    historical_data[columns_to_normalize] = scaler.fit_transform(historical_data[columns_to_normalize])
    
    logging.debug(f"Data after normalization: {historical_data.head()}")
    
    return historical_data

# Function to initialize the environment
def initialize_environment(historical_data, stop_loss_pct, risk_per_trade):
    return TradingEnv(data=historical_data, stop_loss_pct=stop_loss_pct, risk_per_trade=risk_per_trade)

# Function to convert timestamps to numeric values
def convert_timestamps(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp() if isinstance(x, pd.Timestamp) else x)
    return df

# Function to check and log statistics of the input data
def log_data_statistics(data, name):
    logging.info(f"Statistics for {name}:")
    logging.info(f"  Mean: {np.mean(data)}")
    logging.info(f"  Min: {np.min(data)}")
    logging.info(f"  Max: {np.max(data)}")
    
    # Ensure that standard deviation is calculated with axis=0 to avoid future warnings
    logging.info(f"  Std Dev: {np.std(data, axis=0)}")

# Function to run DQN training
def run_dqn_training(env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size):
    log_text = ""
    total_rewards = []
    smoothed_reward = []
    smoothed_loss = []
    latest_loss_display = st.empty()
    total_reward_display = st.empty()
    progress_bar = st.progress(0)
    log_display = st.empty()

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (len(expected_features),))
        state = transform_state(state, scaler, columns_to_normalize, expected_features)
        log_data_statistics(state, "Initial State")
        total_reward = 0
        episode_loss = 0
        episode_trades = []

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Ensure reward is numeric and handle NaN rewards
            if pd.isna(reward) or np.isinf(reward):
                logging.warning(f"Reward became nan or inf at step {time_step} in episode {e+1}")
                reward = 0.0
            else:
                reward = float(reward)

            next_state = pd.DataFrame([next_state], columns=expected_features)
            next_state = convert_timestamps(next_state)  # Convert timestamps to numeric values
            next_state = next_state.apply(pd.to_numeric, errors='coerce').fillna(0)
            log_data_statistics(next_state, "Next State")

            if np.isnan(next_state.values).any() or np.isinf(next_state.values).any():
                logging.warning(f"Next state contains NaN or inf values at step {time_step} in episode {e+1}.")
                next_state = np.nan_to_num(next_state.values.flatten())

            next_state = np.reshape(next_state, (len(expected_features),))
            next_state = transform_state(next_state, scaler, columns_to_normalize, expected_features)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            original_close = env.data.iloc[env.current_step]['Close']
            episode_trades.append((env.current_step, action, original_close))

            if done:
                break

        smoothed_reward.append(smooth([total_reward])[-1])
        smoothed_loss.append(smooth([episode_loss])[-1])

        log_text += f"Episode {e + 1}:\nTotal Reward: {total_reward}\nEpisode Loss: {episode_loss}\nTrades:\n"
        for step, action, price in episode_trades:
            action_str = "Buy" if action == 1 else "Sell" if action == 2 else "Sit"
            log_text += f"  Step {step}: {action_str} at price {price:.6f}\n"
        log_text += "\n"

        if e % 10 == 0:
            gc.collect()

        log_display.text_area("Training Log", log_text, height=150, key=f"log_text_area_{e}")

        # Update memory and CPU usage metrics
        update_metrics()

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-50:])

        if len(agent.memory) > batch_size * 5:
            try:
                loss = agent.train_experience_replay()  # Use the existing method in your Agent class
                if np.isnan(loss).any() or np.isinf(loss).any():
                    logging.error(f"Invalid loss value encountered: {loss}. Skipping this batch.")
                    loss = 0.0
                episode_loss += loss
                latest_loss_display.metric("Latest Loss", round(smoothed_loss[-1], 4))
                log_text += f"  Loss: {loss}\n"
            except Exception as err:
                st.error(f"Error during training: {err}")
                break

        total_reward_display.metric("Total Reward", round(total_reward, 4))
        progress_bar.progress((e + 1) / episodes)

        if e % 50 == 0:
            agent.save(e)

    st.subheader("Training Log")
    st.expander("Training Log", expanded=True).text_area("Training Log", log_text, height=150, key="final_log_text")
    st.success("Training Completed!")

# Function to run Rainbow DQN training
def run_rainbow_dqn_training(env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size):
    log_text = ""
    total_rewards = []
    smoothed_reward = []
    smoothed_loss = []
    latest_loss_display = st.empty()
    total_reward_display = st.empty()
    progress_bar = st.progress(0)
    log_display = st.empty()

    for e in range(episodes):
        logging.info(f"Starting Rainbow DQN episode {e+1}")
        state = env.reset()
        state = np.reshape(state, (len(expected_features),))
        state = transform_state(state, scaler, columns_to_normalize, expected_features)
        log_data_statistics(state, "Initial State")
        total_reward = 0
        episode_loss = 0
        episode_trades = []

        for time_step in range(500):
            action = agent.act(state)  # Assuming agent has a Rainbow-specific act method
            next_state, reward, done, _ = env.step(action)

            if pd.isna(reward) or np.isinf(reward):
                logging.warning(f"Reward became NaN or Inf at step {time_step} in episode {e+1}")
                reward = 0.0
            else:
                reward = float(reward)

            next_state = pd.DataFrame([next_state], columns=expected_features)
            next_state = convert_timestamps(next_state)
            next_state = next_state.apply(pd.to_numeric, errors='coerce').fillna(0)
            log_data_statistics(next_state, "Next State")

            if np.isnan(next_state.values).any() or np.isinf(next_state.values).any():
                logging.warning(f"Next state contains NaN or Inf values at step {time_step} in episode {e+1}")
                next_state = np.nan_to_num(next_state.values.flatten())

            next_state = np.reshape(next_state, (len(expected_features),))
            next_state = transform_state(next_state, scaler, columns_to_normalize, expected_features)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            original_close = env.data.iloc[env.current_step]['Close']
            episode_trades.append((env.current_step, action, original_close))

            if done:
                logging.info(f"Episode {e+1} completed at time step {time_step}")
                break

        smoothed_reward.append(smooth([total_reward])[-1])
        smoothed_loss.append(smooth([episode_loss])[-1])

        log_text += f"Episode {e + 1}:\nTotal Reward: {total_reward}\nEpisode Loss: {episode_loss}\nTrades:\n"
        for step, action, price in episode_trades:
            action_str = "Buy" if action == 1 else "Sell" if action == 2 else "Sit"
            log_text += f"  Step {step}: {action_str} at price {price:.6f}\n"
        log_text += "\n"

        if e % 10 == 0:
            gc.collect()

        log_display.text_area("Training Log", log_text, height=150, key=f"log_text_area_{e}")

        # Update memory and CPU usage metrics
        update_metrics()

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-50:])

        if len(agent.memory) > batch_size * 5:
            try:
                loss = agent.train_experience_replay()  # Assuming a Rainbow-specific experience replay
                if np.isnan(loss).any() or np.isinf(loss).any():
                    logging.error(f"Invalid loss value encountered: {loss}. Skipping this batch.")
                    loss = 0.0
                episode_loss += loss
                latest_loss_display.metric("Latest Loss", round(smoothed_loss[-1], 4))
                log_text += f"  Loss: {loss}\n"
            except Exception as err:
                st.error(f"Error during training: {err}")
                break

        total_reward_display.metric("Total Reward", round(total_reward, 4))
        progress_bar.progress((e + 1) / episodes)

        if e % 50 == 0:
            agent.save(e)

    st.subheader("Training Log")
    st.expander("Training Log", expanded=True).text_area("Training Log", log_text, height=150, key="final_log_text")
    st.success("Training Completed!")
    logging.info("Rainbow DQN training completed")

# Function to run A3C training
def run_a3c_training(env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size):
    log_text = ""
    total_rewards = []
    smoothed_reward = []
    smoothed_loss = []
    latest_loss_display = st.empty()
    total_reward_display = st.empty()
    progress_bar = st.progress(0)
    log_display = st.empty()

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, len(expected_features)))  # Add batch dimension

        state = scaler.transform(state)

        total_reward = 0
        episode_loss = 0
        episode_trades = []

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            reward = float(reward) if not pd.isna(reward) and not np.isinf(reward) else 0.0

            next_state = pd.DataFrame([next_state], columns=expected_features)
            next_state = convert_timestamps(next_state)  # Convert timestamps to numeric values
            next_state = next_state.apply(pd.to_numeric, errors='coerce').fillna(0)

            if np.isnan(next_state.values).any() or np.isinf(next_state.values).any():
                logging.warning(f"Next state contains NaN or inf values at step {time_step} in episode {e+1}.")
                next_state = np.nan_to_num(next_state.values.flatten())

            next_state = np.reshape(next_state, (1, len(expected_features)))
            next_state = scaler.transform(next_state)

            discounted_reward = reward + agent.agent.gamma * agent.agent.critic.predict(next_state) if not done else reward
            advantage = discounted_reward - agent.agent.critic.predict(state)

            action_one_hot = np.zeros(agent.action_size)
            action_one_hot[action] = 1

            actor_loss = agent.agent._actor_optimizer(state, action_one_hot, advantage)
            critic_loss = agent.agent._critic_optimizer(state, discounted_reward)

            episode_loss += actor_loss.numpy() + critic_loss.numpy()
            state = next_state
            total_reward += reward

            original_close = env.data.iloc[env.current_step]['Close']
            episode_trades.append((env.current_step, action, original_close))

            if done:
                break

        smoothed_reward.append(smooth([total_reward])[-1])
        smoothed_loss.append(smooth([episode_loss])[-1])

        log_text += f"Episode {e + 1}:\nTotal Reward: {total_reward}\nEpisode Loss: {episode_loss}\nTrades:\n"
        for step, action, price in episode_trades:
            action_str = "Buy" if action == 1 else "Sell" if action == 2 else "Sit"
            log_text += f"  Step {step}: {action_str} at price {price:.6f}\n"
        log_text += "\n"

        if e % 10 == 0:
            gc.collect()

        log_display.text_area("Training Log", log_text, height=150, key=f"log_text_area_{e}")

        # Update memory and CPU usage metrics
        update_metrics()

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-50:])

        latest_loss_display.metric("Latest Loss", round(smoothed_loss[-1], 4))
        total_reward_display.metric("Total Reward", round(total_reward, 4))
        progress_bar.progress((e + 1) / episodes)

        if e % 50 == 0:
            agent.save(e)

    st.subheader("Training Log")
    st.expander("Training Log", expanded=True).text_area("Training Log", log_text, height=150, key="final_log_text")
    st.success("Training Completed!")

# Modify run_training to include Rainbow DQN
def run_training(strategy, env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size):
    # Normalize strategy name
    strategy = strategy.lower().replace(" ", "_")
    logging.info(f"Selected strategy: {strategy}")
    
    if strategy == "dqn":
        logging.info("Running DQN training...")
        run_dqn_training(env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size)
    elif strategy == "a3c":
        logging.info("Running A3C training...")
        run_a3c_training(env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size)
    elif strategy == "rainbow_dqn":
        logging.info("Running Rainbow DQN training...")
        run_rainbow_dqn_training(env, agent, scaler, columns_to_normalize, expected_features, episodes, batch_size)
    else:
        st.error(f"Unknown strategy: {strategy}")
        logging.error(f"Unknown strategy: {strategy}")
