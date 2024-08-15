import streamlit as st
import pandas as pd

def configure_page():
    st.set_page_config(
        page_title="Trading Simulation",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def sidebar_inputs():
    ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    episodes = st.sidebar.number_input("Episodes", value=1000, min_value=1)
    batch_size = st.sidebar.number_input("Batch Size", value=32, min_value=1)
    strategy = st.sidebar.selectbox("Choose Strategy", ("DQN", "A3C", "Rainbow DQN"))

    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.00001, max_value=0.001, value=0.0005, step=0.00001)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", min_value=0.995, max_value=0.9999, value=0.998, step=0.0001)
    stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
    risk_per_trade = st.sidebar.slider("Risk Per Trade Percentage", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    actor_lr = st.sidebar.slider("Actor Learning Rate", min_value=0.00001, max_value=0.001, value=0.0001, step=0.00001)
    critic_lr = st.sidebar.slider("Critic Learning Rate", min_value=0.00001, max_value=0.001, value=0.001, step=0.00001)

    return (ticker, start_date, end_date, episodes, batch_size, strategy, 
            learning_rate, epsilon_decay, stop_loss_pct, risk_per_trade, 
            actor_lr, critic_lr)

def sidebar_simulation_controls():
    st.sidebar.subheader("Run Simulation")
    simulation_type = st.sidebar.radio("Choose Simulation Type", ("Training", "Backtest", "Paper Trading", "Live Trading", "All"))
    return simulation_type

def show_logs():
    with open('data/training_log.log', 'r') as f:
        st.text_area("Logs", f.read(), height=300)

def show_run_button():
    """Displays the Run button on the sidebar."""
    if "run_button_pressed" not in st.session_state:
        st.session_state.run_button_pressed = False

    if not st.session_state.run_button_pressed:
        return st.sidebar.button("Run")
    else:
        return False

def handle_run_button():
    if st.session_state.run_button_pressed:
        st.sidebar.write("Running...")  # Placeholder for actual logic
        # Place your actual code to handle running the simulation here

def main():
    configure_page()
    simulation_type = sidebar_simulation_controls()

    if show_run_button():
        st.session_state.run_button_pressed = True
        handle_run_button()

    if st.session_state.run_button_pressed:
        # Logic to execute after pressing Run
        st.sidebar.write("Simulation Type: ", simulation_type)
        # Include your simulation execution code here.

if __name__ == "__main__":
    main()
