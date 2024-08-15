import logging
import numpy as np
from models.tensorflow_model import train_model
from data.data_fetching import DataFetcher
from backtesting import Backtest
from backtest_strategy import MyStrategy
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter
from models.model_dqn import Agent, TradingEnv  # Updated import

# Custom plot function
def custom_plot(bt):
    stats = bt._results
    equity_curve = stats['_equity_curve']

    p = figure(x_axis_type="datetime", title="Backtest Results", height=400, width=800)
    p.xaxis.formatter = DatetimeTickFormatter(days='%d %b')

    # Add equity line
    p.line(equity_curve.index, equity_curve['Equity'], line_width=2, legend_label="Equity")

    # Add drawdown line
    p.line(equity_curve.index, equity_curve['DrawdownPct'], line_width=2, color='red', legend_label="Drawdown")

    p.legend.location = "top_left"
    show(p)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Fetch data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data()

    # Train the supervised learning model
    best_model_path = train_model(data, logger)
    print(f"Training completed. Best model saved at: {best_model_path}")

    # Load historical data for backtesting
    historical_data = data_fetcher.fetch_data(ticker="BTC-USD", start_date="2020-01-01", end_date="2023-12-31")
    historical_data = historical_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    historical_data.set_index('Date', inplace=True)

    # Run backtest with TensorFlow model
    bt = Backtest(historical_data, MyStrategy, cash=100000, commission=.002)
    stats = bt.run()
    custom_plot(bt)
    print(stats)

    # Train DQN agent
    state_size = historical_data.shape[1]
    action_size = 3  # [sit, buy, sell]
    agent = Agent(state_size, strategy="t-dqn")  # Initialize the DQN agent
    env = TradingEnv(data=historical_data)  # Initialize the environment with historical data

    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                loss = agent.train_experience_replay(batch_size)
                print(f"Loss: {loss}")

        if e % 50 == 0:
            agent.save(e)
