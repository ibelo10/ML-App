import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from models.model_dqn import Agent, TradingEnv
from data.data_fetching import DataFetcher

class TestAgent(unittest.TestCase):
    def setUp(self):
        # Set up the environment with real data
        data_fetcher = DataFetcher()
        self.mock_data = data_fetcher.fetch_data(ticker="BTC-USD", start_date="2022-01-01", end_date="2023-01-01")
        self.env = TradingEnv(data=self.mock_data)

        # Initialize the agent with a state size based on the environment's observation space
        self.agent = Agent(state_size=self.env.observation_space.shape[0], strategy="t-dqn", pretrained=False)

    def test_model_creation(self):
        # Check if the model is created successfully
        self.assertIsNotNone(self.agent.model)
        print("Model creation test passed.")

    def test_act(self):
        # Test the act method of the agent
        state = self.env.reset()
        action = self.agent.act(state)
        self.assertIn(action, range(self.env.action_space.n))
        print(f"Action taken: {action}")
    
    def test_training_step(self):
        # Add experience to the memory
        state = self.env.reset()
        action = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        self.agent.remember(state, action, reward, next_state, done)
        
        # Train on this experience
        loss = self.agent.train_experience_replay(batch_size=1)
        self.assertIsInstance(loss, float)
        print(f"Training step loss: {loss}")

    def test_save_and_load(self):
        # Save the model
        self.agent.save(episode=1)
        
        # Load the model
        loaded_model = self.agent.load()
        self.assertIsNotNone(loaded_model)
        print("Model save and load test passed.")

class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        # Set up the environment with real data
        data_fetcher = DataFetcher()
        self.mock_data = data_fetcher.fetch_data(ticker="BTC-USD", start_date="2022-01-01", end_date="2023-01-01")
        self.env = TradingEnv(data=self.mock_data)

    def test_reset(self):
        # Test the reset function of the environment
        initial_state = self.env.reset()
        self.assertIsInstance(initial_state, np.ndarray)
        print("Environment reset test passed.")

    def test_step(self):
        # Test the step function of the environment
        initial_state = self.env.reset()
        action = self.env.action_space.sample()
        next_state, reward, done, _ = self.env.step(action)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        print("Environment step test passed.")

    def test_full_episode(self):
        # Run a full episode in the environment
        state = self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.assertIsInstance(next_state, np.ndarray)
            self.assertIsInstance(reward, float)
        print("Full episode test passed.")

if __name__ == '__main__':
    unittest.main()
