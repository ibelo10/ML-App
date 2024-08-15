import random
from collections import deque
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model, clone_model
from keras.layers import Dense, Lambda, Input
from keras.optimizers import Adam
import keras.backend as K
import gym
import gym.spaces
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning"""
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

class TradingEnv(gym.Env):
    """A custom environment for DQN trading using historical data."""
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # [sit, buy, sell]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = self._next_observation()
        reward = np.random.random()
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def build_dueling_model(state_size, action_size):
    inputs = Input(shape=(state_size,))
    layer1 = Dense(128, activation='relu')(inputs)
    layer2 = Dense(256, activation='relu')(layer1)
    layer3 = Dense(256, activation='relu')(layer2)
    layer4 = Dense(128, activation='relu')(layer3)

    value = Dense(1)(layer4)
    advantage = Dense(action_size)(layer4)
    advantage_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
    q_values = Lambda(lambda x: x[0] + (x[1] - x[2]), output_shape=(action_size,))([value, advantage, advantage_mean])

    model = Model(inputs, q_values)
    model.compile(loss=huber_loss, optimizer=Adam(learning_rate=0.001))
    return model

class DuelingDQNAgent:
    def __init__(self, state_size, action_size, model_name="dueling_dqn"):
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = build_dueling_model(state_size, action_size)
        self.target_model = build_dueling_model(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        for state, action, reward, next_state, done in mini_batch:
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = target
            X_train.append(state[0])
            y_train.append(q_values[0])

        loss = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0).history["loss"][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def save(self, episode):
        model_path = f"models/{self.model_name}_{episode}.keras"
        self.model.save(model_path)
        print(f"Model saved at {model_path}")

    def load(self):
        model_path = f"models/{self.model_name}.keras"
        try:
            model = load_model(model_path, custom_objects={"huber_loss": huber_loss})
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return self.model
