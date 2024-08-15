import random
from collections import deque
import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential, load_model, clone_model, Model
from keras.layers import Input, Dense, Layer, Subtract, Add, Dropout
from keras.optimizers import Adam
import gym
import gym.spaces
import logging
import threading

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.log', mode='w'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

# Custom Huber Loss for Q-learning
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

# Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, stop_loss_pct=0.02, risk_per_trade=0.01):
        super(TradingEnv, self).__init__()
        logging.info("Initializing Trading Environment.")
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # [sit, buy, sell]
        self.state_size = self.data.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )
        self.position = 0  # 1: long, -1: short, 0: no position
        self.cash = initial_balance  # Initial cash balance
        self.shares_held = 0  # Number of shares held
        self.initial_cash = initial_balance  # To keep track of initial cash for reward calculation
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade = risk_per_trade
        self.stop_loss_price = None

    def reset(self):
        logging.info("Resetting environment.")
        self.current_step = 0
        self.position = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.stop_loss_price = None
        self.reward_ma = 0  # Reset moving average reward
        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        prev_cash = self.cash
        prev_shares_held = self.shares_held

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            self.current_step = len(self.data) - 1  # Prevent out-of-bounds
        else:
            done = False

        curr_close = self.data.iloc[self.current_step]['Close']

        reward = 0
        if action == 1:  # Buy
            logging.debug("Executing Buy action.")
            if self.cash > 0:  # Ensure there's enough cash to buy
                position_size = (self.risk_per_trade * self.cash) / (self.stop_loss_pct * curr_close)
                shares_bought = position_size
                self.shares_held += shares_bought
                self.cash -= shares_bought * curr_close
                self.position = 1
                self.stop_loss_price = curr_close * (1 - self.stop_loss_pct)

        elif action == 2:  # Sell
            logging.debug("Executing Sell action.")
            if self.shares_held > 0:  # Ensure there are shares to sell
                self.cash += self.shares_held * curr_close
                reward = self.cash - self.initial_cash  # Profit since beginning
                self.shares_held = 0
                self.position = -1
                self.stop_loss_price = None

        portfolio_value = self.cash + (self.shares_held * curr_close)
        reward = (portfolio_value - self.initial_cash) / self.initial_cash  # Reward based on portfolio return

        # Implement a moving average reward smoothing
        self.reward_ma = (0.9 * self.reward_ma + 0.1 * reward) if hasattr(self, 'reward_ma') else reward
        reward = self.reward_ma

        # Normalize the reward to ensure it stays within a reasonable range
        reward = np.clip(reward, -1, 1)

        # Check for NaN in the reward
        if np.isnan(reward):
            logging.warning(f"Reward became nan at step {self.current_step}")
            reward = 0.0  # Reset NaN rewards to zero

        # Penalty for holding too long without significant movement
        if action == 0 and abs(portfolio_value - self.initial_cash) < 0.01:
            reward -= 0.01  # Small penalty for not making significant decisions

        # Dynamic stop-loss or take-profit strategy
        if self.stop_loss_price and (curr_close <= self.stop_loss_price):
            reward -= 0.05  # Penalty for hitting stop-loss
            done = True

        # Provide a small reward for making any action other than 'sit'
        if action != 0:
            reward += 0.01

        next_state = self._next_observation()
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Main Agent Class to Select Strategy
class Agent:
    """Unified Agent for various strategies: DQN, Rainbow DQN, A3C"""
    def __init__(self, state_size, action_size, strategy="dqn", **kwargs):
        self.strategy = strategy.lower()  # Normalize strategy to lowercase
        logging.info(f"Initializing Agent with strategy: {self.strategy}")
        self.state_size = state_size
        self.action_size = action_size

        if self.strategy in ["dqn", "dueling dqn"]:
            logging.info("Initializing Dueling DQN Agent...")
            self.agent = DuelingDQNAgent(self.state_size, self.action_size, **kwargs)
        elif self.strategy in ["rainbow", "rainbow dqn"]:
            logging.info("Initializing Rainbow DQN Agent...")
            self.agent = RainbowDQNAgent(self.state_size, self.action_size, **kwargs)
        elif self.strategy == "a3c":
            logging.info("Initializing A3C Agent...")
            actor_lr = kwargs.pop('actor_lr', 0.0001)
            critic_lr = kwargs.pop('critic_lr', 0.001)
            kwargs.pop('learning_rate', None)
            kwargs.pop('epsilon_decay', None)
            self.agent = A3CAgent(self.state_size, self.action_size, actor_lr=actor_lr, critic_lr=critic_lr, **kwargs)
        else:
            logging.error(f"Unknown strategy: {self.strategy}")
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def remember(self, *args, **kwargs):
        if hasattr(self.agent, 'remember'):
            return self.agent.remember(*args, **kwargs)
        else:
            raise NotImplementedError(f"The 'remember' method is not implemented for {self.strategy}.")

    def act(self, *args, **kwargs):
        return self.agent.act(*args, **kwargs)

    def train_experience_replay(self, *args, **kwargs):
        if hasattr(self.agent, 'train_experience_replay'):
            return self.agent.train_experience_replay(*args, **kwargs)
        else:
            raise NotImplementedError(f"The 'train_experience_replay' method is not implemented for {self.strategy}.")

    def save(self, *args, **kwargs):
        return self.agent.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self.agent.load(*args, **kwargs)

    @property
    def gamma(self):
        return getattr(self.agent, 'gamma', None)  # Return gamma if it exists in the agent

    @property
    def critic(self):
        return getattr(self.agent, 'critic', None)  # Return critic if it exists in the agent

    @property
    def memory(self):
        if hasattr(self.agent, 'memory'):
            return self.agent.memory
        else:
            raise AttributeError(f"The 'memory' attribute is not available for {self.strategy}.")

class AdvantageCenteringLayer(Layer):
    def __init__(self, **kwargs):
        super(AdvantageCenteringLayer, self).__init__(**kwargs)

    def call(self, advantage):
        mean_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return advantage - mean_advantage
    
class DuelingDQNAgent:
    def __init__(self, state_size, action_size, memory_size=10000, batch_size=32, gamma=0.99,
                 learning_rate=0.00025, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, tau=0.125):
        logging.info("Initializing Dueling DQN Agent.")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.tau = tau  # For soft update

        # Dueling Networks
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)

        # Dueling DQN: Separate value and advantage streams
        value_fc = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear')(value_fc)

        adv_fc = Dense(64, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear')(adv_fc)

        # Apply the custom AdvantageCenteringLayer
        advantage_centered = AdvantageCenteringLayer()(advantage)
        q_values = Add()([value, advantage_centered])

        model = Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=huber_loss)
        return model
    
    def act(self, state, is_eval=False):
        if not is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        try:
            # Sample mini-batch from memory with prioritized experience replay
            priorities = []
            for experience in self.memory:
                state, action, reward, next_state, done = experience
                target_q = reward
                if not done:
                    target_q += self.gamma * np.max(self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                current_q = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0][action]
                priorities.append(abs(target_q - current_q))

            sampling_probabilities = np.array(priorities) / np.sum(priorities)
            mini_batch = random.choices(self.memory, weights=sampling_probabilities, k=self.batch_size)

            states, actions, rewards, next_states, dones = map(np.array, zip(*mini_batch))

            # Predict Q-values for both current and next states
            target = self.model.predict(states, verbose=0)
            target_next = self.model.predict(next_states, verbose=0)
            target_val = self.target_model.predict(next_states, verbose=0)

            for i in range(self.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    best_action = np.argmax(target_next[i])
                    target[i][actions[i]] = rewards[i] + self.gamma * target_val[i][best_action]

            # Gradient clipping
            with tf.GradientTape() as tape:
                predictions = self.model(states, training=True)
                loss = tf.keras.losses.MSE(target, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
            self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

            # Soft update of target network
            tau = 0.001
            for t, e in zip(self.target_model.trainable_variables, self.model.trainable_variables):
                t.assign(t * (1.0 - tau) + e * tau)

            # Anneal epsilon for exploration-exploitation trade-off
            if self.epsilon > self.epsilon_min:
                self.epsilon *= np.exp(-self.epsilon_decay)

            # Log the loss and epsilon
            logging.info(f"Training loss: {loss.numpy()}, Epsilon: {self.epsilon}")

            return loss.numpy()

        except Exception as e:
            logging.error(f"Error during training replay: {e}")
            return 0.0


    def save(self, episode):
        # Ensure the directory exists
        model_dir = "KerasModels"
        os.makedirs(model_dir, exist_ok=True)
        
        # Define the model save path
        model_path = os.path.join(model_dir, f"dueling_dqn_{episode}.keras")
        
        # Save the model
        self.model.save(model_path)
        logging.info(f"Model saved at {model_path}")

    def load(self, model_path="KerasModels/dueling_dqn.keras"):
        try:
            # Load the model from the specified path
            self.model = load_model(model_path, custom_objects={"huber_loss": huber_loss})
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
# Rainbow DQN Implementation
class RainbowDQNAgent(Agent):
    def __init__(self, state_size, action_size, memory_size=10000, batch_size=32, gamma=0.99,
                 learning_rate=0.00025, epsilon_decay=0.995, epsilon_min=0.01, tau=0.125, 
                 prioritized_replay=True, n_step=3):
        super().__init__(state_size, action_size)
        logging.info("Initializing Rainbow DQN Agent.")
        
        # Set the memory using the setter
        self._memory = deque(maxlen=memory_size)
        self.prioritized_replay = prioritized_replay
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.tau = tau  # For soft update
        
        # Multi-step learning
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Double Q-learning and Dueling Networks
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    @property
    def memory(self):
        return self._memory
    
    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @memory.setter
    def memory(self, value):
        self._memory = value

    def _build_model(self):
        # Model building logic remains the same
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)

        value_fc = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear')(value_fc)

        adv_fc = Dense(64, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear')(adv_fc)

        advantage_centered = AdvantageCenteringLayer()(advantage)
        q_values = Add()([value, advantage_centered])

        model = Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=huber_loss)
        return model

    def act(self, state, is_eval=False):
        if not is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample mini-batch from memory with prioritized experience replay
        priorities = []
        for experience in self.memory:
            state, action, reward, next_state, done = experience
            target_q = reward
            if not done:
                target_q += self.gamma * np.max(self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
            current_q = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0][action]
            priorities.append(abs(target_q - current_q))

        sampling_probabilities = np.array(priorities) / np.sum(priorities)
        mini_batch = random.choices(self.memory, weights=sampling_probabilities, k=self.batch_size)
        
        states, actions, rewards, next_states, dones = map(np.array, zip(*mini_batch))
        
        # Predict Q-values for both current and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * target_val[i][best_action]

        # Train the model and get loss
        loss = self.model.fit(states, target, epochs=1, verbose=0).history['loss'][0]

        # Soft update of target network
        tau = 0.001
        for t, e in zip(self.target_model.trainable_variables, self.model.trainable_variables):
            t.assign(t * (1.0 - tau) + e * tau)

        # Anneal epsilon for exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= np.exp(-self.epsilon_decay)

        # Log the loss and epsilon
        logging.info(f"Training loss: {loss}, Epsilon: {self.epsilon}")

        return loss

    def save(self, episode):
        model_path = f"models/rainbow_dqn_{episode}.keras"
        self.model.save(model_path)
        logging.info(f"Model saved at {model_path}")

    def load(self, model_path="models/rainbow_dqn.keras"):
        try:
            self.model = load_model(model_path, custom_objects={"huber_loss": huber_loss})
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")

# A3C Implementation
class A3CAgent:
    def __init__(self, state_size, action_size, actor_lr=0.00001, critic_lr=0.0001, gamma=0.99, n_threads=8):
        logging.info("Initializing A3C Agent.")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.n_threads = n_threads

    def _build_actor(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        output = Dense(self.action_size, activation='softmax')(x)
        
        model = Model(inputs, output)
        return model

    def _build_critic(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs, output)
        return model
    
    def act(self, state):
        """Chooses an action based on the current policy (actor network)."""
        state = np.reshape(state, [1, self.state_size])
        
        # Check if the state contains NaN values
        if np.any(np.isnan(state)):
            logging.error("State contains NaN values, returning random action")
            return np.random.choice(self.action_size)
        
        policy = self.actor.predict(state, verbose=0).flatten()
        
        # Ensure the policy is a valid probability distribution
        if np.any(np.isnan(policy)) or np.any(policy < 0) or not np.isclose(np.sum(policy), 1.0):
            logging.error(f"Invalid policy detected: {policy}, returning random action")
            return np.random.choice(self.action_size)

        return np.random.choice(self.action_size, p=policy)

    def _actor_optimizer(self, state, action, advantages):
        with tf.GradientTape() as tape:
            policy = self.actor(state, training=True)
            action_prob = tf.reduce_sum(action * policy, axis=1)
            eligibility = tf.math.log(action_prob + 1e-10) * advantages
            loss = -tf.reduce_mean(eligibility)
        
        grads = tape.gradient(loss, self.actor.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]  # Gradient clipping
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss

    def _critic_optimizer(self, state, discounted_reward):
        with tf.GradientTape() as tape:
            value = self.critic(state, training=True)
            loss = tf.reduce_mean(tf.square(discounted_reward - value))
        
        grads = tape.gradient(loss, self.critic.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]  # Gradient clipping
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    def train(self, env):
        logging.info("Starting A3C training with multiple threads.")
        workers = [Worker(self.actor, self.critic, env, self.actor_optimizer, self.critic_optimizer, self.gamma) for _ in range(self.n_threads)]
        for worker in workers:
            worker.start()

    def save(self, episode):
        actor_path = f"models/a3c_actor_{episode}.keras"
        critic_path = f"models/a3c_critic_{episode}.keras"
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        logging.info(f"Actor and Critic models saved at {actor_path} and {critic_path}")

    def load(self, actor_path="models/a3c_actor.keras", critic_path="models/a3c_critic.keras"):
        try:
            self.actor = tf.keras.models.load_model(actor_path)
            self.critic = tf.keras.models.load_model(critic_path)
            logging.info(f"Actor and Critic models loaded from {actor_path} and {critic_path}")
        except Exception as e:
            logging.error(f"Failed to load actor or critic model: {e}")

class Worker(threading.Thread):
    def __init__(self, actor, critic, env, actor_optimizer, critic_optimizer, gamma):
        threading.Thread.__init__(self)
        self.actor = actor
        self.critic = critic
        self.env = env
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
    
    def run(self):
        logging.info("Worker thread started.")
        state = self.env.reset()
        done = False
        while not done:
            action = self._choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            discounted_reward = reward + self.gamma * self.critic.predict(next_state) if not done else reward
            advantage = discounted_reward - self.critic.predict(state)
            
            action_one_hot = np.zeros(self.actor.output_shape[-1])
            action_one_hot[action] = 1
            
            self.actor_optimizer([state, action_one_hot, advantage])
            self.critic_optimizer([state, discounted_reward])
            
            state = next_state
    
    def _choose_action(self, state):
        policy = self.actor.predict(state)[0]
        return np.random.choice(self.actor.output_shape[-1], p=policy)
