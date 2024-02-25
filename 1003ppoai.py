import gym
import numpy as np
from stable_baselines import PPO2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq import DQN
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy

import tensorflow.compat.v1.layers as tf_layers
import pandas as pd
# define the environment
class CryptoEnvironment(gym.Env):
    def __init__(self, data):
        self.data = data
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(data.columns),))
        self.reset()

    def step(self, action):
        reward = 0
        done = False
        info = {}
        if action == 0:
            self.position = 0
        elif action == 1:
            if self.position == 0:
                self.position = 1
            else:
                reward = self.data['close'][self.current_step] - self.data['close'][self.previous_step]
                self.profit += reward
                if self.profit < -100:
                    done = True
                    reward = -100
                elif self.profit > 100:
                    done = True
                    reward = 100
        self.previous_step = self.current_step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        obs = self._get_obs()
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.previous_step = 0
        self.profit = 0
        self.position = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.data.iloc[self.current_step])

# define the neural network model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Load the dataset
data = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
# Rename the columns
data = pd.DataFrame(data)
"""print(df.iloc[0,:])
exit()"""
data.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot','tbbav','tbqav', 'ignore']

# load and preprocess the data
data = data# load your crypto data here
data['close'] = data['close'].pct_change()
data.dropna(inplace=True)

# create the environment and the neural network model
env = DummyVecEnv([lambda: CryptoEnvironment(data)])
model = PPO2(build_model('env.observation_space.shape'), env, verbose=1)

# train the model
model.learn(total_timesteps=100000)

# use the model to make buy or sell decisions
obs = env.reset()
for i in range(len(data)):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break