from agents import matrix_agent, agentc1, agentc2, agentc3, agentc5, agentc7, agentc9, agentc11

import os
import random
import numpy as np
import pandas as pd

from kaggle_environments import make, evaluate

import tensorflow as tf

import sys
sys.stdout = open('training_out.log', 'w')
sys.stderr = open('training_err.log', 'w')

assert(tf.test.is_gpu_available())

from stable_baselines.bench import Monitor 
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import PPO2 
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.callbacks import EvalCallback

from numpy.random import choice
from connect4gym import ConnectFourGym, SaveBestModelCallback
from common import board_flip, get_win_percentages_and_score



    
# Create ConnectFour environment
env = ConnectFourGym([matrix_agent, 'random', agentc1, agentc2, agentc3, agentc5])

# Create directory for logging training information
log_dir = "logtf1/"
os.makedirs(log_dir, exist_ok=True)

# Logging progress
monitor_env = Monitor(env, log_dir, allow_early_resets=True)

# Create a vectorized environment
vec_env = DummyVecEnv([lambda: monitor_env])

from tensorflow.layers import Dropout, BatchNormalization, Dense, Conv2D
import tensorflow as tf


"""
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 64,
    'num_channels': 64,
})
"""


NUM_CHANNELS = 64

BN1 = BatchNormalization()
BN2 = BatchNormalization()
BN3 = BatchNormalization()
BN4 = BatchNormalization()
BN5 = BatchNormalization()
BN6 = BatchNormalization()


CONV1 = Conv2D(NUM_CHANNELS, kernel_size=3, strides=1, padding='same')
CONV2 = Conv2D(NUM_CHANNELS, kernel_size=3, strides=1, padding='same')
CONV3 = Conv2D(NUM_CHANNELS, kernel_size=3, strides=1)
CONV4 = Conv2D(NUM_CHANNELS, kernel_size=3, strides=1)

FC1 = Dense(128)
FC2 = Dense(64)
FC3 = Dense(7)

DROP1 = Dropout(0.3)
DROP2 = Dropout(0.3)


# 6x7 input
# https://github.com/PaddlePaddle/PARL/blob/0915559a1dd1b9de74ddd2b261e2a4accd0cd96a/benchmark/torch/AlphaZero/submission_template.py#L496
def modified_cnn(inputs, **kwargs):
    relu = tf.nn.relu
    log_softmax = tf.nn.log_softmax
    
    
    layer_1_out = relu(BN1(CONV1(inputs)))
    layer_2_out = relu(BN2(CONV2(layer_1_out)))
    layer_3_out = relu(BN3(CONV3(layer_2_out)))
    layer_4_out = relu(BN4(CONV4(layer_3_out)))
    
    # 3 is width - 4 due to convolition filters, 2 is same for height
    flattened = tf.reshape(layer_4_out, [-1, NUM_CHANNELS * 3 * 2]) 
    
    layer_5_out = DROP1(relu(BN5(FC1(flattened))))
    layer_6_out = DROP2(relu(BN6(FC2(layer_5_out))))
    
    return log_softmax(FC3(layer_6_out))  

# https://www.kaggle.com/c/connectx/discussion/128591
class CustomCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn)

eval_callback = SaveBestModelCallback('RDaneelConnect4_', 1000, ['random', agentc1, agentc3, agentc5, matrix_agent])

model = PPO2(CustomCnnPolicy, vec_env, verbose=1)


# Train agent
model.learn(total_timesteps=30000000, callback=eval_callback)


def dqn_agent(obs, config):
    # Use the best model to select a column
    grid = board_flip(obs.mark, np.array(obs['board']).reshape(6,7,1))
    col, _ = model.predict(grid, deterministic=True)
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move. 
    if is_valid:
        return int(col)
    else:
        grid = grid.reshape(6, 7)
        #sleep(2)
        #print(f'Illegal move attempted! Move: {col}, Boardf:\n{grid}')
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
    
print('=' * 80)
print('R Daneel VS Random')
get_win_percentages_and_score(dqn_agent, 'random')
print('=' * 80)
print('R Daneel VS Heuristic')
get_win_percentages_and_score(dqn_agent, agentc1)
print('=' * 80)
print('R Daneel VS Minmax2')
get_win_percentages_and_score(dqn_agent, agentc2)
print('=' * 80)
print('R Daneel VS Minmax3')
get_win_percentages_and_score(dqn_agent, agentc3)
print('=' * 80)
print('R Daneel VS Minmax5')
get_win_percentages_and_score(dqn_agent, agentc5)
print('=' * 80)
print('R Daneel VS Minmax7')
get_win_percentages_and_score(dqn_agent, agentc7)
print('=' * 80)
print('R Daneel VS Matrix Agent')
get_win_percentages_and_score(dqn_agent, matrix_agent)