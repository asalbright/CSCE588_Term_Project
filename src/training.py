###############################################################################
# training_linear.py
#
# A script for training the pogo_sick env varying the weights on the reward
# 
#
# Created: 03/29/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################

import os
from pathlib import Path
import time
import numpy as np 
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import torch
import stable_baselines3
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.monitor import Monitor 

from pogo_stick_jumping.nonlinear_env import PogoNonlinearEnv
from custom_callbacks import LogMechParamsCallback

# Make sure a GPU is available for torch to utilize
try:
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.cuda.is_available())
except:
    print('No GPU detected')

# Model Parameters
REWARD_FUNCTION = 'RewardHeight'    # 'RewardHeight', 'RewardSpecifiedHeight', 'RewardEfficiency'
MODEL_TYPE = 'PPO'
LEARNING_RATE = 0.02
GAMMA = 0.99
ROLLOUT = 10           # For TD3
TOTAL_SIMS = 1000

batch_size=64 #for PPO
n_steps=50 #for PPO

# Design space parameters
SPRING_K = 5760
VARIANCE = 0.9
MIN_SPRING_K = SPRING_K - VARIANCE * SPRING_K
MAX_SPRING_K = SPRING_K + VARIANCE * SPRING_K  
ZETA = 0.01
MIN_ZETA = ZETA - VARIANCE * ZETA
MAX_ZETA = ZETA + VARIANCE * ZETA

# Training Parameters
NUM_TRIALS = 10    # number of seeds to test
MULTIPROCESS = True    # do you want to multiprocess the different seeds?
NUM_CORES = 2        # How many cores do you want to use?

# Set up the training seeds
INITIAL_SEED = 70503
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=10000, size=(NUM_TRIALS))

# Set up the training save paths
data_name = f'{REWARD_FUNCTION}'
save_path = Path.cwd()
save_path = save_path.joinpath(f'train_{data_name}')
logs_path = save_path.joinpath('logs')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# Function to train the model
def train_agents(seed=12345):

    # Set up training env
    env = PogoNonlinearEnv(ep_steps=1,
                           reward_function=REWARD_FUNCTION,  
                           min_spring_k=MIN_SPRING_K,
                           max_spring_k=MAX_SPRING_K,
                           min_zeta=MIN_ZETA,
                           max_zeta=MAX_ZETA,
                           model_type=MODEL_TYPE)    

    # set the trial seed for use during training
    trial_seed = int(seed)
    env.seed(seed=trial_seed)

    # wrap the env in modified monitor which plots to tensorboard the jumpheight
    env = Monitor(env)

    # create the model
    # Set the model
    if MODEL_TYPE == 'TD3':
        buffer_size = TOTAL_SIMS + 1
        model = TD3("MlpPolicy", 
                    env, 
                    verbose=1, 
                    tensorboard_log=logs_path, 
                    buffer_size=buffer_size, 
                    learning_starts=ROLLOUT, 
                    seed=trial_seed, 
                    gamma=GAMMA)
                    
    elif MODEL_TYPE == 'PPO':
        model = PPO("MlpPolicy",
                    env,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    learning_rate=LEARNING_RATE,
                    clip_range=0.2,
                    verbose=1,
                    tensorboard_log=logs_path,
                    seed=trial_seed)

    # Instantiate a callback that logs parameters to tensorboard
    callback = LogMechParamsCallback()
    # TODO: Change log the name to whatever you want
    # OPEN tensorboard with the following bash command: tensorboard --logdir ./logs/
    #model.learn(total_timesteps=TOTAL_SIMS, callback=callback, tb_log_name=f'{MODEL_TYPE}_{LEARNING_RATE}_{batch_size}_{n_steps}_{int(seed)}')
    model.learn(total_timesteps=TOTAL_SIMS, callback=callback, tb_log_name=f'{MODEL_TYPE}_{LEARNING_RATE}_{n_steps}_{int(seed)}')

# Set up the training multiprocess
def multi_process(function, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(function, i)

if __name__ == "__main__":
    if MULTIPROCESS:
        start = time.time()
        multi_process(train_agents, TRIAL_SEEDS, NUM_CORES)
        end = time.time()
        total = end - start
        print(f'Total Time: {total}')
    else:
        train_agents() # Comment this out if you want to use multiprocessing
