#! /usr/bin/env python

###############################################################################
# nonlinear_env.py
#
# Defines a pogo stick jumping environment for use with the openAI Gym.
# This version has a continuous range of inputs for the mass accel. input
#
# Created: 02/03/21
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Modified:
#   * 
#
# TODO:
#   *
###############################################################################

import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import logging
import numpy as np
from scipy.integrate import solve_ivp
import datetime # for unique filenames
from pathlib import Path
#from stable_baselines3.common.logger import record
from pogo_stick_jumping.ODE_Nonlinear import PogoODEnonlinear

logger = logging.getLogger(__name__)

class PogoNonlinearEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self,
                 ep_steps=1,
                 sim_step_size=0.001,
                 sim_duration=1, 
                 reward_function='RewardHeight',
                 specified_height=0.1,  
                 min_spring_k=5000,
                 max_spring_k=7200,
                 min_zeta=0.0025,
                 max_zeta=0.0175,
                 model_type="TD3"):
        """
        Initialize with the parameters to match earlier papers from team at 
        Georgia Tech and Dr. Vaughan
        """
        self.gravity = 0.276 * 9.81      # accel. due to gravity (m/s^2)
        self.m_rod = 0.175               # mass of the pogo-stick rod (kg)
        self.m_act = 1.003               # mass of the pogo-stick rod (kg)
        self.mass = self.m_rod + self.m_act  # total mass
        self.f = 11.13                   # natural freq. (rad)
        self.wn = self.f * (2 * np.pi)   # Robot frequency (rad/s)
        self.zeta = 0.01                 # Robot damping ratio
        self.c = 2 * self.zeta * self.wn * self.mass  # Calculate damping coeff

        self.counter = 0                 # counter for number of steps
        self.spacing = 0.75 * (1 / self.f)       # Space commands by one period of oscillation
        
        # Define thesholds for trial limits
        self.rod_max_position = np.inf        # max jump height (m)
        self.rod_min_position = 0             # amount the spring can compress by (m)
        self.rod_max_velocity = np.inf        # max rod velocity (m/s)
        self.act_max_position = 0.25          # length of which actuator can move (m)
        self.act_max_velocity = 2.0           # max velocity of actuator (m/s)
        self.act_max_accel = 63.2             # max acceleration of actuator (m/s^2)
        self.act_distance = 0.008             # distance the actuator can move along the rod
        self.spring_limit = -0.008            # amount the spring can compress by

        # variable for logging height reached and spring k chosen
        self.height_reached = 0

        self.viewer = None
        self.state = None
        self.done = False
        self.x_act_accel = 0.0

        self.ep_steps = ep_steps                # maximum number of steps to run
        self.sim_step_size = sim_step_size           # time between steps in simulation
        self.sim_duration = sim_duration        # how long to simulate for
        self.reward_function = reward_function  # variable for choosing what kind of reward function we want to use
        self.specified_height = specified_height
        self.min_spring_k = min_spring_k  
        self.max_spring_k = max_spring_k
        self.min_zeta = min_zeta
        self.max_zeta = max_zeta

        self.model_type = model_type
        self._create_spaces(model_type=self.model_type)

    def _create_spaces(self, model_type):
        if model_type == "TD3":
            # This action space is the range of acceleration mass on the rod
            low_limit = np.array([self.min_spring_k, self.min_zeta])
            high_limit = np.array([self.max_spring_k, self.max_zeta])

            # Create continuous action space
            self.action_space = spaces.Box(low=low_limit,
                                        high=high_limit,
                                        dtype=np.float32)
            
        elif model_type == "PPO":
            # raise NotImplementedError("PPO not fully implemented yet")
            self.action_space_max = 9999
            
            # Create discrete action space
            self.action_space = spaces.MultiDiscrete([self.action_space_max+1, self.action_space_max+1])

        else:
            raise TypeError("Model type not recognized")

        obs_len = int(self.sim_duration / self.sim_step_size + 1)

        low_limit = np.array([self.rod_min_position * np.ones(obs_len),       # max observable jump height
                            -self.rod_max_velocity * np.ones(obs_len),      # max observable jump velocity
                            -self.act_max_position/2 * np.ones(obs_len),    # min observable actuator position
                            -self.act_max_velocity * np.ones(obs_len)])
                                # max observable actuator velocity
        high_limit = np.array([self.rod_max_position * np.ones(obs_len),      # max observable jump height
                            self.rod_max_velocity * np.ones(obs_len),      # max observable jump velocity
                            self.act_max_position/2 * np.ones(obs_len),    # max observable actuator position
                            self.act_max_velocity * np.ones(obs_len)])     # max observable actuator velocity

        self.observation_space = spaces.Box(low=low_limit,
                                            high=high_limit,
                                            dtype=np.float32)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        self.counter = self.counter + 1
        # If using TD3 using continuous action space
        if self.model_type == "TD3":
            self.spring_k = float(action[0])
            self.zeta = float(action[1])
        # If using PPO using discrete action space
        elif self.model_type == "PPO":
            self.spring_k = (action[0] * (self.max_spring_k - self.min_spring_k)) / self.action_space_max + self.min_spring_k
            self.zeta = (action[1] * (self.max_zeta - self.min_zeta)) / self.action_space_max + self.min_zeta

        # Create and instance of the pogo model
        pogo_stick = PogoODEnonlinear(self.m_act, 
                                      self.m_rod, 
                                      self.spring_k,           # Notice the action is the spring constant
                                      self.zeta, 
                                      self.act_max_accel, 
                                      self.act_max_velocity, 
                                      self.act_distance,
                                      self.spring_limit,
                                      self.spacing)
        
        # TODO: ASA, 09/15/21, Consider using random initial conditions for actuator location
        # Initialize the pogo stick initial conditions
        x_init = 0.0
        x_dot_init = 0.0
        x_act_init = 0.0
        x_act_dot_init = 0.0
        
        x0 = [x_init, x_dot_init, x_act_init, x_act_dot_init]
        
        # Simulate the pogo stick
        time, timeseries = pogo_stick.run_simulation(x0, duration=self.sim_duration, max_step=self.sim_step_size)

        # Pull out parts of the simulation data
        x = timeseries[0,:]
        x_dot = timeseries[1,:] 
        x_act = timeseries[2,:] 
        x_act_dot = timeseries[3,:]
        
        # Capture the max height reached
        self.height_reached = np.max(x)
        # Set the return state
        self.state = timeseries

        # End the trial when we reach the maximum number of steps
        # In most cases the max steps is 1
        if self.counter >= self.ep_steps:
            self.done = True

        # Get the reward depending on the reward function
        reward = self.calc_reward(self.reward_function, timeseries)

        return self.state, reward, self.done, {}

    def calc_reward(self, reward_function, timeseries):
        try:
            if reward_function == "RewardHeight": 
                reward = np.max(timeseries[0,:])

            elif reward_function == "RewardSpecifiedHeight":
                height = np.max(timeseries[0,:])
                error = abs(height - self.specified_height) / self.specified_height
                reward = -1 * error

            elif reward_function == "RewardEfficiency":
                raise NotImplementedError("\n RewardEfficiency not implemented yet")
    
        except:
            raise ValueError("REWARD FUNCTION NOT PROPERLY DEFINED PROPERLY")
            print("Proper reward functions are:" ,"\n",
                  "RewardHeight: Rewards jumping high" , "\n",
                  "RewardSpecifiedHeight: Rewards jumping to specified height")
            sys.exit()

        return reward

    def reset(self): 
        # Reset the counter, the jumping flag, the landings counter, the power used list and the height reached list
        self.counter = 0
        
        self.state = np.zeros((4, int(self.sim_duration / self.sim_step_size + 1)))

        # Reset the done flag
        self.done = False
        
        return np.array(self.state)

    def render(self, mode='human', close=False):
        raise NotImplementedError("\n Render not implemented, and not applicable to this environment.")

    def close(self):
        pass
        