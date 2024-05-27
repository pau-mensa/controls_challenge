import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from environment import DriverEnv
from stable_baselines3 import PPO, A2C, TD3, SAC # DQN coming soon
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
import pickle

class BaseController:
  def update(self, target_lataccel, current_lataccel, state, target_future):
    """
    Args:
      target_lataccel: The target lateral acceleration.
      current_lataccel: The current lateral acceleration.
      state: The current state of the vehicle.
      target_future: The future target lateral acceleration plan for the next N frames.
    Returns:
      The control signal to be applied to the vehicle.
    """
    raise NotImplementedError
  

class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3


class RLController(BaseController):
  def __init__(self):
    eval_env = DriverEnv()
    log_dir = "/notebooks/comma/tmp/"
    #log_dir = "/notebooks/comma/controls_challenge/tmp/"
    stats_path = os.path.join(log_dir, "FinalRecurrentPPO2Finetuned_Steering_vec_normalize.pkl")
    #stats_path = os.path.join(log_dir, "model_checkpoints/rl_model_vecnormalize_550000_steps.pkl")
    eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    eval_env = VecNormalize.load(stats_path, eval_env)
    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False
    #eval_env = VecFrameStack(eval_env, 10)
    self.lstm_states = None
    # Episode start signals are used to reset the lstm states
    self.episode_starts = np.ones((1,), dtype=bool)

    # Load the agent
    self.model = RecurrentPPO.load(log_dir + "FinalRecurrentPPO2Finetuned_Steering", env=eval_env)
    self.env = eval_env
    self.reset_env = True
    self.n_updates = 0
    
    
  def update(self, target_lataccel, current_lataccel, state):
    if self.reset_env:
      self.update_options({'target_lataccel': target_lataccel, 'current_lataccel': current_lataccel, 'state': state})
      self.observation = self.env.reset()
      self.observation = self.observation[0]
      action, self.lstm_states = self.model.predict(self.observation, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
      self.reset_env = False
      self.n_updates += 1
      return action[0]*2
    else:
      self.n_updates += 1
      self.update_options({'target_lataccel': target_lataccel, 'current_lataccel': current_lataccel, 'state': state})
      action, self.lstm_states = self.model.predict(self.observation, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
      #ction, _ = self.model.predict(self.observation, deterministic=True)
      #self.observation, _, dones, _ = self.env.step([[action[0][0]]])
      self.observation, _, dones, _ = self.env.step([[action[0]*2]])
      self.observation = self.observation[0]
      self.episode_starts = dones
      #return action[0][0]
      return action[0]*2
    
  def update_options(self, options):
    with open('options_dict.pkl', 'wb') as handle:
      pickle.dump(options, handle, protocol=pickle.HIGHEST_PROTOCOL)