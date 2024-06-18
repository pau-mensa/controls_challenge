from . import BaseController
#from . import DriverEnv

import os
import gymnasium as gym
import numpy as np
from collections import namedtuple
from gymnasium import spaces
import random
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
import pickle
import pandas as pd

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import pickle

STEER_RANGE = [-2, 2]
DEL_T = 0.1
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


class DriverEnv(gym.Env):    
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (1,), dtype = np.float32
        )
        self.last_vals = None
        self.prev_cost = None
        self.prev_error = 0.0
        self.step_idx = 0
        self.driver_number = os.getpid()
        self.observation_space = spaces.Dict(
            spaces={
                'CURRENT_INFO': spaces.Box(low=-np.inf, high=np.inf, shape=(11,)),
                'CURRENT_LATACCEL_DIFF': spaces.Box(low=-np.inf, high=np.inf, shape=(49,)),
                'FUTURE_LATACCEL': spaces.Box(low=-np.inf, high=np.inf, shape=(49,)),
                'FUTURE_ROLL': spaces.Box(low=-np.inf, high=np.inf, shape=(49,)),
                'FUTURE_VEL': spaces.Box(low=-np.inf, high=np.inf, shape=(49,)),
                'FUTURE_A': spaces.Box(low=-np.inf, high=np.inf, shape=(49,)),
                'PAST_LATACCEL': spaces.Box(low=-np.inf, high=np.inf, shape=(20,)),
                'PAST_ACTIONS': spaces.Box(low=-np.inf, high=np.inf, shape=(20,)),
                'PAST_ROLL': spaces.Box(low=-np.inf, high=np.inf, shape=(20,)),
                'PAST_VEL': spaces.Box(low=-np.inf, high=np.inf, shape=(20,)),
                'PAST_A': spaces.Box(low=-np.inf, high=np.inf, shape=(20,))
            })
    
    def get_options(self):
        with open(f'{self.driver_number}_options_dict.pkl', 'rb') as handle:
            options = pickle.load(handle)
        return options
    
    def update_array(self, array, newVal, limit=20):
        array = np.append(array, newVal)
        if len(array) > limit:
            array = np.delete(array, 0)
        return array
    
    def step(self, action):
        options = self.get_options()
        s = options
        state = s['state']
        target_lataccel = s['target_lataccel']
        current_lataccel = s['current_lataccel']
        current_lataccel_history = s['current_lataccel_history']
        action_history = s['action_history']
        state_history = s['state_history']
        
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        act = action.item()*2
        
        f = s['future_plan']
        future = FuturePlan(
            lataccel=f.lataccel if len(f.lataccel) > 0 else [current_lataccel],
            roll_lataccel=f.roll_lataccel if len(f.roll_lataccel) > 0 else [state.roll_lataccel],
            v_ego=f.v_ego if len(f.v_ego) > 0 else [state.v_ego],
            a_ego=f.a_ego if len(f.a_ego) > 0 else [state.a_ego]
        )
        
        self.last_actions = self.update_array(self.last_actions, act, 10)
        
        last_action = self.last_actions[-2] if len(self.last_actions) > 1 else act
        action_diff = act - last_action
        
        immediate_future_jerk_cost = np.mean((np.diff(np.array(future.lataccel[:10])) / DEL_T)**2) if len(future.lataccel) > 1 else 0
        future_jerk_cost = np.mean((np.diff(np.array(future.lataccel)) / DEL_T)**2) if len(future.lataccel) > 1 else 0
        immediate_future_lataccel = np.mean(future.lataccel[:10]) if len(future.lataccel) >= 1 else 0
        future_lataccel = np.mean(future.lataccel) if len(future.lataccel) >= 1 else 0

        observation = {
            'CURRENT_INFO': np.array([
                target_lataccel, current_lataccel, action_diff, last_action,
                error, self.prev_error, np.clip(self.error_integral, 100, -100),
                immediate_future_jerk_cost, future_jerk_cost,
                immediate_future_lataccel, future_lataccel
            ]), 
            'CURRENT_LATACCEL_DIFF': np.pad(np.array(future.lataccel - current_lataccel), (0, 49 - len(np.array(future.lataccel - current_lataccel)))), 
            'FUTURE_LATACCEL': np.pad(np.array(future.lataccel), (0, 49 - len(np.array(future.lataccel)))), 
            'FUTURE_ROLL':np.pad(np.array(future.roll_lataccel), (0, 49 - len(np.array(future.roll_lataccel)))), 
            'FUTURE_VEL':np.pad(np.array(future.v_ego), (0, 49 - len(np.array(future.v_ego)))), 
            'FUTURE_A':np.pad(np.array(future.a_ego), (0, 49 - len(np.array(future.a_ego)))),
            'PAST_LATACCEL':np.array(current_lataccel_history[-20:]),
            'PAST_ACTIONS': np.array(action_history[-20:]),
            'PAST_ROLL': np.array([s.roll_lataccel for s in state_history[-20:]]),
            'PAST_VEL': np.array([s.v_ego for s in state_history[-20:]]),
            'PAST_A': np.array([s.a_ego for s in state_history[-20:]]),
        }
                        
        return observation, 0, False, False, {}
        
    def reset(self, seed=None, options=None):
        opt = self.get_options()
        target_lataccel = opt['target_lataccel']
        current_lataccel = opt['current_lataccel']
        steer_action = (target_lataccel - current_lataccel)*0.085
        self.cum_error = 0
        self.prev_error = 0
        self.error_integral = 0
        self.last_actions = np.array([])
        self.prev_cost = 0
        observation, _, _, _, info = self.step(np.array([steer_action]).astype(np.float32))
        return observation, {}

    def render(self):
        pass

    def close(self):
        pass


class Controller(BaseController):
  def __init__(self):
    eval_env = DriverEnv()
    log_dir = "/notebooks/comma/tmp/"
    stats_path = os.path.join(log_dir, "TRPO_Steering_vec_normalize.pkl")
    eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    eval_env = VecNormalize.load(stats_path, eval_env)
    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False
    eval_env.clip_reward=np.inf
    eval_env.clip_obs=300

    # Load the agent
    self.model = TRPO.load(log_dir + "TRPO_Steering", env=eval_env, seed=42)
    self.model.policy.normalize_images = False
    self.env = eval_env
    self.reset_env = True
    self.n_updates = 0
    self.controller_id = os.getpid()
    self.action_history = []
    self.current_lataccel_history = []
    self.state_history = []
    self.error_integral = 0
    self.prev_error = 0
    
    
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.state_history.append(state)
    self.current_lataccel_history.append(current_lataccel)  
    if len(self.action_history) < 20:
      error = target_lataccel - current_lataccel
      self.error_integral += error
      diff_error = error - self.prev_error
      self.prev_error = error
      act = error * 0.085 + self.error_integral*0.11    
    else:
      self.update_options({
          'target_lataccel': target_lataccel, 
          'current_lataccel': current_lataccel, 
          'state': state, 
          'future_plan': future_plan,
          'current_lataccel_history': self.current_lataccel_history,
          'action_history': self.action_history,
          'state_history': self.state_history
      })
      if self.reset_env:
        self.observation = self.env.reset()
        self.reset_env = False
      action, _ = self.model.predict(self.observation, deterministic=True)
      self.observation, _, dones, _ = self.env.step(action)
      act = action.item()*2
    self.n_updates += 1
    self.action_history.append(act)
    return act
    
    
  def update_options(self, options):
    # We need to pass the parameters through the pkl because the gym interface does not allow for anything other than action in the step function.
    with open(f'{self.controller_id}_options_dict.pkl', 'wb') as handle:
      pickle.dump(options, handle, protocol=pickle.HIGHEST_PROTOCOL)