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

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import pickle

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

class DriverEnv(gym.Env):    
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (1,), dtype = np.float32
        )
        self.last_vals = None
        self.prev_cost = None
        self.prev_error = 0.0
        self.current_lataccel_history = []
        self.state_history = []
        self.action_history = []
        self.target_lataccel_history = []
        self.step_idx = 0
        self.driver_number = os.getpid()
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            #low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            #high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            low=np.array([-np.inf for _ in range(24)]),
            high=np.array([np.inf for _ in range(24)]),
            shape=(24,),
            dtype=np.float32
        )
        
    def calculate_cost(self, target, pred, prev_current_lataccel):
        lat_accel_cost = (target - pred)**2 * 100
        jerk_cost = ((pred - prev_current_lataccel) / DEL_T)**2 * 100
            
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return total_cost
    
    def get_state_target(self, idx):
        state = self.state_history[idx]
        # state = self.data.iloc[step_idx]
        return State(roll_lataccel=state.roll_lataccel, v_ego=state.v_ego, a_ego=state.a_ego), self.target_lataccel_history[idx]
    
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
        if options is None:
            return np.array([random.random() for _ in range(10)]), 0, False, False, {}
        
        state = options
        current_lataccel = options['current_lataccel']
        target_lataccel = options['target_lataccel']
        state = options['state']
        targetfuture = options['targetfuture'][:20]
        s = State(roll_lataccel=state.roll_lataccel, v_ego=state.v_ego, a_ego=state.a_ego)
        
        self.last_actions = self.update_array(self.last_actions, action[0])
            
        self.current_lataccel_history.append(current_lataccel)
        self.state_history.append(s)
        self.action_history.append(action[0])
        self.target_lataccel_history.append(target_lataccel)
        self.cum_error += target_lataccel - current_lataccel
        mean_actions = np.mean(self.last_actions)
        self.last_errors = self.update_array(self.last_errors, target_lataccel - current_lataccel)
        self.last_targets = self.update_array(self.last_targets, target_lataccel)
        mean_targetfuture_short = np.mean(targetfuture[:5]) if len(targetfuture) > 0 else target_lataccel
        mean_targetfuture_long = np.mean(targetfuture) if len(targetfuture) > 0 else target_lataccel
        mean_actions = np.mean(self.last_actions)
        direction = 1 if target_lataccel > current_lataccel else -1
        
        observation = np.array(
            [s.v_ego, s.a_ego, s.roll_lataccel, 
            target_lataccel, current_lataccel,
            mean_targetfuture_short, mean_targetfuture_long,
            (target_lataccel - targetfuture[4]) if len(targetfuture) > 4 else 0, (target_lataccel - targetfuture[-1]) if len(targetfuture) > 0 else 0,
            current_lataccel - mean_targetfuture_short, current_lataccel - mean_targetfuture_long, np.std(targetfuture[:5]) if len(targetfuture) > 0 else 0,
            np.std(targetfuture) if len(targetfuture) > 0 else 0,
            (target_lataccel - current_lataccel) * 0.3,
            abs(target_lataccel - current_lataccel), 
            abs(target_lataccel) - abs(current_lataccel), 
            mean_actions,
            (target_lataccel - current_lataccel) - self.prev_error,
            # self.cum_error,
            action[0] - mean_actions,
            np.std(self.last_actions),
            np.mean(self.last_errors),
            np.std(self.last_errors),
            np.std(self.last_targets),
            direction
        ]).astype(np.float32)
        
        self.prev_error = target_lataccel - current_lataccel
        
        return observation, 0, False, False, {}
        
    def reset(self, seed=None, options=None):
        opt = self.get_options()
        target_lataccel = opt['target_lataccel']
        current_lataccel = opt['current_lataccel']
        state = opt['state']
        steer_command = (target_lataccel - current_lataccel) * 0.3
        self.current_lataccel_history = [] #[current_lataccel]
        self.state_history = [] #[state]
        self.action_history = [] #[steer_command]
        self.target_lataccel_history = [] #[target_lataccel]
        self.last_targets = np.array([])
        self.last_errors = np.array([])
        self.cum_error = 0
        self.prev_error = 0
        self.last_actions = np.array([])
        self.prev_cost = 0
        observation, _, _, _, info = self.step(np.array([steer_command]).astype(np.float32))
        return observation, {}

    def render(self):
        pass

    def close(self):
        pass


class Controller(BaseController):
  def __init__(self):
    eval_env = DriverEnv()
    log_dir = "/notebooks/comma/tmp/"
    #log_dir = "/notebooks/comma/controls_challenge/tmp/"
    stats_path = os.path.join(log_dir, "TempFinalRecurrentPPO3_Steering_vec_normalize.pkl")
    #stats_path = os.path.join(log_dir, "model_checkpoints/rl_model_vecnormalize_950000_steps.pkl")
    eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    eval_env = VecNormalize.load(stats_path, eval_env)
    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False
    #eval_env = VecFrameStack(eval_env, 10)
    self.lstm_states = None
    eval_env.clip_reward=200
    eval_env.clip_obs=100
    # Episode start signals are used to reset the lstm states
    self.episode_starts = np.ones((1,), dtype=bool)

    # Load the agent
    #self.model = RecurrentPPO.load(log_dir + "model_checkpoints/rl_model_950000_steps", env=eval_env, seed=42)
    self.model = RecurrentPPO.load(log_dir + "TempFinalRecurrentPPO3_Steering", env=eval_env, seed=42)
    self.model.policy.normalize_images = False
    self.env = eval_env
    self.reset_env = True
    self.n_updates = 0
    self.controller_id = os.getpid()
    self.actions = []
    
    
  def exponential_moving_average(self, data, span):
    """
    Calculate the Exponential Moving Average (EMA) of a given data array.

    Parameters:
    data (array-like): The input array containing the data points.
    span (int): The span for the EMA.

    Returns:
    np.array: The EMA of the data.
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("data should be a list or numpy array")
    
    if not isinstance(span, int) or span <= 0:
        raise ValueError("span should be a positive integer")

    data = np.asarray(data)
    ema = np.zeros_like(data, dtype=float)
    alpha = 2 / (span + 1)
    
    # The first EMA value is the same as the first data point
    ema[0] = data[0]
    
    # Compute the EMA for each subsequent data point
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema
    
    
  def update_array(self, array, newVal, limit=20):
    array.append(newVal)
    if len(array) > limit:
      array = array[1:]
    return array
    
    
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.reset_env:
      self.update_options({'target_lataccel': target_lataccel, 'current_lataccel': current_lataccel, 'state': state, 'targetfuture': future_plan[2]})
      self.observation = self.env.reset()
      self.observation = self.observation[0]
      self.reset_env = False
      self.n_updates += 1
      self.actions.append((target_lataccel - current_lataccel) * 0.3)
      return (target_lataccel - current_lataccel) * 0.3
    else:
      self.n_updates += 1
      self.update_options({'target_lataccel': target_lataccel, 'current_lataccel': current_lataccel, 'state': state, 'targetfuture': future_plan[2]})
      action, self.lstm_states = self.model.predict(self.observation, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
      self.observation, _, dones, _ = self.env.step([[action[0]*2]])
      self.observation = self.observation[0]
      self.episode_starts = dones
      self.actions = self.update_array(self.actions, action[0]*2, 3)
      act = self.exponential_moving_average(self.actions, 3)[-1]
      print(act, action[0]*2)
      return action[0]*2
    
  def update_options(self, options):
    with open(f'{self.controller_id}_options_dict.pkl', 'wb') as handle:
      pickle.dump(options, handle, protocol=pickle.HIGHEST_PROTOCOL)