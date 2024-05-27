import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from collections import namedtuple
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
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            #low=np.array([-0.5, -5, -0.1, -5, -5, -2, -5, -2, -10, -50]),
            #high=np.array([50, 5, 0.2, 5, 5, 2, 5, 2, 10, 50]),
            #low=np.array([-1, -1, -1, -1, -1, -1, -1]), #, -2]),
            #high=np.array([1, 1, 1, 1, 1, 1, 1]), #, 2]),
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            #shape=(N_CHANNELS, HEIGHT, WIDTH), 
            shape=(11,),
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
        with open('options_dict.pkl', 'rb') as handle:
            options = pickle.load(handle)
        return options
    
    def step(self, action):
        options = self.get_options()
        if options is None:
            return np.array([random.random() for _ in range(10)]), 0, False, False, {}
        
        state = options
        current_lataccel = options['current_lataccel']
        target_lataccel = options['target_lataccel']
        state = options['state']
        s = State(roll_lataccel=state.roll_lataccel, v_ego=state.v_ego, a_ego=state.a_ego)
        
        if len(self.last_actions) > 20:
            # Remove the first element to maintain a length of 25
            self.last_actions = np.delete(self.last_actions, 0)
            
        self.current_lataccel_history.append(current_lataccel)
        self.state_history.append(s)
        self.action_history.append(action[0])
        self.target_lataccel_history.append(target_lataccel)
        mean_actions = np.mean(self.last_actions)
        self.step_idx += 1
        observation = np.array(
            [s.v_ego, s.a_ego, s.roll_lataccel, 
            target_lataccel, current_lataccel,
             (target_lataccel - current_lataccel) * 0.3,
            abs(target_lataccel - current_lataccel), 
            abs(target_lataccel) - abs(current_lataccel), 
            mean_actions,
            (target_lataccel - current_lataccel) - self.prev_error,
             self.cum_error,
             #(action[0] - mean_actions)/mean_actions,
        ]).astype(np.float32)
        
        self.prev_error = target_lataccel - current_lataccel
        self.cum_error += target_lataccel - current_lataccel
        
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
        self.cum_error = 0
        self.prev_error = 0
        self.step_idx = 1
        self.last_actions = np.array([steer_command])
        self.prev_cost = 0
        observation, _, _, _, info = self.step(np.array([steer_command]).astype(np.float32))
        return observation, {}

    def render(self):
        pass

    def close(self):
        pass