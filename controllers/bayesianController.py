from . import BaseController
from tinyphysics import TinyPhysicsModel, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX
import numpy as np
from collections import namedtuple
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from typing import List


State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
PIDPolicy = namedtuple('PIDPolicy', ['p', 'i', 'd'])


class MyPhysicsModel(TinyPhysicsModel):
    
    """
    Physics Model with temperature set to a low value to avoid noise.
    """
    
    def __init__(self, model_path: str, debug: bool) -> None:
        super().__init__(model_path, debug)
        
        
    def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
        tokenized_actions = self.tokenizer.encode(past_preds)
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        input_data = {
          'states': np.expand_dims(states, axis=0).astype(np.float32),
          'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
        }
        # Setting a low temperature when trying to find the true optimal path, instead of having a lucky strike.
        return self.tokenizer.decode(self.predict(input_data, temperature=0.0001))


class Controller(BaseController):
    
    """
    Bayesian Controller that sets a pid policy based on the optimal path defined in the future steps.
    The idea is to define a pid policy based on the values that generate a lower cost along the future plan provided.
    """
    
    def __init__(self):
        self.driving_model = MyPhysicsModel("./models/tinyphysics.onnx", debug=False)
        # Setting the initial policy to base parameters.
        self.current_policy = PIDPolicy(p=0.085, i=0.11, d=0.0)
        self.error_integral = 0
        self.prev_error = 0
        self.action_history = []
        self.state_history = []
        self.preds_history = []
        self.costs = []
        self.prev_target = -np.inf
        self.change_policy_in = 0
        self.n_updates = 0
        
        
    @staticmethod
    def compute_cost(target, pred) -> dict:
        """
        Computes the cost of a path, given a target and a pred array.

        :param target: Targets of the path.
        :type: np.ndarray
        :param pred: Predictions of the path.
        :type: np.ndarray
        :return: The cost of the path in a dict, separated into the lateral acceleration cost and the jerk cost.
        """
        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}
    
    
    def wrapper_evaluate_path(self, model, state_dict):
        """
        Returns an optimizable function with p, i and d parameters.
        """
        # We wrap the function inside because we need to pass the model and the state as non optimizable parameters. (We could also use the class attributes)
        def evaluate_path(p, i, d):
            error_integral = state_dict['ERROR_INTEGRAL']
            prev_error = state_dict['PREV_ERROR']
            actions = state_dict['ACTIONS'].copy()
            sim_states = state_dict['SIM_STATES'].copy()
            past_preds = state_dict['PAST_PREDS'].copy()
            future = state_dict['FUTURE_PLAN']
            current_lataccel = past_preds[-1]
            plan_steps = len(future.lataccel)
            
            # The idea is pretty simple, iterate over the future plan and return the cost for a given p, i and d.
            for idx in range(plan_steps):
                error = (future.lataccel[idx] - current_lataccel)
                error_integral += error
                error_diff = error - prev_error
                prev_error = error
                action = np.clip(p * error + i * error_integral + d * error_diff, STEER_RANGE[0], STEER_RANGE[1])
                actions.append(action)
                pred = model.get_current_lataccel(
                  sim_states=sim_states[-CONTEXT_LENGTH:],
                  actions=actions[-CONTEXT_LENGTH:],
                  past_preds=past_preds[-CONTEXT_LENGTH:]
                )
                pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
                past_preds.append(pred)
                if idx < plan_steps - 1:
                    sim_states.append(State(a_ego=future.a_ego[idx], v_ego=future.v_ego[idx], roll_lataccel=future.roll_lataccel[idx]))
                current_lataccel = pred
            cost = Controller.compute_cost(np.array(future.lataccel[:plan_steps]), np.array(past_preds[-plan_steps:]))
            self.costs.append(cost)
            
            # Sometimes, especially in high lataccel environments, a path with low lataccel error and high jerk error can appear.
            # While it might be tempting to go for that route, the high jerk cost leaves us with a high self correcting and oscillating environment.
            # This chattering can percolate further than the 50 steps of the optimization and leave us in a worse starting point the next state, so we want to make sure the jerk cost is contained.
            # These parameters are hard coded by my knowledge and assumptions of the problem. Better parameters can be optimized and used.
            min_jerk = self.costs[0]['jerk_cost']
            if min_jerk < 50:
                mul = 1 if cost['jerk_cost'] < min_jerk*2 else 100
            else:
                mul = 1 if cost['jerk_cost'] < min_jerk*1.3 else 100
            #print(p, i, d, cost, 1/(cost['total_cost']*mul))
            return 1/(cost['total_cost']*mul)
        return evaluate_path
    
    
    def compute_action(self, target_lataccel, current_lataccel, p=None, i=None, d=None):
        """
        Computes an action based on p, i and d parameters and updates the errors.

        :param target_lataccel: Target lateral acceleration.
        :type: float
        :param current_lataccel: Current lateral acceleration.
        :type: float
        :param p: Term for the error correction.
        :type: float
        :param i: Term for the integral error correction.
        :type: float
        :param d: Term for future error correction.
        :type: float
        :return: The computed action.
        """
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        p = self.current_policy.p if p is None else p
        i = self.current_policy.i if i is None else i
        d = self.current_policy.d if d is None else d
        act = np.clip(p * error + i * self.error_integral + d * error_diff, STEER_RANGE[0], STEER_RANGE[1])
        return act
    
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Updates the state of the bayesian controller and returns an action.

        :param target_lataccel: Target lateral acceleration.
        :type: float
        :param current_lataccel: Current lateral acceleration.
        :type: float
        :param state: Current state.
        :type: State
        :param future_plan: Future plan of the object.
        :type: FuturePlan
        :return: The computed steer action.
        """
        self.n_updates += 1
        self.state_history.append(state)
        self.preds_history.append(current_lataccel)
        self.change_policy_in -= 1
        
        # We don't optimize if we don't have enough history or if the future plan is truncated (since they are not considered in the final calculation)
        if len(self.action_history) >= 20 and self.change_policy_in <= 0 and len(future_plan.lataccel) >= 45:
            self.costs = []
            state_dict = {
                'PREV_ERROR': self.prev_error,
                'ERROR_INTEGRAL': self.error_integral,
                'ACTIONS': self.action_history,
                'SIM_STATES': self.state_history,
                'PAST_PREDS': self.preds_history,
                'FUTURE_PLAN': FuturePlan(lataccel=[target_lataccel] + future_plan.lataccel, a_ego=future_plan.a_ego, v_ego=future_plan.v_ego, roll_lataccel=future_plan.roll_lataccel)
            }
            wrapper_function = self.wrapper_evaluate_path(self.driving_model, state_dict)

            # Bounded region of parameter space
            pbounds = {'p': (0.05, 0.6), 'i': (0.02, 0.2), 'd': (-0.3, 0.3)}

            optimizer = BayesianOptimization(
                f=wrapper_function,
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True,
                verbose=0
            )
            
            # The first point probed is the default policy
            optimizer.probe(
                params={'p': 0.085, 'i': 0.11, 'd': 0.0},
                lazy=False,
            )
            current_target = 1/optimizer.res[0]['target']
            
            # If the cost is low we dont consider changing policy, since the default policy is already optimized and is unlikely to have a better counterpart.
            # This threshold is an heuristic set by hand and to avoid excess compute with low return. With more resources this branch could be removed. Optimizing the threshold is also possible.
            if current_target <= 90 and self.current_policy == PIDPolicy(p=0.085, i=0.11, d=-0.0):
                self.current_policy = PIDPolicy(p=0.085, i=0.11, d=-0.0)
                self.change_policy_in = 50
            else:
                # The baseline is also a good point to probe.
                optimizer.probe(
                    params={'p': 0.3, 'i': 0.05, 'd': -0.1},
                    lazy=True,
                )
                
                # Since we already have two (maybe 3) decent points we can use an aquisition function that maximizes exploitation.
                acquisition_function = UtilityFunction(kind="ei", xi=1e-4)
                
                if self.prev_target != -np.inf:
                    if self.current_policy.p != 0.3 and self.current_policy.p != 0.085:
                        # If we are running a different policy of the two default ones we also probe it.
                        optimizer.probe(
                            params={'p': self.current_policy.p, 'i': self.current_policy.i, 'd': self.current_policy.d},
                            lazy=True,
                        )
                    
                # For the first step we explore the parameter space. For following steps we try to exploit the optimals. 
                # In case we have a high cost path we sample more random combinations, in order to give more information to the optimizer.
                # More compute would mean a better knowledge of the parameter space, which would yield better results.
                # The number of steps can be optimized better than this, because in some cases we run into duplicate data points to probe.
                # When both the default policies and the current policy have a high target it is probably useful to change the acquisition function to a more explorative one.
                optimizer.maximize(
                    init_points=10 if self.prev_target == -np.inf or current_target > 400 else 1,
                    acquisition_function=acquisition_function,
                    n_iter=20 if self.prev_target == -np.inf else 5
                )
                solution = optimizer.max['params']
                
                # Since the default policy is optimized for all paths we do not want to change it unless we have a meaningful parameter combination (meaningful set as at least 70% of the cost)
                # In case we are running a policy that is not the default one we set this meaningful threshold higher, to 85%.
                # These heuristics are totally done by hand and based on my problem knowledge, they can probably be optimized.
                if self.current_policy != PIDPolicy(p=0.085, i=0.11, d=0.0):
                    threshold = 0.85
                else:
                    if current_target <= 90:
                        threshold = 0
                    else:
                        threshold = 0.7
                        
                if 1/optimizer.max['target'] < (current_target*threshold):
                    self.current_policy = PIDPolicy(p=solution['p'], i=solution['i'], d=solution['d'])
                else:
                    self.current_policy = PIDPolicy(p=0.085, i=0.11, d=0.0)
                    
                # We optimize again in 10 steps in low cost environments, and in 5 steps in high cost environments
                # Again, these heuristics are set by hand, they can be optimized.
                self.change_policy_in = 5 if optimizer.max['target'] > 0.005 else 10
            self.prev_target = optimizer.max['target']
            
        action = self.compute_action(target_lataccel, current_lataccel)
        self.action_history.append(action)
        return action
    