import gymnasium as gym
import highway_env

import dataclasses
from frozendict import frozendict
import numpy as np
import scipy as scp
from math import isnan
from copy import deepcopy
import time

from pathos.multiprocessing import ProcessingPool as Pool    # Can run multiprocessing in interactive shell
import multiprocessing as mp
process_count = (mp.cpu_count()-2)
import functools
from functools import lru_cache
import itertools as it

import warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)    # Other options: INFO, WARNING, ERROR, CRITICAL

from typing import TypeVar, Generic, Sequence, Set, Hashable, Union, Callable, Tuple, Mapping


######################### DEFINE NECESSARY DATACLASSES #########################

State = Tuple[frozendict]
Action = int

@dataclasses.dataclass
class MDPTable():
    start_state: Tuple[frozendict]
    transition: dict[dict[Tuple[Tuple[frozendict], int], Tuple[frozendict]], float]
    absorption: dict[Tuple[frozendict], bool]
    reward: dict[Tuple[Tuple[frozendict], int], float]
    state_list: Sequence[Tuple[frozendict]]
    action_list: Sequence[int]
    mdp_type: str


@dataclasses.dataclass
class MDPMatrix():
    start_state: Tuple[frozendict]
    transition: np.ndarray
    reward: np.ndarray
    absorbing_state_mask: np.ndarray
    discount: float
    state_list: Sequence[Tuple[frozendict]]
    action_list: Sequence[int]
    mdp_type: str


@dataclasses.dataclass
class PlanningResult():
    start_state: Tuple[frozendict]
    state_values: dict[Tuple[frozendict], float]
    action_values: dict[Tuple[frozendict], dict[int, float]]
    mdp_type: str




######################### DEFINE GYM-MDP WRAPPER BASE CLASS #########################

class GymMDP:
    max_states = int(1e6)
    discount_rate = 1.0

    MDP_TYPE = "Generic"

    def __init__(self, *args, **kwargs):
        self._mdp_tables = None
        self._mdp_matrices = None
        self._mdp_plan = None
        self.config = kwargs.get("config", None)
        self._env = gym.make(*args, **kwargs)
        self._first_state, self._first_info = self._env.reset()
        self._first_state = GymDiscreteMDP.to_hashable_state(self._first_state)
        self._current_state = deepcopy(self._first_state)
        action_space = self._env.action_space
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self._actions = range(action_space.start, action_space.n)
        else:
            raise NotImplementedError("Only discrete action spaces are currently supported")
            # |Can later be extended for other gym actionspaces
       

    @property
    def first_state(self): 
        """
        Return the current MDPtable object.
        """
        return self._first_state
       

    @property
    def current_state(self): 
        """
        Return the current MDPtable object.
        """
        return self._current_state


    @property
    def mdp_tables(self) -> MDPTable: 
        """
        Return the current MDPtables object.
        """
        return self._mdp_tables


    @property
    def mdp_matrices(self) -> MDPMatrix: 
        """
        Return the current MDPmatrices object.
        """
        if self._mdp_matrices == None:
            self._mdp_matrices = self.get_MDPmatrices()
        return self._mdp_matrices


    @property
    def mdp_plan(self) -> PlanningResult: 
        """
        Return the current MDPplan object.
        """
        if self._mdp_plan == None:
            self._mdp_plan = self.value_iteration()
        return self._mdp_plan


    @property
    def actions(self): 
        """
        Return the current MDPtable object.
        """
        return self._actions


    @property
    def env(self):
        """
        Return the current environment object.
        """
        return self._env
    
    def get_copy(self):
        """Returns a copy of the current instance of the class to perform actions 
        without affecting the current state of the class"""
        return deepcopy(self)


    def get_env_properties(self):
        return self._first_state, self._current_state, self._actions


    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.

        Args:
            action (int): The action to be taken expressed by an integer number.
        """
        raise NotImplementedError("function \'step\' not implemented in GymMDP class, please implement in child class.")
   

    def populate_MDPtable(self, max_depth: int = 2, unknown_reward: float = 0):
        """
        Code to set up MDP tables. Can perform serial or parallel execution of each of the 6 actions 
            in a given environment.

        Parameters:
            max_depth (int): The number of steps to plan ahead (default = 2)
            unknown_reward (float): reward value set for state-action pairs which are left
                                    unexplored due to depth constraint (default = 0)
        """
        raise NotImplementedError("function \'populate_MDPtable\' not implemented in GymMDP class, please implement in child class.")
    

    def get_MDPmatrices(self) -> MDPMatrix:
        """
        Converts MDP tables to mtrices useful for value calculation process

        Parameters:
            None
        """
        if (self._mdp_matrices and self._mdp_matrices.start_state == self._current_state):
            # If it was already calculated for the current state return existing matrix
            return self._mdp_matrices
        tb = self._mdp_tables
        if tb == None:
            raise ValueError("MDP tables not set up yet.")
        state_idx = {s: i for i, s in enumerate(tb.state_list)}
        action_idx = {a: i for i, a in enumerate(tb.action_list)}
        transition_mat = np.zeros((len(state_idx), len(action_idx), len(state_idx)))
        reward_mat = np.ones((len(state_idx), len(action_idx)))*-np.inf
        absorbing_vec = np.array([tb.absorption[s] if s in tb.absorption else False for s in tb.state_list], dtype=bool)
        for (s, a), ns_prob in tb.transition.items():
            for ns, prob in ns_prob.items():
                transition_mat[state_idx[s], action_idx[a], state_idx[ns]] = prob
        for (s, a), r in tb.reward.items():
            reward_mat[state_idx[s], action_idx[a]] = r
        self._mdp_matrices = MDPMatrix(
            start_state=tb.start_state,
            transition=transition_mat,
            reward=reward_mat,
            absorbing_state_mask=absorbing_vec,
            discount=self.discount_rate,
            state_list=tb.state_list,
            action_list=tb.action_list,
            mdp_type=self.MDP_TYPE
        )
        return self._mdp_matrices
        

    def value_iteration(
            self,
            iterations: int = 1000,
            value_epsilon: float = 1e-6
        ) -> PlanningResult:
        """
        Perform value iteration on MDP tables

        Parameters:
            iterations (int): Max number of iterations to calculate state and action 
                                values (default = 100)
            value_epsilon (float): Tollerance for when to terminate iteration process
                                    based on delta between consecutive iterations (default = 1e-6)
        """
        if (self._mdp_plan and self._mdp_plan.start_state == self._current_state):
            # If it was already calculated for the current state return existing plan
            return self._mdp_plan
        mat = self.get_MDPmatrices()
        # if np.all(m.absorbing_state_mask == False) and m.discount == 1:
        #     raise ValueError("No absorbing states found in MDP with discount factor 1")
        value = np.zeros(len(mat.state_list))
        value_ = np.zeros(len(mat.state_list))
        qvalues = np.zeros((len(mat.state_list), len(mat.action_list)))
        for i in range(iterations):
            qvalues[:] = mat.reward + mat.discount*np.einsum("san,n->sa", mat.transition, value)
            # qvalues[m.absorbing_state_mask, :] = 0
            value_[:] = qvalues.max(axis=1)
            max_residual = np.abs(value - value_).max()
            if max_residual < value_epsilon:
                break
            value[:] = value_
        assert max_residual < value_epsilon, "Value iteration did not converge"
        action_values = {}
        for s, q in zip(mat.state_list, qvalues):
            action_values[s] = dict(zip(mat.action_list, q))
        logging.debug(value)
        logging.debug(action_values)
        self._mdp_plan = PlanningResult(
            start_state=mat.start_state,
            state_values=dict(zip(mat.state_list, value.tolist())),
            action_values=action_values,
            mdp_type=self.MDP_TYPE
        )
        return self._mdp_plan


    @classmethod
    # @lru_cache(maxsize=10000)
    def simulateAction(cls, action: int, sim_env, obsrv_features):
        """
        Given an instance of a Highway environment and an action to perform in the environment, the function
        will execute the action without making any changes to the original environment and return the
        next state and a copy of the updated environment instance.

        This function is implemented as a class methods to prevent multiprocessing from creating multiple copies
        of the class object for each process.

        This function returns 1 for transition probability since the highwayenv environment is completely
        deterministic.

        Args:
            action (int): The action to be taken expressed by an integer number.
            sim_env (gym environment): Whether to update the state of the current environment or simulate the action
                             without making changes to the active environment.
            obsrv_features (list): list of features returned by the gym environment as observation.
        """
        # raise NotImplementedError("Please Implement 'simulateAction' method in class "+cls.__name__)
        if not isinstance(sim_env, gym.wrappers.common.OrderEnforcing):
            raise TypeError("Environent parameter is not an instance of GymDiscreteMDP")
        env_copy = deepcopy(sim_env)
        obs, reward, done, truncated, info = env_copy.step(action)
        logging.debug(obs)
        next_state = cls.to_hashable_state(obs, obsrv_features)
        return((action, next_state, 1, reward, done, truncated, info, env_copy))


    @classmethod
    def to_hashable_state(cls, obs, **kwargs):
        """
        Create hashable (discretized) variable using environment state information, for MDP planning.
        The current implementation does nothing, as the implementation of thi function is strictly 
            evironment dependant
        """
        return obs








######################### DEFINE GYM DISCRETE-MDP WRAPPER CLASS #########################

class GymDiscreteMDP(GymMDP):
    max_states = int(1e6)
    discount_rate = 1.0

    MDP_TYPE = "Discrete"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       

    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.

        Args:
            action (int): The action to be taken expressed by an integer number.
        """
        obs, reward, done, truncated, info = self._env.step(action)
        logging.debug(obs)
        # |Here, the classmethod 'to_hashable_state' is being called with self instead of the class name to allow the code 
        # | to call the overridden 'to_hashable_state' method implemented in any child class.
        self._current_state = self.to_hashable_state(obs, self.config["observation"]["features"])
        return self._current_state, reward, done, truncated, info
   

    def populate_MDPtable(self, max_depth: int = 2, unknown_reward: float = 0):
        """
        Code to set up MDP tables. Can perform serial or parallel execution of each of the 6 actions 
            in a given environment.

        Parameters:
            max_depth (int): The number of steps to plan ahead (default = 2)
            unknown_reward (float): reward value set for state-action pairs which are left
                                    unexplored due to depth constraint (default = 0)
        """
        if (self._mdp_tables and self._mdp_tables.start_state == self._current_state):
            # |If it was already calculated for the current state return existing table
            return self._mdp_tables
        start_env = (self._current_state, 0, self._env)
        visited = set()
        transitions = {}
        rewards = {}
        absorption = {}
        frontier = {start_env}
        env2close = set()   # Maintain list of environments to close
        while frontier:
            state, depth, curr_env = frontier.pop()
            visited.add(state)
            if len(visited) >= self.max_states:
                raise ValueError(f"Maximum number of states reached ({self.max_states})")
            env2close.add(curr_env)
            if depth < max_depth:
                # |Parellel execution
                with Pool(process_count) as pool:
                    # |Here, the classmethod 'simulateAction' is being called with self instead of the class name to allow the code 
                    # | to call the overridden 'simulateAction' method implemented in any child class.
                    return_vals = pool.map(functools.partial(self.simulateAction, 
                                                             sim_env=curr_env, 
                                                             obsrv_features=tuple(self.config["observation"]["features"])), self.actions)
                # |OR Sequential execution (useful for bug-fixing)
                # return_vals = []
                # for actn in self.actions:
                #     return_vals.append(HighwayDiscreteMDP.simulateAction(actn, sim_env=curr_env, obsrv_features=self.config["observation"]["features"]))
                for action, next_state, trans_prob, reward, done, truncated, info, updated_env in return_vals:
                    logging.debug(str(state[0]) + ' | ' + str(action) + ' | ' + str(next_state[0]))
                    #2# |Populate transitions table
                    if (state, action) not in transitions:
                        transitions[(state, action)] = {}
                    if next_state not in transitions[(state, action)]:
                        transitions[(state, action)][next_state] = trans_prob
                    #2# |Populate reward functions
                    if (state[0], action) in rewards:
                        rewards[(state, action)] += reward * trans_prob
                    else:
                        rewards[(state, action)] = reward * trans_prob
                    #2# |Set absorption states
                    absorption[next_state] = done or truncated
                    #2# |Populate frontier
                    if (next_state not in visited) and not truncated:
                        frontier.add((next_state, depth + 1, updated_env))
                    else:
                        updated_env.close()
            else:
                #2# |Set rewards for unexplored state-action pairs to 0
                for action in self.actions:
                    if not (state, action) in rewards:
                        rewards[(state, action)] = unknown_reward
        # |Indent the lines below (logging) inside for more detailed updates on MDP status
        MDPstatus = "Current Depth: " + str(depth) + " | Frontier: " + str(len(frontier)) +\
                    " | Visited: " + str(len(visited)) + " | Transitions:" + str(len(transitions))
        logging.info(MDPstatus)
        env2close.discard(start_env[2]) # Keep the first step active
        for curr_env in env2close:
            # |Close all duplicate environments
            curr_env.close()
        env2close = set()   # Close ununsed environments at the end of the loop
        self._mdp_tables = MDPTable(start_state=self._current_state, transition=transitions, absorption=absorption, 
                                    reward=rewards, state_list=visited, action_list=set(self.actions))
        return self._mdp_tables
    
   
    def populate_MDPtable_async(self, max_depth: int = 2, unknown_reward: float = 0):
        """
        # TODO: Complete function implementation
        Code to set up MDP tables which runs multiple environments in parallel for each permutation
            of action sequence (determined by the depth of execution).

        Parameters:
            max_depth (int): The number of steps to plan ahead (default = 2)
            unknown_reward (float): reward value set for state-action pairs which are left
                                    unexplored due to depth constraint (default = 0)
        """
        if (self._mdp_tables and self._mdp_tables.start_state == self._current_state):
            # |If it was already calculated for the current state return existing table
            return self._mdp_tables
        # |Generate all permutations of action sequences
        actn_seq_permutations = it.permutations(self.actions, max_depth)
        # |Parellel execution
        results_cache = mp.Manager().dict()
        with Pool(process_count) as pool:
            return_vals = pool.map(functools.partial(self.simulateAction_async, 
                                                    sim_env=self._env, 
                                                    obsrv_features=tuple(self.config["observation"]["features"]),
                                                    results_cache = results_cache,
                                                    curr_state = self._current_state),
                                    actn_seq_permutations)
        return 0
        ## REFERENCE CODE ##
        if (self._mdp_tables and self._mdp_tables.start_state == self._current_state):
            # |If it was already calculated for the current state return existing table
            return self._mdp_tables
        start_env = (self._current_state, 0, self._env)
        visited = set()
        transitions = {}
        rewards = {}
        absorption = {}
        frontier = {start_env}
        env2close = set()   # Maintain list of environments to close
        while frontier:
            state, depth, curr_env = frontier.pop()
            visited.add(state)
            if len(visited) >= self.max_states:
                raise ValueError(f"Maximum number of states reached ({self.max_states})")
            env2close.add(curr_env)
            if depth < max_depth:
                # |Parellel execution
                with Pool(process_count) as pool:
                    return_vals = pool.map(functools.partial(HighwayDiscreteMDP.simulateAction, 
                                                             sim_env=curr_env, 
                                                             obsrv_features=tuple(self.config["observation"]["features"])), self.actions)
                # |OR Sequential execution (useful for bug-fixing)
                # return_vals = []
                # for actn in self.actions:
                #     return_vals.append(HighwayDiscreteMDP.simulateAction(actn, sim_env=curr_env, obsrv_features=self.config["observation"]["features"]))
                for action, next_state, trans_prob, reward, done, truncated, info, updated_env in return_vals:
                    logging.debug(str(state[0]) + ' | ' + str(action) + ' | ' + str(next_state[0]))
                    #2# |Populate transitions table
                    if (state, action) not in transitions:
                        transitions[(state, action)] = {}
                    if next_state not in transitions[(state, action)]:
                        transitions[(state, action)][next_state] = trans_prob
                    #2# |Populate reward functions
                    if (state[0], action) in rewards:
                        rewards[(state, action)] += reward * trans_prob
                    else:
                        rewards[(state, action)] = reward * trans_prob
                    #2# |Set absorption states
                    absorption[next_state] = done or truncated
                    #2# |Populate frontier
                    if (next_state not in visited) and not truncated:
                        frontier.add((next_state, depth + 1, updated_env))
                    else:
                        updated_env.close()
            else:
                #2# |Set rewards for unexplored state-action pairs to 0
                for action in self.actions:
                    if not (state, action) in rewards:
                        rewards[(state, action)] = unknown_reward
        # |Indent the lines below (logging) inside for more detailed updates on MDP status
        MDPstatus = "Current Depth: " + str(depth) + " | Frontier: " + str(len(frontier)) +\
                    " | Visited: " + str(len(visited)) + " | Transitions:" + str(len(transitions))
        logging.info(MDPstatus)
        env2close.discard(start_env[2]) # Keep the first step active
        for curr_env in env2close:
            # |Close all duplicate environments
            curr_env.close()
        env2close = set()   # Close ununsed environments at the end of the loop
        self._mdp_tables = MDPTable(start_state=self._current_state, transition=transitions, absorption=absorption, 
                                    reward=rewards, state_list=visited, action_list=set(self.actions))
        return self._mdp_tables
    

    @classmethod
    def simulateAction_async(cls, sim_env, obsrv_features, results_cache: dict, curr_state, action_seq: list[int], sim_depth: int = 0):
        """
        Given an instance of a gym environment and a sequence of actions to perform in the environment, 
        the function will execute the action sequence without making any changes to the original environment
        and return all intermediary states and a copy of the updated environment instance.

        This function is implemented as a class methods to prevent multiprocessing from creating multiple copies
        of the class object for each process.

        Args:
            action (int): The action to be taken expressed by an integer number.
            sim_env (gym environment): Whether to update the state of the current environment or simulate the action
                             without making changes to the active environment.
            obsrv_features (list): list of features returned by the gym environment as observation.
            sim_depth: The current depth of the recursive fuction-call stack. First call at 0.
        """
        # raise NotImplementedError("Please Implement 'simulateAction_async' method in class "+cls.__name__)
        # |Return result if already available from cache
        curr_action = action_seq[sim_depth]
        if (curr_state, curr_action) in results_cache:
            curr_action, next_state, _, reward, done, truncated, info, env_copy = results_cache[(curr_state, curr_action)]
        else:
            # |Else, simulate action and save current results to cache
            if sim_depth == 0:
                # |If first call, check environment type and make a copy of the environment
                if not results_cache:
                    raise ValueError("Please provide and empty dictionary for \'results_cache\'")
                if not isinstance(sim_env, gym.wrappers.common.OrderEnforcing):
                    raise TypeError("Environent parameter is not an instance of GymDiscreteMDP")
                env_copy = deepcopy(sim_env)
            else:
                # |Else, carry on with the copy of the first call
                env_copy = sim_env
            obs, reward, done, truncated, info = env_copy.step(curr_action)
            logging.debug(obs)
            next_state = cls.to_hashable_state(obs, obsrv_features)
            results_cache[(curr_state, curr_action)] = (curr_action, next_state, 1, reward, done, truncated, info, env_copy)
        curr_result = [(curr_action, next_state, 1, reward, done, truncated, info, env_copy)]
        if sim_depth + 1 < len(action_seq):
            next_state_results = cls.simulateAction_async(next_state, action_seq, env_copy, obsrv_features, results_cache, sim_depth + 1)
            curr_result.extend(next_state_results)
        return curr_result
    





######################### DEFINE GYM GRIDWORLD-MDP WRAPPER BASE CLASS #########################

class GymGridworldMDP(GymMDP):
    max_states = int(1e6)
    discount_rate = 1.0

    MDP_TYPE = "Gridworld"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       

    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.

        Args:
            action (int): The action to be taken expressed by an integer number.
        """
        obs, reward, done, truncated, info = self._env.step(action)
        logging.debug(obs)
        # |Here, the classmethod 'to_hashable_state' is being called with self instead of the class name to allow the code 
        # | to call the overridden 'to_hashable_state' method implemented in any child class.
        self._current_state = self.to_hashable_state(obs, self.config["observation"]["features"])
        return self._current_state, reward, done, truncated, info
        
   
    def populate_MDPtable(self, coordinate_list: list[State], action_list: list[int] = None, unknown_reward: float = 0, parallel_exec:bool = True, **kwargs):
        """
        Code to set up MDP tables which runs multiple environments in parallel for each permutation
            of action sequence (determined by the depth of execution).
        This logic is very different from the base-class logic, where the environment starts off in 
            a certain state and the decision tree is constructed based on actions taken in each 
            consecutive state. Here, the environment is divided into sections. Vehicles can only 
            exist in coordinates represented by those sections. This allows much coarser granularity
            of the environment dicretization and speeds up MDP table generation.

        Parameters:
            coordinate_list (list[State]): List of discretized grid coordinates, that will be used as initial 
                                    conditions for simulation
            action_list (list[int]): List of actions that will be simulated for each coordinate in 
                                   the coordinate list
            unknown_reward (float): Reward value set for non-existant state transitions
            parallel_exec (bool): Whether to run steps in parallel or sequential for-loop
        """
        # |Convert to list in case an interator object was passed
        coordinate_list = list(coordinate_list)
        if (self._mdp_tables and self._mdp_tables.start_state == coordinate_list):
            # |If it was already calculated for the current state-list return existing table
            return self._mdp_tables
        if not action_list:
            action_list = self._actions
        # |Generate all state-action pairs
        coord_action_pairs = it.product(coordinate_list, action_list)
        # |Create KDTree of all coordinates to effectively search for closest coordinate
        search_coordinates = scp.spatial.KDTree(coordinate_list)
        if parallel_exec:
            # |Parellel execution
            print("Simulation start")
            start = time.time()
            with Pool(process_count) as pool:
                return_vals = pool.map(functools.partial(self.simulateAction, 
                                                        sim_env=self._env, obsrv_features=tuple(self.config["observation"]["features"]),
                                                        **kwargs),
                                        coord_action_pairs)
            print("Simulation complete in ",str(time.time()-start)," seconds")
        else:
            # |DEBUG: Sequential execution test
            return_vals = []
            for st_act in list(coord_action_pairs):
                return_vals.append(self.simulateAction(st_act, sim_env=self._env, 
                                                            obsrv_features=tuple(self.config["observation"]["features"]),
                                                            **kwargs)
                                    )
        # |Populate MDP tables
        transitions = {}
        rewards = {}
        absorption = {}
        for init_state, action, next_state, trans_prob, reward, done, truncated, info in return_vals:
            logging.debug(str(init_state) + ' | ' + str(action) + ' | ' + str(next_state))
            #2# |Populate transitions table
            if (init_state, action) not in transitions:
                transitions[(init_state, action)] = {}
            #3# |Update next state as closest state from coordinates table before updating table
            closestCoord_dist, colsestCoord_indx = search_coordinates.query(next_state, 1)
            next_state = coordinate_list[colsestCoord_indx]
            if next_state not in transitions[(init_state, action)]:
                transitions[(init_state, action)][next_state] = trans_prob
            #2# |Populate reward functions
            if (init_state[0], action) in rewards:
                rewards[(init_state, action)] += reward * trans_prob
            else:
                rewards[(init_state, action)] = reward * trans_prob
            #2# |Set absorption states
            absorption[next_state] = done or truncated
        # Reset MDP matices and plan, since new table was calculated
        self._mdp_matrices = self._mdp_plan = None
        self._mdp_tables = MDPTable(start_state=start_state, transition=transitions, absorption=absorption, 
                            reward=reward, state_list=coordinate_list, action_list=action_list)
        return self._mdp_tables
    

    @classmethod
    # @lru_cache(maxsize=10000)
    def simulateAction(cls, action: int, sim_env, obsrv_features):
        """
        Given an instance of a Highway environment and an action to perform in the environment, the function
        will execute the action without making any changes to the original environment and return the
        next state and a copy of the updated environment instance.

        This function is implemented as a class methods to prevent multiprocessing from creating multiple copies
        of the class object for each process.

        This function returns 1 for transition probability since the highwayenv environment is completely
        deterministic.

        Args:
            action (int): The action to be taken expressed by an integer number.
            sim_env (gym environment): Whether to update the state of the current environment or simulate the action
                             without making changes to the active environment.
            obsrv_features (list): list of features returned by the gym environment as observation.
        """
        # raise NotImplementedError("Please Implement 'simulateAction' method in class "+cls.__name__)
        if not isinstance(sim_env, gym.wrappers.common.OrderEnforcing):
            raise TypeError("Environent parameter is not an instance of GymDiscreteMDP")
        env_copy = deepcopy(sim_env)
        obs, reward, done, truncated, info = env_copy.step(action)
        logging.debug(obs)
        next_state = cls.to_hashable_state(obs, obsrv_features)
        return((action, next_state, 1, reward, done, truncated, info, env_copy))


    def simulateTrajectory(self, action):
        """
        GIven an starting coordinate, the function will execute a series of actions based on the 
            calculated optimal policy for a gridowrls representation if the gym environment.

        Args:
            action (int): The action to be taken expressed by an integer number.
        """
        obs, reward, done, truncated, info = self._env.step(action)
        logging.debug(obs)
        # |Here, the classmethod 'to_hashable_state' is being called with self instead of the class name to allow the code 
        # | to call the overridden 'to_hashable_state' method implemented in any child class.
        self._current_state = self.to_hashable_state(obs, self.config["observation"]["features"])
        return self._current_state, reward, done, truncated, info


    @classmethod
    def to_hashable_state(cls, obs, obs_config):
        """
        Create hashable (discretized) variable using environment state information, for MDP planning.
        The current implementation does nothing, as the implementation of thi function is strictly 
            evironment dependant
        """
        return obs





######################### DEFINE OTHER DATA-CLASSES #########################


@dataclasses.dataclass(order=True)
class OptimalPolicy():
    inverse_temperature: float
    random_action_prob: float


    def __init__(self, mdp: GymDiscreteMDP, inverse_temperature: float=float('inf'), random_action_prob: float=0.0):
        if mdp.mdp_plan == None:
            raise ValueError("MDP planning result not set up for the class object.")
        self.inverse_temperature = inverse_temperature
        self.random_action_prob = random_action_prob
        self._state_value_dict = mdp.mdp_plan.state_values
        self._action_value_dict = mdp.mdp_plan.action_values


    def __call__(self, state: Tuple[frozendict]):
        qvalues = self.action_values(state)
        actions, values = zip(*qvalues.items())
        if not (self.inverse_temperature == float('inf') and self.random_action_prob == 0):
            probs = np.exp((np.array(values) - max(values))*self.inverse_temperature)
            probs /= probs.sum()
            probs = self.random_action_prob/len(actions) + (1-self.random_action_prob)*probs
            values = probs
        max_value = max(values)
        logging.debug("Action values: "+str(list(zip(actions, values))))
        return [a for a, v in zip(actions, values) if v == max_value]
    

    def action_values(self, state: Tuple[frozendict]) -> dict[int, float]:
        try:
            return self._action_value_dict[state]
        except KeyError:
            raise KeyError("State "+str(state)+" not explored.")
    

    def state_value(self, state: Tuple[frozendict]) -> float:
        try:
            return self._state_value_dict[state]
        except KeyError:
            raise KeyError("State "+str(state)+" not explored.")
    

    def __hash__(self):
        return hash((self.mdp, self.inverse_temperature, self.random_action_prob))
    

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, OptimalPolicy):
            return False
        return self.mdp == value.mdp and \
            self.inverse_temperature == value.inverse_temperature and \
            self.random_action_prob == value.random_action_prob