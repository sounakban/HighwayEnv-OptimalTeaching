import gymnasium as gym
import highway_env

import dataclasses
from frozendict import frozendict
import numpy as np
import copy

from pathos.multiprocessing import ProcessingPool as Pool    # Can run multiprocessing in interactive shell
import multiprocessing as mp
process_count = (mp.cpu_count()-2)
import functools

import warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)    # Other options: INFO, WARNING, ERROR, CRITICAL

from typing import TypeVar, Generic, Sequence, Set, Hashable, Union, Callable, Tuple, Mapping


class GymDiscreteMDP:
    max_states = int(1e6)
    discount_rate = 1.0


    def __init__(self, *args, **kwargs):
        self._mdp_tables = None
        self._mdp_matrices = None
        self._mdp_plan = None
        if "config" in kwargs:
            self.config = kwargs.get("config", None)
        self._env = gym.make(*args, **kwargs)
        self.obs, self.info = self._env.reset()
        self._initial_state = self.to_hashable_state(self.obs)
        self._curr_state = copy.deepcopy(self._initial_state)
        action_space = self._env.action_space
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self._actions = range(action_space.start, action_space.n)
        else:
            raise NotImplementedError("Only discrete action spaces are currently supported")
            # |Can later be extended for other gym actionspaces


    @property
    def mdp_tables(self): 
        """
        Return the current MDPtables object.
        """
        return self._mdp_tables


    @property
    def mdp_matrices(self): 
        """
        Return the current MDPmatrices object.
        """
        if self._mdp_matrices == None:
            self._mdp_matrices = self.get_MDPmatrices()
        return self._mdp_matrices


    @property
    def mdp_plan(self): 
        """
        Return the current MDPplan object.
        """
        if self._mdp_plan == None:
            self._mdp_plan = self.value_iteration()
        return self._mdp_plan
       

    @property
    def initial_state(self): 
        """
        Return the current MDPtable object.
        """
        return self._initial_state
       

    @initial_state.setter 
    def initial_state(self, initial_state): 
        """
        Set the current MDPtable object.
        """
        self._initial_state = initial_state
       

    @property
    def current_state(self): 
        """
        Return the current MDPtable object.
        """
        return self._current_state
       

    @initial_state.setter 
    def current_state(self, current_state): 
        """
        Set the current MDPtable object.
        """
        self._current_state = current_state


    @property
    def actions(self): 
        """
        Return the current MDPtable object.
        """
        return self._actions


    @property
    def env(self):
        """
        Return a copy of the current state of the environment,
        so it can be set back to simulate the outcome of various actions.
        """
        raise NotImplementedError("Please Implement this method")


    @env.setter
    def env(self, env):
        """
        Set current environent state to the one passed in the argument.
        """
        raise NotImplementedError("Please Implement this method")


    def to_hashable_state(self, obs):
        """
        Create hashable (discretized) variable using environment state information.
        """        
        raise NotImplementedError("Please Implement this method")


    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.
        Can be overridden based on simulation-specific properties.

        This function returns 1 for transition probability since a standard gym 
        environment is completely deterministic.
        """
        obs, reward, done, truncated, info = self._env.step(action)
        logging.debug(obs)
        next_state = self.to_hashable_state(obs)
        self._current_state = next_state    # Update intital state to reflect updated environment state.
        return next_state, 1, reward, done, truncated, info
    

    def create_MDPTable(self, transition, absorption, reward, state_list, action_list):
        """
        Set the current MDPtable object.
        """
        self._mdp_tables = MDPTable(transition=transition, absorption=absorption, reward=reward, state_list=state_list, action_list=action_list)
   

    def populate_MDPtable(self, max_depth: int = 2, unknown_reward: float = 0):
        """
        Code to set up MDP tables

        Parameters:
            max_depth (int): The number of steps to plan ahead (default = 2)
            unknown_reward (float): reward value set for state-action pairs which are left
                                    unexplored due to depth constraint (default = 0)
        """        
        raise NotImplementedError("Please Implement this method")
    

    def get_MDPmatrices(self):
        """
        Converts MDP tables to mtrices useful for value calculation process

        Parameters:
            None
        """
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
        return  MDPMatrix(
            transition=transition_mat,
            reward=reward_mat,
            absorbing_state_mask=absorbing_vec,
            discount=self.discount_rate,
            state_list=tb.state_list,
            action_list=tb.action_list
        )
        

    def value_iteration(
            self,
            iterations: int = 100,
            value_epsilon: float = 1e-6
        ):
        """
        Perform value iteration on MDP tables

        Parameters:
            iterations (int): Max number of iterations to calculate state and action 
                                values (default = 100)
            value_epsilon (float): Tollerance for when to terminate iteration process
                                    based on delta between consecutive iterations (default = 1e-6)
        """
        m = self.get_MDPmatrices()
        # if np.all(m.absorbing_state_mask == False) and m.discount == 1:
        #     raise ValueError("No absorbing states found in MDP with discount factor 1")
        value = np.zeros(len(m.state_list))
        value_ = np.zeros(len(m.state_list))
        qvalues = np.zeros((len(m.state_list), len(m.action_list)))
        for i in range(iterations):
            qvalues[:] = m.reward + m.discount*np.einsum("san,n->sa", m.transition, value)
            # qvalues[m.absorbing_state_mask, :] = 0
            value_[:] = qvalues.max(axis=1)
            max_residual = np.abs(value - value_).max()
            if max_residual < value_epsilon:
                break
            value[:] = value_
        assert max_residual < value_epsilon, "Value iteration did not converge"
        action_values = {}
        for s, q in zip(m.state_list, qvalues):
            action_values[s] = dict(zip(m.action_list, q))
        logging.debug(value)
        logging.debug(action_values)
        return PlanningResult(
            state_values=dict(zip(m.state_list, value.tolist())),
            action_values=action_values
        )


    @classmethod
    def mdp_factory(cls, *args, gym_env, **kwargs):
        """
        Return gym environment object based on selection
        """
        if gym_env == "highway":
            highwaymdp = HighwayDiscreteMDP(*args, **kwargs)    
            return highwaymdp
        return None


    @classmethod
    def exeuteAction(cls, action, envMDP):
        """
        Given an instance of a gym environment and an action the environment will execute 
        the action and return the next state and a copy of the updated environment instance.
        """
        if not isinstance(envMDP, cls):
            raise TypeError("Environent parameter is not an instance of GymDiscreteMDP")
        envMDP_copy = copy.deepcopy(envMDP)
        # logging.debug(envMDP_copy.env.unwrapped.road.vehicles[0])
        next_state, trans_prob, reward, done, truncated, info = envMDP_copy.step(action)
        return((action, next_state, trans_prob, reward, done, truncated, info, envMDP_copy))






class HighwayDiscreteMDP(GymDiscreteMDP):
    '''
    The class currently only supports kinematics observation space in HighwayEnv
    '''
    def __init__(self, *args, **kwargs):
        if (not "config" in kwargs 
            or not "observation" in kwargs["config"]
            or not "features" in kwargs["config"]["observation"]
            or not "Kinematics" == kwargs["config"]["observation"]["type"]):
            kwargs["config"] = self.default_config()
            warnings.warn("Config not specified/does not match requirement. USING DEFAULT CONFIG.\n \
                  To use custom config, please use Kinematics observation space, \
                  and specify (at least) the following features in config:\n \
                  \tpresence, x, y, vx, vy, heading.")
        super().__init__(*args, **kwargs)
        # self.action_dict = self._env.unwrapped.action_type.actions_indexes
        # |Set perception distance to maximum. So, state of all cars in the environment 
        # |are available irrespective of whether they are in the visibility window.
        self._env.unwrapped.PERCEPTION_DISTANCE = float('inf')
    
    
    @property
    def env(self):
        return copy.deepcopy(self._env)


    @env.setter
    def env(self, env):
        self._env = copy.deepcopy(env)
    

    def default_config(self):
        return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 50,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
            "absolute": True,
            "order": "sorted",
            "observe_intentions": False,
            "include_obstacles": True
            }
        }


    def get_env_properties(self):
        return self.initial_state, self.actions


    def to_hashable_state(self, obs):
        road_objects = []
        for road_obj in obs:
            feature_vals = {k: v for k,v in zip(self.config["observation"]["features"], road_obj)}
            veh = {}
            veh["position"] = tuple(np.round((feature_vals["x"],feature_vals["y"]), 2))
            veh["speed"] = tuple(np.round((feature_vals["vx"],feature_vals["vy"]), 2))
            veh["heading"] = np.round(feature_vals["heading"], 3)
            road_objects.append(frozendict(veh))
        return tuple(road_objects)


    # def step(self, action):
    #     # set_vehicles(state, self._env)
    #     obs, reward, done, truncated, info = self._env.step(action)
    #     logging.debug(action, obs[0])
    #     next_state = self.to_hashable_state(obs)
    #     return next_state, reward, done, truncated, info
   

    def populate_MDPtable(self, max_depth: int = 2, unknown_reward: float = 0):
        """
        Code to set up MDP tables

        Parameters:
            max_depth (int): The number of steps to plan ahead (default = 2)
            unknown_reward (float): reward value set for state-action pairs which are left
                                    unexplored due to depth constraint (default = 0)
        """
        start_env = (self.initial_state, 0, self)
        visited = set()
        transitions = {}
        rewards = {}
        absorption = {}
        frontier = {start_env}
        env2close = set()   # Maintain list of environments to close
        while frontier:
            state, depth, curr_MDPstate = frontier.pop()
            visited.add(state)
            if len(visited) >= self.max_states:
                raise ValueError(f"Maximum number of states reached ({self.max_states})")
            env2close.add(curr_MDPstate)
            if depth < max_depth:
                with Pool(process_count) as pool:
                    return_vals = pool.map(functools.partial(HighwayDiscreteMDP.exeuteAction, envMDP=curr_MDPstate), self.actions)
                for action, next_state, trans_prob, reward, done, truncated, info, updated_MDPstate in return_vals:
                    logging.debug(str(state[0]) + ' | ' + str(action) + ' | ' + str(next_state[0]))
                    #2# Populate transitions table
                    if (state, action) not in transitions:
                        transitions[(state, action)] = {}
                    if next_state not in transitions[(state, action)]:
                        transitions[(state, action)][next_state] = trans_prob
                    #2# Populate reward functions
                    if (state[0], action) in rewards:
                        rewards[(state, action)] += reward * trans_prob
                    else:
                        rewards[(state, action)] = reward * trans_prob
                    #2# set absorption states
                    absorption[next_state] = done or truncated
                    #2# Populate visited
                    if next_state not in visited:
                        frontier.add((next_state, depth + 1, updated_MDPstate))
                    else:
                        updated_MDPstate.env.close()
            else:
                #2# Set rewards for unexplored state-action pairs to 0
                for action in self.actions:
                    if not (state, action) in rewards:
                        rewards[(state, action)] = unknown_reward
                MDPstatus = "Current Depth: " + str(depth) + " | Frontier: " + str(len(frontier)) +\
                            " | Visited: " + str(len(visited)) + " | Transitions:" + str(len(transitions))
                logging.info(MDPstatus)
        env2close.discard(start_env[2]) #Keep the first step active
        for curr_MDPstate in env2close:
            # close all duplicate environments
            curr_MDPstate.env.close()
        env2close = set()   # Close ununsed environments at the end of the loop
        self.create_MDPTable(transition=transitions, absorption=absorption, reward=rewards, 
                             state_list=visited, action_list=set(self.actions))
        

    def copy_class_with_env(self, env):
        curr_copy = copy.deepcopy(self)
        curr_copy.env = env
        return curr_copy

    def get_construals_singleobj(self):        
        veh_list = self._env.unwrapped.road.vehicles[:]     # Only contains list of vehicles
        ego_veh = veh_list[0]
        veh_list = veh_list[1:]                             # Update list to only contain ado vehicles
        obj_list = veh_list[:]                              # Contains list of all objects
        obj_list.extend(self._env.unwrapped.road.objects)
        contrued_envs = []
        # Add an empty environment
        env_copy = self.env
        env_copy.unwrapped.road.vehicles = [ego_veh]
        env_copy.unwrapped.road.objects = []
        contrued_envs.append(self.copy_class_with_env(env_copy))
        # Add other environments
        for obj in obj_list:
            env_copy = self.env
            if obj in veh_list:
                # Object is a vehicle
                env_copy.unwrapped.road.vehicles = [ego_veh, obj]
                env_copy.unwrapped.road.objects = []
            else:
                # Object is roadobject
                env_copy.unwrapped.road.vehicles = [ego_veh]
                env_copy.unwrapped.road.objects = [obj]
            # env_copy.unwrapped.road.vehicles.remove(veh)
            contrued_envs.append(self.copy_class_with_env(env_copy))
        return contrued_envs


    # def set_vehicles(self, vehicles):
    #     # |Would like to avoid using this funtion if possible.
    #     for v, new_v in zip(self._env.unwrapped.road.vehicles, vehicles):
    #         assert id(v) == new_v['id']
    #         v.position = np.array(new_v['position'])
    #         v.heading = new_v['heading']
    #         v.speed = new_v['speed']


    # def step(self, state, action):
    #     # |Avoid using this overloaded function, use super step funtion instead
    #     self.set_vehicles(state, self._env)
    #     obs, reward, done, truncated, info = self._env.step(self.action_dict[action])
    #     next_state = self.to_hashable_state(obs)
    #     return next_state, reward, done, truncated, info






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
    



@dataclasses.dataclass
class MDPTable():
    transition: dict[dict[Tuple[Tuple[frozendict], int], Tuple[frozendict]], float]
    absorption: dict[Tuple[frozendict], bool]
    reward: dict[Tuple[Tuple[frozendict], int], float]
    state_list: Sequence[Tuple[frozendict]]
    action_list: Sequence[int]


@dataclasses.dataclass
class MDPMatrix():
    transition: np.ndarray
    reward: np.ndarray
    absorbing_state_mask: np.ndarray
    discount: float
    state_list: Sequence[Tuple[frozendict]]
    action_list: Sequence[int]


@dataclasses.dataclass
class PlanningResult():
    state_values: dict[Tuple[frozendict], float]
    action_values: dict[Tuple[frozendict], dict[int, float]]