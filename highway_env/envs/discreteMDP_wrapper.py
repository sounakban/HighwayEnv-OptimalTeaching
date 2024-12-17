import gymnasium as gym
import highway_env

from frozendict import frozendict
import numpy as np
import copy

import warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)    # Other options: INFO, WARNING, ERROR, CRITICAL


class GymDiscreteMDP:
    def __init__(self, *args, **kwargs):
        if "config" in kwargs:
            self.config = kwargs.get("config", None)
        self.env = gym.make(*args, **kwargs)
        self.obs, self.info = self.env.reset()
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.actions = range(action_space.start, action_space.n)
        else:
            raise NotImplementedError("Only discrete action spaces are currently supported")
            # |Can later be extended for other gym actionspaces


    def to_hashable_state(self, obs):
        """
        Create hashable variable using environment state information.
        """        
        raise NotImplementedError("Please Implement this method")

    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.
        Can be overridden based on simulation-specific properties.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        logging.debug(obs)
        next_state = self.to_hashable_state(obs)
        return next_state, reward, done, truncated, info

    def copy_env(self):
        """
        Return a copy of the current state of the environment,
        so it can be set back to simulate the outcome of various actions.
        """
        raise NotImplementedError("Please Implement this method")

    def set_env(self, env):
        """
        Set current environent state to the one passed in the argument.
        """
        raise NotImplementedError("Please Implement this method")

    @classmethod
    def mdp_factory(*args, gym_env, **kwargs):
        """
        
        """
        if gym_env == "highway":
            highwaymdp = HighwayDiscreteMDP(*args, **kwargs)    
            return highwaymdp
        return 0



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
        # self.action_dict = self.env.unwrapped.action_type.actions_indexes
        # |Set perception distance to maximum. So, state of all cars in the environment 
        # |are available irrespective of whether they are in the visibility window.
        self.env.unwrapped.PERCEPTION_DISTANCE = float('inf')
        self.initial_state = self.to_hashable_state(self.obs)
    
    def get_env_properties(self):
        return self.initial_state, self.actions

    # def step(self, action):
    #     # set_vehicles(state, self.env)
    #     obs, reward, done, truncated, info = self.env.step(action)
    #     logging.debug(action, obs[0])
    #     next_state = self.to_hashable_state(obs)
    #     return next_state, reward, done, truncated, info

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

    def copy_env(self):
        return copy.deepcopy(self.env)

    def set_env(self, env):
        self.env = copy.deepcopy(env)

    # def set_vehicles(self, vehicles):
    #     # |Would like to avoid using this funtion if possible.
    #     for v, new_v in zip(self.env.unwrapped.road.vehicles, vehicles):
    #         assert id(v) == new_v['id']
    #         v.position = np.array(new_v['position'])
    #         v.heading = new_v['heading']
    #         v.speed = new_v['speed']

    # def step(self, state, action):
    #     # |Avoid using this overloaded function, use super step funtion instead
    #     self.set_vehicles(state, self.env)
    #     obs, reward, done, truncated, info = self.env.step(self.action_dict[action])
    #     next_state = self.to_hashable_state(obs)
    #     return next_state, reward, done, truncated, info



def mdp_factory(*args, gym_env, **kwargs):
    if gym_env == "highway":
        highwaymdp = HighwayDiscreteMDP(*args, **kwargs)    
        return highwaymdp
    return 0
