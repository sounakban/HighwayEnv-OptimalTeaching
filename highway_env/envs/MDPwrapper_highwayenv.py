from .MDPwrapper_Gym import GymDiscreteMDP, GymGridworldMDP
import highway_env

from frozendict import frozendict
import numpy as np
import scipy as scp
from math import isnan
from copy import deepcopy

from pathos.multiprocessing import ProcessingPool as Pool    # Can run multiprocessing in interactive shell
import multiprocessing as mp
process_count = (mp.cpu_count()-2)
from functools import lru_cache
import itertools as it

import warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)    # Other options: INFO, WARNING, ERROR, CRITICAL

from typing import TypeVar, Generic, Sequence, Set, Hashable, Union, Callable, Tuple, Mapping







######################### DEFINE HIGHWAYENV DISCRETE-MDP WRAPPER CLASS #########################

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
        self._first_state = self.obs_to_hashable(self._first_state, self.config["observation"]["features"])
        self._current_state = self.obs_to_hashable(self._current_state, self.config["observation"]["features"])
        # self.action_dict = self._env.unwrapped.action_type.actions_indexes
        # |Set perception distance to maximum. So, state of all cars in the environment 
        # |are available irrespective of whether they are in the visibility window.
        self._env.unwrapped.PERCEPTION_DISTANCE = float('inf')
        

    def default_config(self):
        return {
        "observation": {
            "vehicles_count": 50,   # Number of vehicles to show in the observation. Keep greater than value 
                                    #   of vehicles out outside obervation dictionary to observe all vehicles
                                    #   in the environment.
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": False, # Normalize object coordinates
            "absolute": True,   # Provide absolute coordinate of vehicles
            "order": "sorted",
            "observe_intentions": False,
            "include_obstacles": True,
            "see_behind": True  # Report vehicles behind the ego vehicle
            }
        }
    

    def get_obsFeatures(self, ):
        return self.config["observation"]["features"]
        

    def get_construals_singleobj(self):
        '''
        Returns a list of environments each containing the ego vehicle along with a single object or vehicle.
        Te first environment in the list '''
        veh_list = self._env.unwrapped.road.vehicles[:]         # Only contains list of vehicles
        ego_veh = veh_list[0]
        veh_list = set(veh_list[1:])                            # Update list to only contain ado vehicles
        obj_list = set(self._env.unwrapped.road.objects)        # Contains list of all objects
        obj_list = obj_list.union(veh_list)                     # Add vehicles to object list
        # print(obj_list, veh_list)
        contrued_envs = []
        # Add an empty environment
        curr_construal = self.get_copy()
        temp_env = curr_construal.env
        temp_env.unwrapped.road.vehicles = [ego_veh]
        temp_env.unwrapped.road.objects = []
        contrued_envs.append(curr_construal)
        # Add other environments
        for obj in obj_list:
            curr_construal = self.get_copy()
            temp_env = curr_construal.env
            if obj in veh_list:
                # Object is a vehicle
                temp_env.unwrapped.road.vehicles = [ego_veh, obj]
                temp_env.unwrapped.road.objects = []
            else:
                # Object is roadobject
                temp_env.unwrapped.road.vehicles = [ego_veh]
                temp_env.unwrapped.road.objects = [obj]
            # env_copy.unwrapped.road.vehicles.remove(veh)
            contrued_envs.append(curr_construal)
        return contrued_envs


    @classmethod
    def obs_to_hashable(cls, obs, **kwargs):
        """
        Given an observation (obtained from a step function), extract relevant features and store in 
            a hashable (discretized) variable.

        Args:
            obs (Gym Observation space): The environment where the agent state needs to be set.

        Kwrgs:
            obs_config (list): The list of features in the observation values returned by the step function
        """
        obs_config = kwargs.get("obs_config", None)
        if not obs_config:
            raise ValueError("Please provide obs_config as argument")
        road_objects = []
        for road_obj in obs:
            feature_vals = {k: v for k,v in zip(obs_config, road_obj)}
            veh = {}
            veh["position"] = tuple(np.round((feature_vals["x"],feature_vals["y"]), 2))
            veh["speed"] = tuple(np.round((feature_vals["vx"],feature_vals["vy"]), 2))
            # Replace NaN values of headings with -1, allows comparison of states.
            veh["heading"] = np.round(feature_vals.get("heading", -1))
            road_objects.append(frozendict(veh))
        return tuple(road_objects)
        

    def setState_allVeh(self, vehicles):
        for v, new_v in zip(self._env.unwrapped.road.vehicles, vehicles):
            assert id(v) == new_v['id']
            v.position = np.array(new_v['position'])
            v.heading = new_v['heading']
            v.speed = new_v['speed']
       

    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.

        Args:
            action (int): The action to be taken expressed by an integer number.
        """
        obs, reward, done, truncated, info = self._env.step(action)
        logging.debug(obs)
        # |Here, the classmethod 'obs_to_hashable' is being called with self instead of the class name to allow the code 
        # | to call the overridden 'obs_to_hashable' method implemented in any child class.
        self._current_state = self.obs_to_hashable(obs, obs_config = self.config["observation"]["features"])
        return self._current_state, reward, done, truncated, info
        
    @classmethod
    def setState_egoVeh(self, v, new_coord, veh_speed):
        v.position = np.array(new_coord)
        v.heading = 0
        v.speed = veh_speed
        v.target_lane_index = (v.target_lane_index[0], v.target_lane_index[1], int(new_coord[1]/4))
        v.target_speed = veh_speed





######################### DEFINE HIGHWAYENV GYM-MDP WRAPPER CLASS #########################

class HighwayGridworldMDP(GymGridworldMDP):
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
        self._first_state = self.obs_to_hashable(self._first_state, obs_config = self.config["observation"]["features"])
        self._current_state = self.obs_to_hashable(self._current_state, obs_config = self.config["observation"]["features"])
        # self.action_dict = self._env.unwrapped.action_type.actions_indexes
        # |Set perception distance to maximum. So, state of all cars in the environment 
        # |are available irrespective of whether they are in the visibility window.
        self._env.unwrapped.PERCEPTION_DISTANCE = float('inf')
        

    def default_config(self):
        return {
        "observation": {
            "vehicles_count": 50,   # Number of vehicles to show in the observation. Keep greater than value 
                                    #   of vehicles out outside obervation dictionary to observe all vehicles
                                    #   in the environment.
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": False, # Normalize object coordinates
            "absolute": True,   # Provide absolute coordinate of vehicles
            "order": "sorted",
            "observe_intentions": False,
            "include_obstacles": True,
            "see_behind": True  # Report vehicles behind the ego vehicle
            }
        }
    
    
    def get_obsFeatures(self, ):
        return self.config["observation"]["features"]
        

    def get_construals_singleobj(self):
        '''
        Returns a list of environments each containing the ego vehicle along with a single object or vehicle.
        Te first environment in the list 
        '''
        veh_list = self._env.unwrapped.road.vehicles[:]         # Only contains list of vehicles
        ego_veh = veh_list[0]
        veh_list = set(veh_list[1:])                            # Update list to only contain ado vehicles
        obj_list = set(self._env.unwrapped.road.objects)        # Contains list of all objects
        obj_list = obj_list.union(veh_list)                     # Add vehicles to object list
        # print(obj_list, veh_list)
        contrued_envs = []
        # Add an empty environment
        curr_construal = self.get_copy()
        temp_env = curr_construal.env
        temp_env.unwrapped.road.vehicles = [ego_veh]
        temp_env.unwrapped.road.objects = []
        contrued_envs.append(curr_construal)
        # Add other environments
        for obj in obj_list:
            curr_construal = self.get_copy()
            temp_env = curr_construal.env
            if obj in veh_list:
                # Object is a vehicle
                temp_env.unwrapped.road.vehicles = [ego_veh, obj]
                temp_env.unwrapped.road.objects = []
            else:
                # Object is roadobject
                temp_env.unwrapped.road.vehicles = [ego_veh]
                temp_env.unwrapped.road.objects = [obj]
            # env_copy.unwrapped.road.vehicles.remove(veh)
            contrued_envs.append(curr_construal)
        return contrued_envs
        

    def setState_allVeh(self, vehicles):
        for v, new_v in zip(self._env.unwrapped.road.vehicles, vehicles):
            assert id(v) == new_v['id']
            v.position = np.array(new_v['position'])
            v.heading = new_v['heading']
            v.speed = new_v['speed']
       

    def step(self, action):
        """
        The function takes an action and returns MDP-compatible state information.

        Args:
            action (int): The action to be taken expressed by an integer number.
        """
        obs, reward, done, truncated, info = self._env.step(action)
        logging.debug(obs)
        # |Here, the classmethod 'obs_to_hashable' is being called with self instead of the class name to allow the code 
        # | to call the overridden 'obs_to_hashable' method implemented in any child class.
        self._current_state = self.obs_to_hashable(obs, obs_config = self.config["observation"]["features"])
        return self._current_state, reward, done, truncated, info


    @classmethod
    def obs_to_hashable(cls, obs, **kwargs):
        """
        Given an observation (obtained from a step function), extract relevant features and store in 
            a hashable (discretized) variable.

        Args:
            obs (Gym Observation space): The observation space from which the agent state needs to be extracted.

        Kwrgs:
            obs_config (list): The list of features in the observation values returned by the step function
        """
        obs_config = kwargs.get("obs_config", None)
        if not obs_config:
            raise ValueError("Please provide value for obs_config argument")
        road_objects = []
        for road_obj in obs:
            feature_vals = {k: v for k,v in zip(obs_config, road_obj)}
            veh = {}
            veh["position"] = (feature_vals["x"],feature_vals["y"])
            veh["speed"] = (feature_vals["vx"],feature_vals["vy"])
            # Replace NaN values of headings with -1, allows comparison of states.
            veh["heading"] = feature_vals.get("heading", -1)
            road_objects.append(frozendict(veh))
        return road_objects[0]["position"] + (road_objects[0]["speed"][0],)
        return tuple(road_objects)
    

    def coordinate_to_state(self, coord_list):
        """
        Takes a list of starting coordinates as inputs and returns a list of all possible states based on 
        all possible combinations of environment features.
        """
        ego_vehicle = self._env.unwrapped.road.vehicles[0]
        if ego_vehicle.__class__.__name__ == "MDPVehicle":
            permissible_speeds = ego_vehicle.DEFAULT_TARGET_SPEEDS
        else:
            raise ValueError("The current implementation only works with MDP ego vehicles")
        updated_state_list = [coord+(spd,) for coord, spd in it.product(coord_list,permissible_speeds)]
        return updated_state_list

    @classmethod
    def get_agentState(cls, env, **kwargs):
        """
        Given an observation (obtained from a step function), extract relevant features and store in 
            a hashable (discretized) variable.

        Args:
            env (Gym Environment): The environment from which the agent state needs to be extracted.
        """
        return env.unwrapped.road.vehicles[0]["position"]
            

    @classmethod
    def set_agentState(cls, env, target_state, **kwargs):
        """
        GIven an an environment and agent properties, set the state of the agent.

        Args:
            env (Gym Environment): The environment where the agent state needs to be set.
            target_state: the state to which the agent has to be set.

        Kwrgs:
            speed (int): The speed to set for the ego vehicle
        """
        if not target_state:
            raise ValueError("Starting coordinate for vehicle not provided")
        v = env.unwrapped.road.vehicles[0]
        new_coord = target_state[:2]
        new_speed = target_state[2]
        # new_speed = kwargs.get("speed", None)
        # if not new_speed:
        #     new_speed = v.target_speed
        #     logging.info("Maintaining current vehicle speed. For custom speed, provide \'speed\' argument.")
        new_coord = np.array(new_coord).astype(float)
        v.position = new_coord
        v.heading = 0
        v.speed = new_speed
        v.target_lane_index = (v.target_lane_index[0], v.target_lane_index[1], int(new_coord[1]/4))
        v.target_speed = new_speed
        # print(new_env.unwrapped.road.vehicles)
        # print("-------------------")
        return env


