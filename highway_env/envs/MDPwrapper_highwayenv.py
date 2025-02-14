from .MDPwrapper_Gym import GymDiscreteMDP, GymGridworldMDP
import highway_env

from frozendict import frozendict
import numpy as np
import scipy as scp
from math import isnan
from copy import deepcopy
import time

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
        self._first_state = self.to_hashable_state(self._first_state, self.config["observation"]["features"])
        self._current_state = self.to_hashable_state(self._current_state, self.config["observation"]["features"])
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
    def to_hashable_state(cls, obs, obs_config):
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
        self._first_state = self.to_hashable_state(self._first_state, self.config["observation"]["features"])
        self._current_state = self.to_hashable_state(self._current_state, self.config["observation"]["features"])
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
    def to_hashable_state(cls, obs, obs_config):
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


    @classmethod
    def simulateAction(cls, coord_action, sim_env = None, obsrv_features = None, **kwargs):
        """
        Given an instance of a gym environment and a sequence of actions to perform in the environment, 
        the function will execute the action sequence without making any changes to the original environment
        and return all intermediary states and a copy of the updated environment instance.

        This function is implemented as a class methods to prevent multiprocessing from creating multiple copies
        of the class object for each process.

        Args:
            sim_env (gym environment): Whether to update the state of the current environment or simulate the action
                             without making changes to the active environment.
            obsrv_features (list): list of features returned by the gym environment as observation.
            state_action: The initial state and action pair to sumulate
        """
        curr_coord, curr_action = coord_action
        new_env = deepcopy(sim_env)
        v = new_env.unwrapped.road.vehicles[0]
        veh_speed = kwargs.get("veh_speed", v.MAX_SPEED)
        cls.setState_egoVeh(v, curr_coord, veh_speed)
        obs, reward, done, truncated, info = new_env.step(curr_action)
        # print(new_env.unwrapped.road.vehicles)
        # print("-------------------")
        next_coord = new_env.unwrapped.road.vehicles[0].position
        return (curr_coord, curr_action, next_coord, 1, reward, done, truncated, info)
        

    def setState_allVeh(self, vehicles):
        for v, new_v in zip(self._env.unwrapped.road.vehicles, vehicles):
            assert id(v) == new_v['id']
            v.position = np.array(new_v['position'])
            v.heading = new_v['heading']
            v.speed = new_v['speed']
        

    @classmethod
    def setState_egoVeh(self, v, new_coord, veh_speed):
        v.position = np.array(new_coord)
        v.heading = 0
        v.speed = veh_speed
        v.target_lane_index = (v.target_lane_index[0], v.target_lane_index[1], int(new_coord[1]/4))
        v.target_speed = veh_speed


