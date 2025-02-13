# Allows importing modules from parent directory
import os
import sys
sys.path.append(os.getcwd())

from highway_env.envs.discreteMDP_wrapper import HighwayDiscreteMDP, OptimalPolicy

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)    # Other options: INFO, WARNING, ERROR, CRITICAL

import numpy as np
import itertools as it


if __name__ == '__main__':
    # Define Configuration
    num_of_vehicles = 2
    num_of_ice = 5
    env_length = 3000   # Max car speed (for MDP vehicle) is 30m/s At constant max speed, the simulation will last 100s in simulation time
    lane_count = 3
    config = {    
            ## Parameters of interest ##
            "observation": {
                # For more details about observation parameters check out "highway_env\envs\common\observation.py"
                "type": "Kinematics",
                "vehicles_count": num_of_vehicles+num_of_ice+5,   # Number of vehicles (and objects) to show in the observation. 
                                                                    #   Keep greater than value of vehicles out outside obervation
                                                                    #   dictionary to observe all vehicles in the environment.
                "features": ["presence", "x", "y", "vx", "vy"],# "heading"],
                "normalize": False, # Normalize object coordinates
                "absolute": True,   # Provide absolute coordinate of vehicles
                "order": "sorted",
                "observe_intentions": False,
                "include_obstacles": True,
                "see_behind": True  # Report vehicles behind the ego vehicle
                },
            ## Parameters specialized for the icy highway environment ##
            "ice_count": num_of_ice,    # Number of ice sheets in the environment
            "env_len":  env_length,    # Length of the road
            ## Keep these to default, because the fast versions of the environments implement different values ##
            ## of these variables for faster execution ##
            "vehicles_count": num_of_vehicles,
            "lanes_count": lane_count,
            "simulation_frequency": 5,
            "duration": (env_length/20)+5,  # [in simulation seconds], minimum speed for MDP vehicle is 20m/s, with extra 5s
            "disable_collision_checks": True,    # Check collisions for other vehicles
            "enable_lane_change": False,
            ## Other parameters aleady set by default configurations ##
            # "action": {
            #     "type": "DiscreteMetaAction",
            # },
            # "controlled_vehicles": 1,
            # "initial_lane_id": None,
            # "ego_spacing": 2,
            # "vehicles_density": 1,
            # "collision_reward": -1,  # The reward received when colliding with a vehicle.
            # "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # # zero for other lanes.
            # "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # # lower speeds according to config["reward_speed_range"].
            # "lane_change_reward": 0,  # The reward received at each lane change action.
            # "reward_speed_range": [20, 30],
            # "normalize_reward": True,
            # "offroad_terminal": False
            }
    # highway_mdp = HighwayDiscreteMDP('highway-v0', config=config, render_mode='rgb_array')
    # highway_mdp = HighwayDiscreteMDP('highway-v0', config=config, render_mode=None)
    # highway_mdp = HighwayDiscreteMDP('highway-fast-v0', config=config, render_mode=None)
    # highway_mdp = HighwayDiscreteMDP('highway-icy-v0', config=config, render_mode=None)
    # highway_mdp = HighwayDiscreteMDP('highway-icy-fast-v0', config=config, render_mode=None)
    highway_mdp = HighwayDiscreteMDP('highway-icy-custom-v0', config=config, render_mode=None)


    # Create discrete world grid
    temp = highway_mdp.env.unwrapped.road.network.lanes_list()
    x_max = max([lane.end[0] for lane in temp])
    y_max = max([lane.end[1] for lane in temp])
    #2# |Divide the environment into 600 blocks along the x-axis
    x_grid = np.linspace(0, x_max, 600)
    y_grid = np.linspace(0, y_max, lane_count)
    state_list = list(it.product(x_grid, y_grid))
    [x_max, y_max]


    # veh_speed for MDP vehicle can be 20/25/30
    temp = highway_mdp.populate_MDPtable_StaticGridworld(state_list, veh_speed = 25)