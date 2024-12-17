from highway_env.envs.discreteMDP_wrapper import HighwayDiscreteMDP

import cProfile
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)    # Other options: INFO, WARNING, ERROR, CRITICAL

import copy
import functools
import multiprocessing as mp
process_count = (mp.cpu_count()-2)
from pathos.multiprocessing import ProcessingPool as Pool    # Can run multiprocessing in interactive shell





config = {
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

highway_mdp = HighwayDiscreteMDP('highway-v0', config=config, render_mode='human')

initial_state, actions = highway_mdp.get_env_properties()





max_depth = 2   # The number of steps to plan ahead

def exeuteAction(action, envMDP: HighwayDiscreteMDP):
    """
    Given an instance of a gym environment and an action the environment will execute 
    the action and return the next state and a copy of the updated environment instance.
    """
    # print("In function")
    envMDP_copy = copy.deepcopy(envMDP)
    # logging.debug(envMDP_copy.env.unwrapped.road.vehicles[0])
    next_state, reward, done, truncated, info = envMDP_copy.step(action)
    logging.debug(' | '.join((state[0], action, next_state[0])))
    return((action, next_state, reward, done, truncated, info, envMDP_copy.copy_env()))

visited = set()
transitions = {}
frontier = {(initial_state, 0, highway_mdp.copy_env())}
while frontier:
    state, depth, curr_env = frontier.pop()
    visited.add(state)
    if depth < max_depth:
        highway_mdp.set_env(curr_env)
        print("calling function")
        # with mp.Pool(process_count) as pool:
        with Pool(process_count) as pool:
            return_vals = pool.map(functools.partial(exeuteAction, envMDP=highway_mdp), actions)
        for action, next_state, reward, done, truncated, info, updated_env in return_vals:
            if (state[0], action) not in transitions:
                transitions[(state[0], action)] = {}            
            if next_state[0] not in transitions[(state[0], action)]:
                transitions[(state[0], action)][next_state[0]] = 0
            transitions[(state[0], action)][next_state[0]] += 1
            if next_state not in visited:
                frontier.add((next_state, depth + 1, updated_env))
    MDPstatus = "Current Depth: " + str(depth) + " | Frontier: " + str(len(frontier)) +\
                " | Visited: " + str(len(visited)) + " | Transitions:" + str(len(transitions))
    logging.info(MDPstatus)

