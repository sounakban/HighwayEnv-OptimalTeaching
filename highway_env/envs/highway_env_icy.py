from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Ice1


Observation = np.ndarray


class HighwayEnvIcy(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        # # DEBUG: Print all vehicle positions
        # print([veh.position for veh in self.road.vehicles])

        # Add ice
        length, width = 15, 5
        for pos in ice_locations_subset:
            ice = Ice1(self.road, pos)
            ice.LENGTH, ice.WIDTH = (length, width)
            ice.diagonal = np.sqrt(ice.LENGTH**2 + ice.WIDTH**2)
            self.road.objects.append(ice)
        # TODO: Add logic for randomizing ice locations
        # length, width = 15, 5
        # for _ in range(self.config["ice_count"]):
        #     ice = Ice1.create_random(
        #             self.road, spacing=1 / self.config["vehicles_density"]
        #         )
        #     ice.LENGTH, ice.WIDTH = (length, width)
        #     ice.diagonal = np.sqrt(ice.LENGTH**2 + ice.WIDTH**2)
        #     self.road.objects.append(ice)



    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvIcyFast(HighwayEnvIcy):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                # "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False



class HighwayEnvIcyCustom(HighwayEnvIcy):
    """
    A variant of highway-v0 with custom execution configurations:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                # "lanes_count": 3,
                # "vehicles_count": 20,
                # "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False




# Predefined ice locations to start off
ice_locations = [[151.090512, 4], [172.45562252, 8], [195.84555251, 4], [218.85401083, 4], [244.21423732, 8], [268.52539651, 0], [294.43759937, 8], \
                 [317.3234107, 0], [343.98100414, 8], [368.8641181, 0], [392.53761495, 4], [415.93365471, 4], [437.50992641, 0], [460.59978171, 8], \
                 [482.42279356, 0], [505.44612845, 0], [527.55962487, 0], [551.07811977, 0], [575.47471293, 4], [598.6810601, 4], [620.76286179, 8], \
                 [152.28984434, 8], [176.47905523, 0], [200.81360755, 4], [226.48931567, 0], [251.46993398, 0], [274.03411667, 0], [295.55216614, 4], \
                 [317.65268166, 4], [339.70709028, 4], [365.08529909, 0], [386.54341389, 0], [409.07980441, 0], [432.77450139, 4], [455.24697461, 8], \
                 [480.15938788, 8], [505.65292351, 4], [527.77836731, 0], [550.54712234, 0], [574.27064148, 4], [599.12448343, 8], [623.887315, 4], \
                 [148.85346292, 0], [174.72989247, 8], [197.68861996, 8], [222.62100312, 4], [248.66586847, 4], [273.18419713, 0], [294.97381391, 8], \
                 [320.47937532, 0], [342.74130652, 0], [365.70929579, 8], [387.36808351, 0], [412.31466842, 0], [437.1464462, 0], [462.40385877, 4], \
                 [486.17421069, 4], [510.56525507, 4], [533.35388573, 8], [556.17651788, 4], [582.24095046, 0], [607.64325676, 0], [632.10601223, 8], \
                 [153.59833047, 8], [179.04649029, 4], [201.46458985, 8], [224.88475098, 4], [248.43447295, 4], [273.87161948, 4], [296.21920674, 4], \
                 [319.25837177, 8], [343.72797618, 8], [368.714841, 4], [393.41812949, 0], [414.77486297, 0], [441.81648307, 0], [464.07467143, 4], \
                 [485.39355587, 0], [510.81805766, 4], [536.10782126, 8], [561.21038789, 0], [584.16580601, 0], [608.02916678, 4], [633.43185766, 4]]

ice_locations_subset = [[151.090512, 4], [218.85401083, 8], [268.52539651, 0], [294.43759937, 0], [343.98100414, 4], [392.53761495, 8], [460.59978171, 4], \
                        [505.44612845, 0], [551.07811977, 8], [633.43185766, 0]]