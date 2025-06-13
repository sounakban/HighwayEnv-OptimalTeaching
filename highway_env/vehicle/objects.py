from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np

from highway_env import utils


if TYPE_CHECKING:
    from highway_env.road.lane import AbstractLane
    from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]


class RoadObject(ABC):
    """
    Common interface for objects that appear on the road.

    For now we assume all objects are rectangular.
    """

    LENGTH: float = 2  # Object length [m]
    WIDTH: float = 2  # Object width [m]

    def __init__(
        self,
        road: Road,
        position: Sequence[float],
        heading: float = 0,
        speed: float = 0
    ):
        """
        :param road: the road instance where the object is placed in
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        :param speed: cartesian speed of object in the surface
        """
        self.road = road
        self.position = np.array(position, dtype=np.float64)
        self.heading = heading
        self.speed = speed
        self.lane_index = (
            self.road.network.get_closest_lane_index(self.position, self.heading)
            if self.road
            else np.nan
        )
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None

        # Enable collision with other collidables
        self.collidable = True

        # Collisions have physical effects
        self.solid = True

        # If False, this object will not check its own collisions, but it can still collide with other objects that do
        # check their collisions.
        self.check_collisions = True
        
        # Objects slip when they intersect with ice
        self.slipped = None

        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(self.position.shape)

    @classmethod
    def make_on_lane(
        cls,
        road: Road,
        lane_index: LaneIndex,
        longitudinal: float,
        speed: float | None = None,
    ) -> RoadObject:
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: a road object containing the road network
        :param lane_index: index of the lane where the object is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: a RoadObject at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(
            road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed
        )

    def handle_collisions(self, other: RoadObject, dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        if will_intersect:
            if self.solid and other.solid:
                if isinstance(other, Obstacle):
                    self.impact = transition
                elif isinstance(self, Obstacle):
                    other.impact = transition
                else:
                    self.impact = transition / 2
                    other.impact = -transition / 2

        if intersecting:
            if self.solid and other.solid:
                self.crashed = True
                other.crashed = True
            elif (self.solid and not other.solid) or (not self.solid and other.solid):
                # If one is solid, the other is not                
                if not (type(self).__name__ == "IDMVehicle" or type(other).__name__ == "IDMVehicle"):
                    # Slip is only implemented for ego vehicles
                    if type(other).__name__.__contains__("Ice"):
                        self.slipped = other
                    elif type(self).__name__.__contains__("Ice"):
                        other.slipped = self
            if not self.solid:
                self.hit = True
            if not other.solid:
                other.hit = True

    def _is_colliding(self, other, dt):
        # Fast spherical pre-check
        if (
            np.linalg.norm(other.position - self.position)
            > (self.diagonal + other.diagonal) / 2 + self.speed * dt
        ):
            return (
                False,
                False,
                np.zeros(
                    2,
                ),
            )
        # Accurate rectangular check
        return utils.are_polygons_intersecting(
            self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt
        )

    # Just added for sake of compatibility
    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": 0.0,
            "vy": 0.0,
            "cos_h": np.cos(self.heading),
            "sin_h": np.sin(self.heading),
            "cos_d": 0.0,
            "sin_d": 0.0,
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    def polygon(self) -> np.ndarray:
        points = np.array(
            [
                [-self.LENGTH / 2, -self.WIDTH / 2],
                [-self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, -self.WIDTH / 2],
            ]
        ).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def lane_distance_to(self, other: RoadObject, lane: AbstractLane = None) -> float:
        """
        Compute the signed distance to another object along a lane.

        :param other: the other object
        :param lane: a lane
        :return: the distance to the other other [m]
        """
        if not other:
            return np.nan
        if not lane:
            lane = self.lane
        return (
            lane.local_coordinates(other.position)[0]
            - lane.local_coordinates(self.position)[0]
        )

    @property
    def on_road(self) -> bool:
        """Is the object on its current lane, or off-road?"""
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other: RoadObject) -> float:
        return self.direction.dot(other.position - self.position)

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):
    """Obstacles on the road."""

    def __init__(
        self, road, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid = True

    @classmethod
    def create_random(
        cls,
        road: Road,
        lane_from: str | None = None,
        lane_to: str | None = None,
        lane_id: int | None = None,
        spacing: float = 1,
    ) -> Ice:
        """
        Create a random obstacle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: An Ice object with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id or road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        default_spacing = 12
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(ic.position)[0] for ic in road.objects])
            if len(road.objects)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        ic = cls(road, lane.position(x0, 0))
        return ic


class Landmark(RoadObject):
    """Landmarks of certain areas on the road that must be reached."""

    def __init__(
        self, road, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid = False


# May be used to implement different ice objects (with different mechanics)
# NOTE: When defining a new Ice class, do not forget to add an entry to graphics.py > get_color()
class Ice(RoadObject):
    """Ice on road that leads to erratic behavior of car."""

    def __init__(
        self, road, position: Sequence[float]):
        super().__init__(road, position, heading = 0, speed = 0)
        self.solid = False

    @classmethod
    def create_random(
        cls,
        road: Road,
        lane_from: str | None = None,
        lane_to: str | None = None,
        lane_id: int | None = None,
        spacing: float = 1,
    ) -> Ice:
        """
        Create a random ice on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: An Ice object with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id or road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        default_spacing = 12
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(ic.position)[0] for ic in road.objects])
            if len(road.objects)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        ic = cls(road, lane.position(x0, 0))
        return ic



class Ice1(Ice):
    """Ice on road that leads to erratic behavior of car."""

    def __init__(
        self, road, position: Sequence[float]#, heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position)#, heading = 0, speed = 0)
        self.solid = False

    @classmethod
    def create_from_ice(cls, ice_object: Ice):
        return Ice1(road = ice_object.road, 
                    position = ice_object.position, 
                    heading = ice_object.heading, 
                    speed = ice_object.speed)