import os
import sys

# CARLA's PythonAPI ships the `agents` package alongside the `carla` module.
# Prefer whatever is already on PYTHONPATH; fall back to the in-repo checkout.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CARLA_API = os.path.join(_REPO_ROOT, 'carla', 'PythonAPI', 'carla')
if os.path.isdir(_CARLA_API) and _CARLA_API not in sys.path:
    sys.path.append(_CARLA_API)

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from typing import List, Tuple


class PathPlanner:
    def __init__(self, world: carla.World, carla_map: carla.Map, sampling_resolution: float = 1.0):
        """
        Args:
            world: CARLA world object
            carla_map: CARLA map object
            sampling_resolution: Waypoint sampling distance (meters)
        """
        self.world = world
        self.map = carla_map
        self.grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    
    def calculate_route(
        self, 
        start: carla.Location, 
        goal: carla.Location
    ) -> List[carla.Waypoint]:
        route = self.grp.trace_route(start, goal)
        
        if not route:
            return []
        
        # Extract waypoints from (waypoint, road_option) tuples
        waypoints = [wp_tuple[0] for wp_tuple in route]
        return waypoints
    
    def get_spawn_points(self) -> List[carla.Transform]:
        return self.map.get_spawn_points()
    
    def get_road_width_at_waypoint(self, waypoint: carla.Waypoint) -> Tuple[float, float]:
        left_width = waypoint.lane_width / 2.0
        right_width = waypoint.lane_width / 2.0
        
        # Check for adjacent lanes
        left_lane = waypoint.get_left_lane()
        if left_lane and left_lane.lane_type == carla.LaneType.Driving:
            left_width += left_lane.lane_width
        else:
            left_width += 4
        
        right_lane = waypoint.get_right_lane()
        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
            right_width += right_lane.lane_width
        
        return left_width, right_width