import os
import sys

sys.path.append('/home/ave/Desktop/carla_mpc_residual_learning/carla/PythonAPI/carla')

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
            left_width += left_lane.lane_width / 2.0
        
        right_lane = waypoint.get_right_lane()
        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
            right_width += right_lane.lane_width / 2.0
        
        return left_width, right_width