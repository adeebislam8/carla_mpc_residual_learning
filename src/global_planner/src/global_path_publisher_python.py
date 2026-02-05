"""
PathPlanner: CARLA path planning without ROS

Replaces: global_path_publisher.py (ROS-based)
Features:
- Uses CARLA's GlobalRoutePlanner
- Returns waypoints as Python objects
- No ROS dependencies
"""
import os
import sys

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '../../..'))

# Add CARLA PythonAPI
sys.path.append(os.path.join(project_root, 'carla/PythonAPI/carla'))
sys.path.append(os.path.join(project_root, 'carla/PythonAPI'))

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from typing import List, Tuple


class PathPlanner:
    """
    Generate global paths using CARLA's navigation API
    """
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
        """
        Calculate route from start to goal
        
        Args:
            start: Start location
            goal: Goal location
            
        Returns:
            List of waypoints along the route
        """
        route = self.grp.trace_route(start, goal)
        
        if not route:
            return []
        
        # Extract waypoints from (waypoint, road_option) tuples
        waypoints = [wp_tuple[0] for wp_tuple in route]
        return waypoints
    
    def get_spawn_points(self) -> List[carla.Transform]:
        """Get all spawn points in the map"""
        return self.map.get_spawn_points()
    
    def get_road_width_at_waypoint(self, waypoint: carla.Waypoint) -> Tuple[float, float]:
        """
        Get road width at a waypoint
        
        Returns:
            (left_width, right_width) relative to waypoint
        """
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