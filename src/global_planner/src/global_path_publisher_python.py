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
        # Left-side allowance.  Measured on Town01 (tools/check_road_width.py):
        # 76.4% of driving waypoints have a real opposing lane and 23.6% do not --
        # and 100% of that 23.6% are JUNCTIONS, 0% open road.  So get_left_lane()
        # returning None never means "the road ends here"; it means the lane links
        # break inside an intersection.
        #
        # The old `else: += 4` therefore fired only at junctions, granting 4 m of
        # room out towards the corner furniture.  Because Town01 lanes are exactly
        # 4.00 m, that invented 4 m matched a real oncoming lane precisely, so the
        # corridor logged an identical n_min = -4.80 in both cases and the fault was
        # invisible from that number.  Corner fences, guardrails and poles are ~90%
        # of collisions.
        #
        # A junction still has open pavement and the route cuts a curve across it,
        # so the allowance must not go to zero -- but it should not reach the corner
        # either.  JUNCTION_MARGIN is the knob: raise it if turns become infeasible,
        # lower it if the car still clips corners.
        JUNCTION_MARGIN = 1.5
        SHOULDER = 0.3          # genuinely no lane: road edge, keep tight

        left_lane = waypoint.get_left_lane()
        if left_lane and left_lane.lane_type == carla.LaneType.Driving:
            left_width += left_lane.lane_width
        elif waypoint.is_junction:
            left_width += JUNCTION_MARGIN
        else:
            left_width += SHOULDER

        right_lane = waypoint.get_right_lane()
        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
            right_width += right_lane.lane_width
        
        return left_width, right_width