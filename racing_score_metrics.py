import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

@dataclass
class RacingMetrics:
    """Stores all racing performance metrics for an episode"""
    # Core Performance
    lap_time: float = 0.0
    success: bool = False
    collision: bool = False
    
    # Speed Metrics
    avg_speed: float = 0.0
    max_speed: float = 0.0
    speed_consistency: float = 0.0  # Std dev of speed
    time_at_target_speed: float = 0.0  # % of time within ±10% of target
    
    # Path Following
    avg_lateral_error: float = 0.0  # Average |d|
    max_lateral_error: float = 0.0
    lateral_consistency: float = 0.0  # Std dev of d
    path_completion: float = 0.0  # % of path completed
    
    # Smoothness (Jerk/Aggressiveness)
    avg_throttle_change: float = 0.0  # Control smoothness
    avg_steering_change: float = 0.0
    max_throttle_change: float = 0.0
    max_steering_change: float = 0.0
    
    # Safety
    close_calls: int = 0  # Obstacles within 3m
    time_in_danger: float = 0.0  # Time spent too close to obstacles
    lane_violations: int = 0
    
    # Efficiency
    progress_per_step: float = 0.0  # s gained per timestep
    fuel_efficiency: float = 0.0  # Distance per throttle unit
    
    # Racing-specific
    overtakes: int = 0  # Successfully passed NPCs
    defensive_duration: float = 0.0  # Time defending position
    clean_sectors: int = 0  # Sectors without incidents

    # CARLA Standard Metrics (for comparability)
    route_completion: float = 0.0  # 0-1, % of route completed
    infractions_per_km: float = 0.0  # Normalized infraction count
    driving_score: float = 0.0  # CARLA leaderboard-style score
    
    # Breakdown of infractions (CARLA standard)
    collisions_layout: int = 0  # Collisions with static objects
    collisions_pedestrians: int = 0
    collisions_vehicles: int = 0
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    off_road_infractions: int = 0
    route_deviations: int = 0
    route_timeouts: int = 0


class RacingScoreCalculator:
    """Calculate comprehensive racing scores"""
    
    def __init__(self):
        # Score weights (total = 1000 points)
        self.weights = {
            'completion': 250,
            'speed': 200,
            'safety': 200,
            'smoothness': 150,
            'precision': 150,
            'efficiency': 50,
        }

    def calculate_carla_driving_score(self, metrics: RacingMetrics) -> float:
        """
        Official CARLA Leaderboard Driving Score
        DS = RC * penalty_factor
        where penalty_factor = Π (1 - β_i)^n_i
        """
        route_completion = metrics.route_completion
        
        # Penalty coefficients (from CARLA Leaderboard)
        penalties = {
            'collision_layout': 0.65,
            'collision_vehicle': 0.60,
            'collision_pedestrian': 0.50,
            'red_light': 0.70,
            'stop_sign': 0.80,
            'off_road': 0.60,
            'route_deviation': 0.30,
            'route_timeout': 0.70,
        }
        
        penalty_factor = 1.0
        penalty_factor *= (1 - penalties['collision_layout']) ** metrics.collisions_layout
        penalty_factor *= (1 - penalties['collision_vehicle']) ** metrics.collisions_vehicles
        penalty_factor *= (1 - penalties['collision_pedestrian']) ** metrics.collisions_pedestrians
        penalty_factor *= (1 - penalties['red_light']) ** metrics.red_light_violations
        penalty_factor *= (1 - penalties['stop_sign']) ** metrics.stop_sign_violations
        penalty_factor *= (1 - penalties['off_road']) ** metrics.off_road_infractions
        penalty_factor *= (1 - penalties['route_deviation']) ** metrics.route_deviations
        penalty_factor *= (1 - penalties['route_timeout']) ** metrics.route_timeouts
        
        driving_score = route_completion * penalty_factor
        return driving_score * 100  # Scale to 0-100
    
    def calculate_score(self, metrics: RacingMetrics, 
                       episode_length: int,
                       done_reason: str) -> Dict[str, float]:
        """
        Calculate comprehensive racing score (0-1000 points)
        
        Inspired by:
        - F1: Speed, consistency, tire management
        - NASCAR: Passing efficiency, defensive driving
        - iRacing: Safety rating, incident points
        """
        scores = {}
        
        # 1. COMPLETION SCORE (0-250 points)
        if done_reason == 'success':
            completion_score = 250.0
        elif done_reason == 'collision':
            completion_score = 0.0
        elif done_reason == 'timeout':
            # Partial credit based on progress
            completion_score = 100.0 * metrics.path_completion
        elif done_reason == 'stall':
            completion_score = 50.0 * metrics.path_completion
        else:
            completion_score = 0.0
        
        scores['completion'] = completion_score
        
        # 2. SPEED SCORE (0-200 points)
        # Based on average speed and consistency
        target_speed = 15  # m/s
        
        # Speed achievement (0-100)
        speed_ratio = min(metrics.avg_speed / target_speed, 1.2)
        speed_achievement = min(100.0 * speed_ratio, 100.0)
        
        # Speed consistency (0-100) - lower std is better
        if metrics.speed_consistency > 0:
            consistency_score = max(0, 100.0 - (metrics.speed_consistency * 10))
        else:
            consistency_score = 100.0
        
        scores['speed'] = speed_achievement + consistency_score
        
        # 3. SAFETY SCORE (0-200 points)
        # iRacing-inspired incident points system
        incident_points = 0
        incident_points += metrics.close_calls * 2  # 2 pts per close call
        incident_points += metrics.lane_violations * 4  # 4 pts per violation
        if metrics.collision:
            incident_points += 100  # Major penalty
        
        # Convert to score (fewer incidents = higher score)
        safety_score = max(0, 200.0 - incident_points)
        scores['safety'] = safety_score
        
        # 4. SMOOTHNESS SCORE (0-150 points)
        # F1-inspired control precision
        # Lower control changes = smoother = better
        
        throttle_smoothness = max(0, 75.0 - metrics.avg_throttle_change * 200)
        steering_smoothness = max(0, 75.0 - metrics.avg_steering_change * 200)
        
        scores['smoothness'] = throttle_smoothness + steering_smoothness
        
        # 5. PRECISION SCORE (0-150 points)
        # Path following accuracy
        
        # Lateral error (0-100)
        max_acceptable_error = 3.5  # meters
        avg_error_score = max(0, 100.0 * (1 - metrics.avg_lateral_error / max_acceptable_error))
        
        # Consistency (0-50)
        lateral_consistency_score = max(0, 50.0 - metrics.lateral_consistency * 20)
        
        scores['precision'] = avg_error_score + lateral_consistency_score
        
        # 6. EFFICIENCY SCORE (0-50 points)
        # Progress per step and fuel efficiency
        
        if episode_length > 0:
            progress_score = min(25.0, metrics.progress_per_step * 50)
        else:
            progress_score = 0.0
        
        fuel_score = min(25.0, metrics.fuel_efficiency * 2)
        
        scores['efficiency'] = progress_score + fuel_score
        
        # TOTAL SCORE
        scores['total'] = sum(scores.values())
        scores['grade'] = self._get_grade(scores['total'])
        
        return scores
    
    def _get_grade(self, total_score: float) -> str:
        """Convert score to letter grade"""
        if total_score >= 900: return 'S'   # Elite
        elif total_score >= 800: return 'A+' # Excellent
        elif total_score >= 700: return 'A'  # Very Good
        elif total_score >= 600: return 'B+' # Good
        elif total_score >= 500: return 'B'  # Above Average
        elif total_score >= 400: return 'C+' # Average
        elif total_score >= 300: return 'C'  # Below Average
        elif total_score >= 200: return 'D'  # Poor
        else: return 'F'  # Failed


class RacingMetricsTracker:
    """Track metrics during episode"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.speeds = []
        self.lateral_errors = []
        self.throttle_changes = []
        self.steering_changes = []
        self.s_values = []
        self.close_calls = 0
        self.lane_violations = 0
        self.prev_throttle = 0.0
        self.prev_steering = 0.0
        self.start_time = time.time()
        self.danger_time = 0.0
        self.total_throttle = 0.0
    
    def update(self, env_state: Dict):
        """Update metrics each step"""
        # Speed tracking
        self.speeds.append(env_state['speed'])
        
        # Lateral error tracking
        self.lateral_errors.append(abs(env_state['d']))
        
        # Control smoothness
        throttle_change = abs(env_state['throttle'] - self.prev_throttle)
        steering_change = abs(env_state['steering'] - self.prev_steering)
        self.throttle_changes.append(throttle_change)
        self.steering_changes.append(steering_change)
        
        self.prev_throttle = env_state['throttle']
        self.prev_steering = env_state['steering']
        
        # Progress tracking
        self.s_values.append(env_state['s'])
        
        # Fuel tracking
        self.total_throttle += abs(env_state['throttle'])
        
        # Safety tracking
        for obs in env_state['obstacles']:
            if obs[0] > -50:  # Valid obstacle
                dist = np.sqrt((env_state['s'] - obs[0])**2 + 
                             (env_state['d'] - obs[1])**2)
                if dist < 3.0:
                    self.close_calls += 1
                    self.danger_time += 0.05  # dt
        
        # Lane violations
        n_min = env_state.get('n_min', -5.25)  # fallback to Town01 default
        n_max = env_state.get('n_max', 1.75)

        if env_state['d'] < n_min or env_state['d'] > n_max:
            self.lane_violations += 1
    
    def finalize(self, path_length: float, done_reason: str) -> RacingMetrics:
        metrics = RacingMetrics()

        metrics.lap_time = time.time() - self.start_time
        metrics.success = (done_reason == 'success')
        metrics.collision = (done_reason == 'collision')
        
        if self.speeds:
            metrics.avg_speed = np.mean(self.speeds)
            metrics.max_speed = np.max(self.speeds)
            metrics.speed_consistency = np.std(self.speeds)
            
            # Calculate time at target speed
            # Target is 8.33 m/s, we accept ±10% (7.5 to 9.16 m/s)
            target_speed = 15
            tolerance = 0.10  # 10%
            lower_bound = target_speed * (1 - tolerance)  # 7.497
            upper_bound = target_speed * (1 + tolerance)  # 9.163
            
            # Count how many timesteps were in this range
            in_range = sum(1 for s in self.speeds if lower_bound <= s <= upper_bound)
            metrics.time_at_target_speed = in_range / len(self.speeds)
        
        if self.lateral_errors:
            metrics.avg_lateral_error = np.mean(self.lateral_errors)
            metrics.max_lateral_error = np.max(self.lateral_errors)
            metrics.lateral_consistency = np.std(self.lateral_errors)
        
        if self.s_values:
            metrics.path_completion = self.s_values[-1] / path_length
            
            # Also store in CARLA-standard field (same value)
            metrics.route_completion = metrics.path_completion
            
            metrics.progress_per_step = (self.s_values[-1] - self.s_values[0]) / len(self.s_values)
        
        if self.throttle_changes:
            metrics.avg_throttle_change = np.mean(self.throttle_changes)
            metrics.max_throttle_change = np.max(self.throttle_changes)
        
        if self.steering_changes:
            metrics.avg_steering_change = np.mean(self.steering_changes)
            metrics.max_steering_change = np.max(self.steering_changes)
        
        metrics.close_calls = self.close_calls
        metrics.time_in_danger = self.danger_time
        metrics.lane_violations = self.lane_violations
        
        if self.total_throttle > 0 and self.s_values:
            distance = self.s_values[-1] - self.s_values[0]
            metrics.fuel_efficiency = distance / self.total_throttle
        
        # 1. Collision infractions
        if metrics.collision:
            metrics.collisions_layout = 1

        if self.lane_violations > 20:  # More than 1 second off-road
            metrics.off_road_infractions = int(self.lane_violations / 20)
        
        target_speed = 15  # m/s
        expected_time = path_length / target_speed
        
        if metrics.lap_time > 2.0 * expected_time and not metrics.success:
            metrics.route_timeouts = 1
        
        # Now calculate the official CARLA Driving Score using our infractions
        from racing_score_metrics import RacingScoreCalculator  # Import at top of file
        calc = RacingScoreCalculator()
        metrics.driving_score = calc.calculate_carla_driving_score(metrics)
        
        total_infractions = (
            metrics.collisions_layout +
            metrics.collisions_vehicles +
            metrics.collisions_pedestrians +
            metrics.red_light_violations +
            metrics.stop_sign_violations +
            metrics.off_road_infractions +
            metrics.route_deviations +
            metrics.route_timeouts
        )
        
        # Distance traveled in kilometers
        if self.s_values:
            distance_km = (self.s_values[-1] - self.s_values[0]) / 1000.0
            
            if distance_km > 0:
                metrics.infractions_per_km = total_infractions / distance_km
            else:
                # Didn't move, but had infractions (e.g., immediate collision)
                metrics.infractions_per_km = float('inf') if total_infractions > 0 else 0.0
        
        return metrics