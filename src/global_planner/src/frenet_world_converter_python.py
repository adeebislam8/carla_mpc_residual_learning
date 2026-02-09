import numpy as np
from typing import Tuple, List
from scipy.interpolate import CubicSpline


class FrenetCartesianConverter:
    def __init__(self, waypoints: List[List[float]]):
        """
        Args:
            waypoints: List of [x, y] coordinates defining the reference path
        """
        if len(waypoints) < 4:
            raise ValueError(f"Need at least 4 waypoints, got {len(waypoints)}")
        
        waypoints = np.array(waypoints)
        x_coords = waypoints[:, 0]
        y_coords = waypoints[:, 1]
        
        x_coords, y_coords = self._remove_duplicate_points(x_coords, y_coords)
        
        if len(x_coords) < 4:
            raise ValueError(f"After removing duplicates, only {len(x_coords)} waypoints remain (need at least 4)")
        
        # Compute arc length
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.zeros(len(x_coords))
        s[1:] = np.cumsum(ds)
        
        if not np.all(np.diff(s) > 0):
            raise ValueError("Arc length s is not strictly increasing after removing duplicates")
        
        # Create splines
        self.x_spline = CubicSpline(s, x_coords)
        self.y_spline = CubicSpline(s, y_coords)
        self.s_max = s[-1]
        
        # Store for curvature calculation
        self._s_values = s
    
    @staticmethod
    def _remove_duplicate_points(x_coords: np.ndarray, y_coords: np.ndarray, 
                                 min_distance: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove consecutive waypoints that are too close together
        """
        if len(x_coords) == 0:
            return x_coords, y_coords
        
        # Always keep first point
        filtered_x = [x_coords[0]]
        filtered_y = [y_coords[0]]
        
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - filtered_x[-1]
            dy = y_coords[i] - filtered_y[-1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Only keep if far enough from previous kept point
            if distance >= min_distance:
                filtered_x.append(x_coords[i])
                filtered_y.append(y_coords[i])
        
        return np.array(filtered_x), np.array(filtered_y)
    
    def get_frenet(self, cartesian_state: List[float]) -> Tuple[float, float, float]:
        """
        Convert Cartesian (x, y, yaw) to Frenet (s, d, alpha)
        
        Args:
            cartesian_state: [x, y, yaw] in world coordinates
            
        Returns:
            (s, d, alpha) where:
                s: arc length along reference path
                d: lateral deviation from path
                alpha: heading error relative to path
        """
        x, y, yaw = cartesian_state
        
        # Find closest point on path
        s_guess = self._find_closest_s(x, y)
        
        # Refine with Newton's method
        s = self._refine_s(x, y, s_guess)
        s = np.clip(s, 0, self.s_max)
        
        # Get path point at s
        x_ref = self.x_spline(s)
        y_ref = self.y_spline(s)
        
        # Get path tangent
        dx_ds = self.x_spline.derivative()(s)
        dy_ds = self.y_spline.derivative()(s)
        yaw_ref = np.arctan2(dy_ds, dx_ds)
        
        # Calculate lateral deviation
        dx = x - x_ref
        dy = y - y_ref
        d = np.sqrt(dx**2 + dy**2)
        
        # Determine sign of d (left or right of path)
        cross = dx * dy_ds - dy * dx_ds
        d = - d if cross >= 0 else d
        
        # Heading error
        alpha = self._normalize_angle(yaw - yaw_ref)
        
        return s, d, alpha
    
    def get_cartesian(self, frenet_state: List[float]) -> Tuple[float, float, float]:
        """
        Convert Frenet (s, d, alpha) to Cartesian (x, y, yaw)
        
        Args:
            frenet_state: [s, d, alpha] in Frenet coordinates
            
        Returns:
            (x, y, yaw) in world coordinates
        """
        s, d, alpha = frenet_state
        s = np.clip(s, 0, self.s_max)
        
        # Get path point
        x_ref = self.x_spline(s)
        y_ref = self.y_spline(s)
        
        # Get path tangent
        dx_ds = self.x_spline.derivative()(s)
        dy_ds = self.y_spline.derivative()(s)
        yaw_ref = np.arctan2(dy_ds, dx_ds)
        
        # Calculate world position
        x = x_ref - d * np.sin(yaw_ref)
        y = y_ref + d * np.cos(yaw_ref)
        yaw = self._normalize_angle(yaw_ref + alpha)
        
        return x, y, yaw
    
    def get_curvature(self, s: float) -> float:
        """
        Get path curvature at arc length s
        
        Args:
            s: arc length
            
        Returns:
            kappa: curvature (1/radius)
        """
        s = np.clip(s, 0, self.s_max)
        
        # First derivatives
        dx_ds = self.x_spline.derivative()(s)
        dy_ds = self.y_spline.derivative()(s)
        
        # Second derivatives
        d2x_ds2 = self.x_spline.derivative(2)(s)
        d2y_ds2 = self.y_spline.derivative(2)(s)
        
        # Curvature formula
        numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
        denominator = (dx_ds**2 + dy_ds**2)**(3/2)
        
        kappa = numerator / (denominator + 1e-6)
        return kappa
    
    def get_path_length(self) -> float:
        """Get total path length"""
        return self.s_max
    
    def _find_closest_s(self, x: float, y: float) -> float:
        """Find initial guess for closest s using coarse search"""
        s_samples = np.linspace(0, self.s_max, 100)
        x_samples = self.x_spline(s_samples)
        y_samples = self.y_spline(s_samples)
        
        distances = (x_samples - x)**2 + (y_samples - y)**2
        idx = np.argmin(distances)
        
        return s_samples[idx]
    
    def _refine_s(self, x: float, y: float, s_init: float, max_iter: int = 10) -> float:
        """Refine s estimate using Newton's method"""
        s = s_init
        
        for _ in range(max_iter):
            # Path point and derivatives
            x_ref = self.x_spline(s)
            y_ref = self.y_spline(s)
            dx_ds = self.x_spline.derivative()(s)
            dy_ds = self.y_spline.derivative()(s)
            
            # Error vector
            ex = x_ref - x
            ey = y_ref - y
            
            # Newton step
            numerator = ex * dx_ds + ey * dy_ds
            denominator = dx_ds**2 + dy_ds**2 + 1e-6
            
            s_new = s - numerator / denominator
            s_new = np.clip(s_new, 0, self.s_max)
            
            # Check convergence
            if abs(s_new - s) < 1e-4:
                break
            
            s = s_new
        
        return s
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class FrenetConverter:
    """
    High-level wrapper for Frenet conversions
    Replaces ROS service interface with direct function calls
    """
    def __init__(self, waypoints: List[List[float]]):
        """
        Args:
            waypoints: List of [x, y] coordinates
        """
        self.converter = FrenetCartesianConverter(waypoints)
    
    def world_to_frenet(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        """
        Convert world coordinates to Frenet
        
        Replaces ROS service: /world2frenet
        
        Args:
            x, y: World position
            yaw: Heading in radians
            
        Returns:
            (s, d, alpha): Frenet coordinates
        """
        return self.converter.get_frenet([x, y, yaw])
    
    def frenet_to_world(self, s: float, d: float, alpha: float) -> Tuple[float, float, float]:
        """
        Args:
            s: Arc length
            d: Lateral deviation
            alpha: Heading error
            
        Returns:
            (x, y, yaw): World coordinates
        """
        return self.converter.get_cartesian([s, d, alpha])
    
    def get_curvature(self, s: float) -> float:
        """Get path curvature at arc length s"""
        return self.converter.get_curvature(s)
    
    def get_path_length(self) -> float:
        """Get total path length"""
        return self.converter.get_path_length()
    
    def sample_path(self, s_start: float, s_end: float, num_samples: int = 20) -> np.ndarray:
        """
        Sample path points in Frenet frame
        
        Args:
            s_start: Starting arc length
            s_end: Ending arc length
            num_samples: Number of samples
            
        Returns:
            Array of shape (num_samples, 3) with [s, x, y] for each sample
        """
        s_values = np.linspace(s_start, s_end, num_samples)
        path_samples = []
        
        for s in s_values:
            x, y, _ = self.frenet_to_world(s, 0, 0)
            path_samples.append([s, x, y])
        
        return np.array(path_samples)