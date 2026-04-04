"""
Simple PID Controller for autonomous driving.
Converts high-level driving commands to low-level vehicle controls.
"""

import numpy as np
from collections import deque


class PIDController:
    """PID controller for vehicle throttle, brake, and steering."""
    
    def __init__(self, kp_throttle=0.5, ki_throttle=0.1, kd_throttle=0.1,
                 kp_steering=0.8, ki_steering=0.05, kd_steering=0.05,
                 window_size=20):
        """
        Args:
            kp_throttle: Proportional gain for throttle
            ki_throttle: Integral gain for throttle
            kd_throttle: Derivative gain for throttle
            kp_steering: Proportional gain for steering
            ki_steering: Integral gain for steering
            kd_steering: Derivative gain for steering
            window_size: Window size for computing integral/derivative
        """
        self.kp_throttle = kp_throttle
        self.ki_throttle = ki_throttle
        self.kd_throttle = kd_throttle
        
        self.kp_steering = kp_steering
        self.ki_steering = ki_steering
        self.kd_steering = kd_steering
        
        self.throttle_window = deque(maxlen=window_size)
        self.steering_window = deque(maxlen=window_size)
    
    def compute_throttle_brake(self, current_speed, target_speed, dt=0.05):
        """
        Compute throttle/brake command.
        
        Args:
            current_speed: Current vehicle speed (m/s)
            target_speed: Target speed (m/s)
            dt: Time step (seconds)
            
        Returns:
            throttle: [0, 1] or brake: [-1, 0] command
        """
        error = target_speed - current_speed
        self.throttle_window.append(error)
        
        p = self.kp_throttle * error
        i = self.ki_throttle * np.mean(self.throttle_window) if len(self.throttle_window) > 1 else 0
        d = self.kd_throttle * (error - (self.throttle_window[-2] if len(self.throttle_window) > 1 else error)) / dt if len(self.throttle_window) > 1 else 0
        
        command = np.clip(p + i + d, -1.0, 1.0)
        
        if command >= 0:
            return command, 0.0  # throttle, brake
        else:
            return 0.0, -command  # throttle, brake
    
    def compute_steering(self, current_yaw, target_yaw, dt=0.05):
        """
        Compute steering command.
        
        Args:
            current_yaw: Current vehicle yaw (radians)
            target_yaw: Target yaw (radians)
            dt: Time step (seconds)
            
        Returns:
            steer: [-1, 1] steering command
        """
        error = target_yaw - current_yaw
        # Normalize to [-pi, pi]
        error = np.arctan2(np.sin(error), np.cos(error))
        
        self.steering_window.append(error)
        
        p = self.kp_steering * error
        i = self.ki_steering * np.mean(self.steering_window) if len(self.steering_window) > 1 else 0
        d = self.kd_steering * (error - (self.steering_window[-2] if len(self.steering_window) > 1 else error)) / dt if len(self.steering_window) > 1 else 0
        
        steer = np.clip(p + i + d, -1.0, 1.0)
        return steer
    
    def reset(self):
        """Reset controller state."""
        self.throttle_window.clear()
        self.steering_window.clear()
