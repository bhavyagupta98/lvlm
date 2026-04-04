"""
PID Controller for LangCoop (CARLA 0.9.16 compatible)
Based on LangCoop's VLMControllerSpeedCurvature.
"""

import numpy as np
from collections import deque
from typing import Dict


class PIDController:
    """Window-based PID controller similar to upstream LangCoop control."""

    def __init__(self, kp: float, ki: float, kd: float, window_size: int = 20):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.window = deque(maxlen=window_size)

    def step(self, error: float) -> float:
        self.window.append(float(error))

        if len(self.window) >= 2:
            integral = float(np.mean(self.window))
            derivative = float(self.window[-1] - self.window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self.kp * error + self.ki * integral + self.kd * derivative

    def reset(self):
        self.window.clear()


class LangCoopController:
    """
    PID Controller for speed-curvature based control.
    Compatible with CARLA 0.9.16.
    """
    
    def __init__(self, **kwargs):
        """Initialize controller with PID gains."""
        turn_kp = kwargs.get('turn_kp', kwargs.get('steer_kp', 1.0))
        turn_ki = kwargs.get('turn_ki', 0.2)
        turn_kd = kwargs.get('turn_kd', 0.1)
        turn_n = kwargs.get('turn_n', 30)

        speed_kp = kwargs.get('speed_kp', 5.0)
        speed_ki = kwargs.get('speed_ki', 1.0)
        speed_kd = kwargs.get('speed_kd', 0.1)
        speed_n = kwargs.get('speed_n', 5)

        self.turn_controller = PIDController(turn_kp, turn_ki, turn_kd, window_size=turn_n)
        self.speed_controller = PIDController(speed_kp, speed_ki, speed_kd, window_size=speed_n)

        self.clip_delta = float(kwargs.get('clip_delta', 0.35))
        self.brake_ratio = float(kwargs.get('brake_ratio', 1.1))
        self.brake_speed = float(kwargs.get('brake_speed', 0.1))
        # CRITICAL: Curvature scaling matches upstream LangCoop
        # Upstream: curvature (degrees) / 10 → deg2rad(-90/10) → -0.157 rad
        # We do: curvature * scale, so scale = π/(180*10) ≈ 0.001745
        self.curvature_scale = float(kwargs.get('curvature_scale', 0.001745))

        self.max_throttle = float(kwargs.get('max_throttle', 0.75))
        self.max_brake = float(kwargs.get('max_brake', 1.0))
        self.max_steer = float(kwargs.get('max_steer', 1.0))
        
    def run_step(self, route_info: Dict, curr_speed: float, buffer_idx: int = 0) -> Dict:
        """
        Compute control commands from route information.
        
        Args:
            route_info: Dict with 'target_speed' and 'curvature'
            curr_speed: Current vehicle speed (m/s)
            buffer_idx: Index into prediction buffer (0 = current)
            
        Returns:
            Dict with 'throttle', 'brake', 'steer'
        """
        # Extract target speed and curvature
        target_speeds = route_info.get('target_speed', [8.0])
        curvatures = route_info.get('curvature', [0.0])
        
        if isinstance(target_speeds, list):
            target_speed = target_speeds[min(buffer_idx, len(target_speeds)-1)]
        else:
            target_speed = float(target_speeds)
        
        if isinstance(curvatures, list):
            curvature = curvatures[min(buffer_idx, len(curvatures)-1)]
        else:
            curvature = float(curvatures)
        
        steer = self.turn_controller.step(curvature * self.curvature_scale)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        speed_delta = float(np.clip(target_speed - curr_speed, 0.0, self.clip_delta))
        throttle = self.speed_controller.step(speed_delta)
        throttle = float(np.clip(throttle, 0.0, self.max_throttle))

        brake = 0.0
        if target_speed < self.brake_speed:
            brake = self.max_brake
            throttle = 0.0
        elif curr_speed > target_speed * self.brake_ratio:
            overspeed = curr_speed - target_speed
            brake = float(np.clip(overspeed / max(curr_speed, 1e-3), 0.0, self.max_brake))
            throttle = 0.0
        
        return {
            'throttle': float(throttle),
            'brake': float(brake),
            'steer': float(steer)
        }
    
    def reset(self):
        """Reset integral error."""
        self.turn_controller.reset()
        self.speed_controller.reset()
