"""
LangCoop VLM Agent - Full Integration
Follows the LangCoop architecture with VLMPlannerSpeedCurvature for planning.
"""

import carla
import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional
import sys

# Add vlmdrive to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlmdrive import VLMDRIVE_REGISTRY
from common.registry import build_object_within_registry_from_config
from common.io import load_config_from_yaml

logger = logging.getLogger(__name__)


class LangCoopVLMAgent:
    """
    Full LangCoop VLM Agent integrating:
    - VLMPlannerSpeedCurvature for vision-based planning
    - Chain-of-Thought (CoT) reasoning
    - Ego history tracking
    - PID controller for vehicle control
    """
    
    def __init__(self, agent_config_path: str, vlm_planner_config_path: str):
        """
        Initialize agent with LangCoop VLM architecture.
        
        Args:
            agent_config_path: Path to agent configuration YAML
            vlm_planner_config_path: Path to VLM planner configuration YAML
        """
        self.agent_config = self._load_config(agent_config_path)
        self.vlm_config = load_config_from_yaml(vlm_planner_config_path)
        
        self.vehicle = None
        self.sensors = []
        self.sensor_data = {}
        self.initialized = False
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Agent state tracking
        self.current_speed = 0.0
        self.current_yaw = 0.0
        self.waypoints_queue = []
        self.target_speed = 5.0
        self.target_yaw = 0.0
        
        # Perception memory bank for VLM planner (stores historical data)
        self.perception_memory_bank = []
        self.max_history_frames = 10
        self.frame_count = 0
        self.agent_idx = 0  # Index for this agent (0 for single-agent)
        
        # Initialize VLM Planner
        self._setup_vlm_planner()
        
        # Initialize Controller
        self._setup_controller()
        
        logger.info(f"LangCoopVLMAgent initialized with config: {agent_config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_vlm_planner(self):
        """Initialize VLM planner using VLMPlannerSpeedCurvature."""
        try:
            planning_model_config = self.vlm_config['model']
            self.vlm_planner = build_object_within_registry_from_config(
                VLMDRIVE_REGISTRY,
                planning_model_config
            )
            self.vlm_planner.to(self.device)
            self.vlm_planner.eval()
            logger.info(f"VLM Planner initialized: {planning_model_config['type']}")
        except Exception as e:
            logger.error(f"Failed to initialize VLM planner: {e}")
            raise
    
    def _setup_controller(self):
        """Initialize PID controller for vehicle control."""
        try:
            control_config_path = self.agent_config.get('control', {}).get('control_config')
            if control_config_path:
                control_config = load_config_from_yaml(control_config_path)
                control_model_config = control_config['model']
                self.controller = build_object_within_registry_from_config(
                    VLMDRIVE_REGISTRY,
                    control_model_config
                )
                logger.info("Controller initialized from config")
            else:
                # Fallback to simple PID controller
                from ..controllers import PIDController
                self.controller = PIDController()
                logger.info("Fallback PID controller initialized")
        except Exception as e:
            logger.warning(f"Failed to load controller from config: {e}, using fallback")
            from ..controllers import PIDController
            self.controller = PIDController()
    
    def setup(self, vehicle: carla.Actor):
        """
        Setup agent with assigned vehicle.
        
        Args:
            vehicle: CARLA vehicle actor
        """
        self.vehicle = vehicle
        world = vehicle.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # Setup sensors
        self._setup_camera_sensor(blueprint_library, world)
        self._setup_imu_sensor(blueprint_library, world)
        
        self.initialized = True
        logger.info(f"Vehicle setup complete with {len(self.sensors)} sensors")
    
    def _setup_camera_sensor(self, blueprint_library, world):
        """Setup RGB camera sensor."""
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.agent_config.get('camera', {}).get('width', 800)))
        camera_bp.set_attribute('image_size_y', str(self.agent_config.get('camera', {}).get('height', 600)))
        camera_bp.set_attribute('fov', str(self.agent_config.get('camera', {}).get('fov', 100)))
        
        camera_transform = carla.Transform(
            location=carla.Location(x=1.3, y=0.0, z=2.3),
            rotation=carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda image: self._on_camera_image(image))
        self.sensors.append(camera)
        logger.debug("RGB camera sensor attached")
    
    def _setup_imu_sensor(self, blueprint_library, world):
        """Setup IMU sensor for velocity and acceleration."""
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(location=carla.Location(x=0.0, y=0.0, z=0.0))
        
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        imu.listen(lambda imu_data: self._on_imu_data(imu_data))
        self.sensors.append(imu)
        logger.debug("IMU sensor attached")
    
    def _on_camera_image(self, image: carla.Image):
        """Process camera image from CARLA."""
        # Convert to numpy array
        image_data = np.array(image.raw_data).reshape((image.height, image.width, 4))
        # Remove alpha channel and convert BGR -> RGB
        image_rgb = image_data[:, :, :3]
        self.sensor_data['camera'] = image_rgb
        self.sensor_data['camera_timestamp'] = image.timestamp
    
    def _on_imu_data(self, imu_data: carla.IMUMeasurement):
        """Process IMU data."""
        self.sensor_data['accelerometer'] = imu_data.accelerometer
        self.current_speed = np.linalg.norm([
            self.vehicle.get_velocity().x,
            self.vehicle.get_velocity().y,
            self.vehicle.get_velocity().z
        ])
        
        rotation = self.vehicle.get_transform().rotation
        self.current_yaw = np.radians(rotation.yaw)
        self.sensor_data['imu_timestamp'] = imu_data.timestamp
    
    def set_target_speed(self, speed: float):
        """Set target driving speed (m/s)."""
        max_speed = self.agent_config.get('control', {}).get('max_speed', 20.0)
        self.target_speed = max(0.0, min(speed, max_speed))
    
    def set_target_waypoint(self, waypoint: carla.Waypoint):
        """Set target waypoint for navigation."""
        self.waypoints_queue.append(waypoint)
        target_transform = waypoint.transform
        self.target_yaw = np.radians(target_transform.rotation.yaw)
    
    def _update_perception_memory_bank(self):
        """
        Update perception memory bank with current frame data.
        This follows LangCoop's structure for storing historical data.
        """
        if 'camera' not in self.sensor_data:
            return
        
        transform = self.vehicle.get_transform()
        location = transform.location
        
        # Build current frame data
        frame_data = {
            'timestamp': self.sensor_data.get('camera_timestamp', self.frame_count * 0.5),
            'detmap_pose': [torch.tensor([location.x, location.y, self.current_yaw], device=self.device)],
            'ego_yaw': [self.current_yaw],
            'front_image': self.sensor_data['camera'],
            'target': [self._get_target_waypoint()],
            'ego_speed': [self.current_speed]
        }
        
        self.perception_memory_bank.append(frame_data)
        
        # Keep only recent history
        if len(self.perception_memory_bank) > self.max_history_frames:
            self.perception_memory_bank.pop(0)
        
        self.frame_count += 1
    
    def _get_target_waypoint(self):
        """Get target waypoint relative to current position."""
        if self.waypoints_queue:
            target_wp = self.waypoints_queue[0]
            target_loc = target_wp.transform.location
            current_loc = self.vehicle.get_transform().location
            
            # Return relative position
            return [
                target_loc.x - current_loc.x,
                target_loc.y - current_loc.y
            ]
        return [10.0, 0.0]  # Default: 10m forward
    
    def step(self) -> carla.VehicleControl:
        """
        Execute one step of agent perception, planning, and control.
        Uses VLM planner for decision-making.
        
        Returns:
            CARLA vehicle control command
        """
        if not self.initialized or self.vehicle is None:
            return carla.VehicleControl()
        
        # Update perception memory bank
        self._update_perception_memory_bank()
        
        # Get planning from VLM (requires at least 2 frames for history)
        if len(self.perception_memory_bank) >= 2:
            planned_route = self._plan_with_vlm()
        else:
            # Fallback to simple planning
            planned_route = {
                'target_speed': [self.target_speed],
                'curvature': [0.0],
                'dt': 0.5
            }
        
        # Control
        control = self._compute_control(planned_route)
        
        return control
    
    def _plan_with_vlm(self) -> Dict:
        """
        Use VLM planner for decision-making following LangCoop architecture.
        
        Returns:
            Planned route with speed and curvature
        """
        try:
            # Prepare model config for VLM planner
            model_config = {
                'planning': {
                    'prompt_usage': self.agent_config.get('planning', {}).get('prompt_usage', {}),
                    'prompt_template': self.agent_config.get('planning', {}).get('prompt_template', {})
                },
                'collab': {
                    'sharing_modalities': []  # Single agent, no collaboration
                }
            }
            
            # Call VLM planner's forward method
            with torch.no_grad():
                predicted_results = self.vlm_planner.forward(
                    self.perception_memory_bank,
                    model_config
                )
            
            # Extract result for this agent
            if isinstance(predicted_results, list):
                predicted_result = predicted_results[self.agent_idx]
            else:
                predicted_result = predicted_results
            
            logger.debug(f"VLM planned: speed={predicted_result.get('target_speed', [0])[0]:.2f} m/s")
            return predicted_result
            
        except Exception as e:
            logger.warning(f"VLM planning failed: {e}, using fallback")
            return {
                'target_speed': [self.target_speed],
                'curvature': [0.0],
                'dt': 0.5
            }
    
    def _compute_control(self, planned_route: Dict) -> carla.VehicleControl:
        """
        Compute vehicle control from planned route.
        
        Args:
            planned_route: Dictionary with target_speed and curvature
            
        Returns:
            CARLA vehicle control command
        """
        # Extract target speed and curvature
        target_speed = planned_route['target_speed'][0] if isinstance(planned_route['target_speed'], list) else planned_route['target_speed']
        curvature = planned_route.get('curvature', [0.0])[0] if 'curvature' in planned_route else 0.0
        
        # Use controller to compute throttle/brake/steer
        if hasattr(self.controller, 'run_step'):
            # VLMController from LangCoop
            route_info = {
                'target_speed': target_speed,
                'curvature': curvature
            }
            control_dict = self.controller.run_step(
                route_info=route_info,
                curr_speed=self.current_speed
            )
            
            control = carla.VehicleControl()
            control.throttle = float(max(0.0, min(1.0, control_dict.get('throttle', 0.0))))
            control.brake = float(max(0.0, min(1.0, control_dict.get('brake', 0.0))))
            control.steer = float(max(-1.0, min(1.0, control_dict.get('steer', 0.0))))
        else:
            # Fallback PID controller
            throttle, brake = self.controller.compute_throttle_brake(
                self.current_speed, target_speed
            )
            steer = self.controller.compute_steering(
                self.current_yaw, self.target_yaw
            )
            
            control = carla.VehicleControl()
            control.throttle = float(max(0.0, throttle))
            control.brake = float(max(0.0, brake))
            control.steer = float(steer)
        
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def get_sensor_data(self) -> Dict:
        """Get current sensor readings."""
        return self.sensor_data.copy()
    
    def destroy(self):
        """Cleanup and destroy sensors."""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()
        logger.info("Agent sensors destroyed")
