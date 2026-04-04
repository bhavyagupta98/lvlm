"""
LangCoop VLM Agent for CARLA 0.9.16 + Python 3.12
Self-contained implementation without old vlmdrive dependencies.
"""

import carla
import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, List

from ..vlm import VLMPlannerSpeedCurvature
from ..controllers import LangCoopController

logger = logging.getLogger(__name__)


class LangCoopAgent:
    """
    Full LangCoop VLM Agent for CARLA 0.9.16 + Python 3.12.
    
    Features:
    - VLMPlannerSpeedCurvature with Chain-of-Thought reasoning
    - Ego history tracking (speed, curvature, waypoints)
    - Speed-curvature prediction (5 timesteps, 2.5 seconds)
    - Compatible with local vLLM
    """
    
    def __init__(self, agent_config_path: str):
        """
        Initialize LangCoop agent.
        
        Args:
            agent_config_path: Path to agent configuration YAML
        """
        self.config = self._load_config(agent_config_path)
        
        self.vehicle = None
        self.sensors = []
        self.sensor_data = {}
        self.initialized = False
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Agent state
        self.current_speed = 0.0
        self.current_yaw = 0.0
        self.waypoints_queue = []
        self.target_speed = float(self.config.get('control', {}).get('target_speed', 12.0))
        self.target_yaw = 0.0
        self.last_planned_route = None
        self.waypoint_reach_threshold = float(
            self.config.get('planning', {}).get('waypoint_reach_threshold', 4.0)
        )
        
        # Perception memory bank (LangCoop architecture)
        self.perception_memory_bank = []
        self.max_history_frames = 10
        self.frame_count = 0
        self.agent_idx = 0
        
        # Initialize VLM Planner
        self._setup_vlm_planner()
        
        # Initialize Controller
        self._setup_controller()
        
        logger.info(f"LangCoop Agent initialized (CARLA 0.9.16 + Python 3.12)")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_vlm_planner(self):
        """Initialize VLM planner (VLMPlannerSpeedCurvature)."""
        try:
            vlm_config = self.config.get('vlm', {})
            
            if not vlm_config.get('enabled', False):
                logger.warning("VLM disabled in config")
                self.vlm_planner = None
                return
            
            self.vlm_planner = VLMPlannerSpeedCurvature(
                api_model_name=vlm_config.get('api_model_name', 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'),
                api_base_url=vlm_config.get('api_base_url', 'http://localhost:8000/v1'),
                api_key=vlm_config.get('api_key', 'EMPTY')
            )
            
            self.vlm_planner.to(self.device)
            self.vlm_planner.eval()
            
            logger.info("VLM Planner initialized: VLMPlannerSpeedCurvature")
            
        except Exception as e:
            logger.error(f"Failed to initialize VLM planner: {e}")
            self.vlm_planner = None
    
    def _setup_controller(self):
        """Initialize LangCoop controller."""
        control_config = self.config.get('control', {})
        self.controller = LangCoopController(
            speed_kp=control_config.get('speed_kp', 0.8),
            speed_ki=control_config.get('speed_ki', 0.1),
            speed_kd=control_config.get('speed_kd', 0.2),
            steer_kp=control_config.get('steer_kp', 1.0),
            turn_kp=control_config.get('turn_kp', control_config.get('steer_kp', 1.0)),
            turn_ki=control_config.get('turn_ki', 0.2),
            turn_kd=control_config.get('turn_kd', 0.1),
            turn_n=control_config.get('turn_n', 30),
            speed_n=control_config.get('speed_n', 5),
            clip_delta=control_config.get('clip_delta', 0.35),
            brake_ratio=control_config.get('brake_ratio', 1.1),
            brake_speed=control_config.get('brake_speed', 0.1),
            max_throttle=control_config.get('max_throttle', 0.75),
            max_brake=control_config.get('max_brake', 1.0),
            max_steer=control_config.get('max_steer', 1.0),
            curvature_scale=control_config.get('curvature_scale', 3.0)
        )
        logger.info("LangCoop Controller initialized")
    
    def setup(self, vehicle: carla.Actor):
        """Setup agent with vehicle."""
        self.vehicle = vehicle
        world = vehicle.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # Setup sensors
        self._setup_camera_sensor(blueprint_library, world)
        self._setup_imu_sensor(blueprint_library, world)
        
        self.initialized = True
        logger.info(f"Vehicle setup complete with {len(self.sensors)} sensors")
    
    def _setup_camera_sensor(self, blueprint_library, world):
        """Setup RGB camera."""
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_config = self.config.get('camera', {})
        camera_bp.set_attribute('image_size_x', str(camera_config.get('width', 800)))
        camera_bp.set_attribute('image_size_y', str(camera_config.get('height', 600)))
        camera_bp.set_attribute('fov', str(camera_config.get('fov', 100)))
        
        camera_transform = carla.Transform(
            location=carla.Location(x=1.3, y=0.0, z=2.3),
            rotation=carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda image: self._on_camera_image(image))
        self.sensors.append(camera)
        logger.debug("Camera sensor attached")
    
    def _setup_imu_sensor(self, blueprint_library, world):
        """Setup IMU sensor."""
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(location=carla.Location(x=0.0, y=0.0, z=0.0))
        
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        imu.listen(lambda imu_data: self._on_imu_data(imu_data))
        self.sensors.append(imu)
        logger.debug("IMU sensor attached")
    
    def _on_camera_image(self, image: carla.Image):
        """Process camera image."""
        image_data = np.array(image.raw_data).reshape((image.height, image.width, 4))
        image_rgb = image_data[:, :, :3]  # Remove alpha channel
        self.sensor_data['camera'] = image_rgb
        self.sensor_data['camera_timestamp'] = image.timestamp
        self.sensor_data['camera_frame'] = image.frame
    
    def _on_imu_data(self, imu_data: carla.IMUMeasurement):
        """Process IMU data."""
        velocity = self.vehicle.get_velocity()
        self.current_speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        
        rotation = self.vehicle.get_transform().rotation
        self.current_yaw = np.radians(rotation.yaw)
        
        self.sensor_data['accelerometer'] = imu_data.accelerometer
        self.sensor_data['imu_timestamp'] = imu_data.timestamp
    
    def set_target_speed(self, speed: float):
        """Set target speed."""
        max_speed = self.config.get('control', {}).get('max_speed', 20.0)
        self.target_speed = max(0.0, min(speed, max_speed))
    
    def set_target_waypoint(self, waypoint: carla.Waypoint):
        """Set target waypoint."""
        self.waypoints_queue.append(waypoint)
        target_transform = waypoint.transform
        self.target_yaw = np.radians(target_transform.rotation.yaw)
    
    def _update_perception_memory_bank(self):
        """Update perception memory bank with current frame (LangCoop architecture)."""
        if 'camera' not in self.sensor_data:
            return
        
        transform = self.vehicle.get_transform()
        location = transform.location
        
        # Build frame data
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
    
    def _get_target_waypoint(self) -> List[float]:
        """Get target waypoint relative to current position."""
        if self.waypoints_queue:
            current_transform = self.vehicle.get_transform()
            current_loc = current_transform.location

            # Drop waypoints that are already reached or are clearly behind ego.
            # This prevents stale behind-target prompts that can deadlock planning.
            while self.waypoints_queue:
                first_wp_loc = self.waypoints_queue[0].transform.location
                dx0 = first_wp_loc.x - current_loc.x
                dy0 = first_wp_loc.y - current_loc.y

                cos_yaw0 = np.cos(-self.current_yaw)
                sin_yaw0 = np.sin(-self.current_yaw)
                local_y0 = -cos_yaw0 * dx0 + sin_yaw0 * dy0

                reached = current_loc.distance(first_wp_loc) <= self.waypoint_reach_threshold
                behind_ego = local_y0 > 2.0
                if reached or behind_ego:
                    self.waypoints_queue.pop(0)
                    continue
                break

            if self.waypoints_queue:
                target_wp = self.waypoints_queue[0]
                target_loc = target_wp.transform.location

                dx = target_loc.x - current_loc.x
                dy = target_loc.y - current_loc.y

                cos_yaw = np.cos(-self.current_yaw)
                sin_yaw = np.sin(-self.current_yaw)

                local_x = sin_yaw * dx + cos_yaw * dy
                local_y = -cos_yaw * dx + sin_yaw * dy

                return [local_x, local_y]

        return [10.0, 0.0]  # Default: 10m forward
    
    def step(self) -> carla.VehicleControl:
        """
        Execute one agent step (LangCoop architecture).
        
        Returns:
            CARLA vehicle control command
        """
        if not self.initialized or self.vehicle is None:
            return carla.VehicleControl()
        
        # Update perception memory bank
        self._update_perception_memory_bank()
        
        # Plan with VLM (requires at least 2 frames)
        if self.vlm_planner and len(self.perception_memory_bank) >= 2:
            planned_route = self._plan_with_vlm()
        else:
            # Fallback
            planned_route = {
                'target_speed': [self.target_speed],
                'curvature': [0.0],
                'dt': 0.5
            }
        
        # Control
        self.last_planned_route = planned_route
        control = self._compute_control(planned_route)
        
        return control
    
    def _plan_with_vlm(self) -> Dict:
        """Plan using VLM planner (VLMPlannerSpeedCurvature)."""
        try:
            # Prepare model config with prompts
            model_config = {
                'planning': {
                    'prompt_template': self.config.get('planning', {}).get('prompt_template', {})
                }
            }
            
            # Call VLM planner
            with torch.no_grad():
                predicted_results = self.vlm_planner.forward(
                    self.perception_memory_bank,
                    model_config
                )
            
            # Extract result
            if isinstance(predicted_results, list):
                predicted_result = predicted_results[self.agent_idx]
            else:
                predicted_result = predicted_results
            
            logger.debug(f"VLM planned: speed={predicted_result['target_speed'][0]:.2f} m/s, "
                        f"curvature={predicted_result['curvature'][0]:.3f} rad/m")
            
            return predicted_result
            
        except Exception as e:
            logger.warning(f"VLM planning failed: {e}, using fallback")
            return {
                'target_speed': [self.target_speed],
                'curvature': [0.0],
                'dt': 0.5
            }
    
    def _compute_control(self, planned_route: Dict) -> carla.VehicleControl:
        """Compute vehicle control from planned route."""
        # Use LangCoop controller
        control_dict = self.controller.run_step(
            route_info=planned_route,
            curr_speed=self.current_speed,
            buffer_idx=0
        )
        
        control = carla.VehicleControl()
        control.throttle = float(max(0.0, min(1.0, control_dict['throttle'])))
        control.brake = float(max(0.0, min(1.0, control_dict['brake'])))
        control.steer = float(max(-1.0, min(1.0, control_dict['steer'])))
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def get_sensor_data(self) -> Dict:
        """Get current sensor readings."""
        return self.sensor_data.copy()
    
    def destroy(self):
        """Cleanup sensors."""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()
        logger.info("Agent sensors destroyed")
