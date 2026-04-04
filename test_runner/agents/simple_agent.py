"""
Lightweight autonomous agent for CARLA simulation.
Implements basic perception, planning, and control.
"""

import carla
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Optional, Tuple
import cv2
from PIL import Image
import logging

from ..controllers import PIDController

logger = logging.getLogger(__name__)


class SimpleAutonomousAgent:
    """
    Minimal autonomous agent for CARLA v0.9.16.
    Handles sensor data, basic planning, and vehicle control.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize agent with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.vehicle = None
        self.sensors = []
        self.sensor_data = {}
        self.controller = PIDController()
        self.initialized = False
        
        # Agent state
        self.current_speed = 0.0
        self.current_yaw = 0.0
        self.waypoints_queue = []
        self.target_speed = 5.0  # m/s
        self.target_yaw = 0.0
        
        # VLM configuration (optional)
        self.vlm_enabled = False
        self.vlm_client = None
        self._setup_vlm()
        
        logger.info(f"Agent initialized with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_vlm(self):
        """Setup VLM client if configured in config file."""
        vlm_config = self.config.get('vlm', {})
        if not vlm_config.get('enabled', False):
            logger.debug("VLM disabled in config")
            return
        
        self.vlm_enabled = True
        vlm_type = vlm_config.get('type', 'openai')
        
        try:
            if vlm_type == 'openai':
                from openai import OpenAI
                api_key = vlm_config.get('api_key')
                api_base = vlm_config.get('api_base_url', 'https://api.openai.com/v1')
                self.vlm_client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info(f"VLM client initialized (OpenAI compatible): {api_base}")
            
            elif vlm_type == 'openrouter':
                from openai import OpenAI
                api_key = vlm_config.get('api_key')
                self.vlm_client = OpenAI(
                    api_key=api_key,
                    base_url='https://openrouter.ai/api/v1'
                )
                logger.info("VLM client initialized (OpenRouter)")
            
            elif vlm_type == 'anthropic':
                from anthropic import Anthropic
                api_key = vlm_config.get('api_key')
                self.vlm_client = Anthropic(api_key=api_key)
                logger.info("VLM client initialized (Anthropic)")
            
            else:
                logger.warning(f"Unknown VLM type: {vlm_type}")
                self.vlm_enabled = False
        
        except ImportError as e:
            logger.warning(f"VLM client libraries not installed: {e}. Disabling VLM.")
            self.vlm_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize VLM client: {e}")
            self.vlm_enabled = False
    
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
        camera_bp.set_attribute('image_size_x', str(self.config['camera']['width']))
        camera_bp.set_attribute('image_size_y', str(self.config['camera']['height']))
        camera_bp.set_attribute('fov', str(self.config['camera']['fov']))
        
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
        # Remove alpha channel
        image_rgb = image_data[:, :, :3]
        self.sensor_data['camera'] = image_rgb
    
    def _on_imu_data(self, imu_data: carla.IMUMeasurement):
        """Process IMU data."""
        self.sensor_data['velocity'] = imu_data.accelerometer
        self.current_speed = np.linalg.norm(self.vehicle.get_velocity())
        
        rotation = self.vehicle.get_transform().rotation
        self.current_yaw = np.radians(rotation.yaw)
    
    def set_target_speed(self, speed: float):
        """Set target driving speed (m/s)."""
        self.target_speed = max(0.0, min(speed, self.config['max_speed']))
    
    def set_target_waypoint(self, waypoint: carla.Waypoint):
        """Set target waypoint for navigation."""
        self.waypoints_queue.append(waypoint)
        target_transform = waypoint.transform
        self.target_yaw = np.radians(target_transform.rotation.yaw)
    
    def step(self) -> carla.VehicleControl:
        """
        Execute one step of agent perception, planning, and control.
        
        Returns:
            CARLA vehicle control command
        """
        if not self.initialized or self.vehicle is None:
            return carla.VehicleControl()
        
        # Get sensor data (perception)
        camera_image = self.sensor_data.get('camera')
        
        # Basic planning: maintain target speed and waypoint
        # (Can be extended with VLM inference)
        planned_speed = self._plan_speed(camera_image)
        planned_yaw = self._plan_steer()
        
        # Control
        throttle, brake = self.controller.compute_throttle_brake(
            self.current_speed, planned_speed
        )
        steer = self.controller.compute_steering(
            self.current_yaw, planned_yaw
        )
        
        # Create control command
        control = carla.VehicleControl()
        control.throttle = float(max(0.0, throttle))
        control.brake = float(max(0.0, brake))
        control.steer = float(steer)
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def _plan_speed(self, camera_image: Optional[np.ndarray]) -> float:
        """
        Plan target speed based on sensor data.
        Can use VLM inference for scene understanding if configured.
        
        Args:
            camera_image: Current camera image (BGR) or None
            
        Returns:
            Planned speed (m/s)
        """
        if self.vlm_enabled and camera_image is not None:
            try:
                return self._plan_speed_with_vlm(camera_image)
            except Exception as e:
                logger.warning(f"VLM inference failed: {e}. Using default speed.")
                return self.target_speed
        
        return self.target_speed
    
    def _plan_speed_with_vlm(self, camera_image: np.ndarray) -> float:
        """
        Use VLM to infer recommended speed from camera image.
        
        Args:
            camera_image: Camera image (BGR, uint8)
            
        Returns:
            Recommended speed (m/s)
        """
        import base64
        from PIL import Image
        import io
        
        # Convert numpy array to PNG bytes
        img_pil = Image.fromarray(cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        vlm_config = self.config.get('vlm', {})
        model = vlm_config.get('model', 'gpt-4-vision')
        vlm_type = vlm_config.get('type', 'openai')
        
        prompt = (
            "Analyze this driving scene and recommend a speed (0-25 m/s). "
            "Consider: traffic, pedestrians, road conditions, obstacles. "
            "Respond with ONLY a single number (e.g., 10.5)"
        )
        
        try:
            # Use appropriate API based on VLM type
            if vlm_type == 'anthropic':
                # Anthropic Claude API
                response = self.vlm_client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                speed_str = response.content[0].text.strip()
            else:
                # OpenAI-compatible API (OpenAI, OpenRouter, vLLM)
                response = self.vlm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=10,
                    temperature=0.3
                )
                speed_str = response.choices[0].message.content.strip()
            
            # Parse speed from response
            speed = float(speed_str)
            speed = max(0.0, min(25.0, speed))  # Clamp to valid range
            
            logger.debug(f"VLM recommended speed: {speed} m/s")
            return speed
        
        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            raise
    
    def _plan_steer(self) -> float:
        """
        Plan target steering angle based on waypoints.
        
        Returns:
            Target yaw angle (radians)
        """
        if self.waypoints_queue:
            return self.target_yaw
        return self.current_yaw
    
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
