"""
Multi-view LangCoop agent.

Extends the existing LangCoopAgent with additional RGB cameras so we can
capture front/left/right/rear/BEV views without changing the planning path.
The VLM planner still consumes the front camera via sensor_data["camera"].
"""

import carla
import numpy as np

from .langcoop_agent import LangCoopAgent


class MultiViewLangCoopAgent(LangCoopAgent):
    """LangCoop agent with a richer camera rig for logging and analysis."""

    def _setup_camera_sensor(self, blueprint_library, world):
        """Attach front/left/right/rear/BEV cameras."""
        camera_config = self.config.get('camera', {})
        width = int(camera_config.get('width', 800))
        height = int(camera_config.get('height', 600))
        fov = float(camera_config.get('fov', 100))

        rigs = [
            (
                'front_camera',
                carla.Transform(
                    location=carla.Location(x=1.3, y=0.0, z=2.3),
                    rotation=carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                ),
                width,
                height,
                fov,
            ),
            (
                'left_camera',
                carla.Transform(
                    location=carla.Location(x=0.5, y=0.0, z=2.2),
                    rotation=carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0),
                ),
                width,
                height,
                fov,
            ),
            (
                'right_camera',
                carla.Transform(
                    location=carla.Location(x=0.5, y=0.0, z=2.2),
                    rotation=carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
                ),
                width,
                height,
                fov,
            ),
            (
                'rear_camera',
                carla.Transform(
                    location=carla.Location(x=-1.6, y=0.0, z=2.2),
                    rotation=carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),
                ),
                width,
                height,
                fov,
            ),
            (
                'bev_camera',
                carla.Transform(
                    location=carla.Location(x=0.0, y=0.0, z=20.0),
                    rotation=carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
                ),
                640,
                640,
                90.0,
            ),
        ]

        for sensor_name, sensor_transform, sensor_width, sensor_height, sensor_fov in rigs:
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(sensor_width))
            camera_bp.set_attribute('image_size_y', str(sensor_height))
            camera_bp.set_attribute('fov', str(sensor_fov))
            camera = world.spawn_actor(camera_bp, sensor_transform, attach_to=self.vehicle)
            camera.listen(lambda image, key=sensor_name: self._on_named_camera_image(image, key))
            self.sensors.append(camera)

    def _on_named_camera_image(self, image: carla.Image, sensor_name: str):
        """Store a named RGB camera frame."""
        image_data = np.array(image.raw_data).reshape((image.height, image.width, 4))
        image_rgb = image_data[:, :, :3]
        self.sensor_data[sensor_name] = image_rgb
        self.sensor_data[f'{sensor_name}_timestamp'] = image.timestamp
        self.sensor_data[f'{sensor_name}_frame'] = image.frame

        # Preserve the base planner contract: the front camera remains under "camera".
        if sensor_name == 'front_camera':
            self.sensor_data['camera'] = image_rgb
            self.sensor_data['camera_timestamp'] = image.timestamp
            self.sensor_data['camera_frame'] = image.frame
