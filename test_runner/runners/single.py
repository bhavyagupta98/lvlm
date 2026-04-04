#!/usr/bin/env python3
"""
LangCoop Test Runner - Full Integration
Complete independent module following LangCoop architecture.

Features:
- Connects to CARLA at carla-rpc
- Uses VLMPlannerSpeedCurvature with local vLLM
- Chain-of-Thought (CoT) reasoning
- Full metrics calculation (RC%, DS, infractions)
- Visualization and reporting
"""

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
import carla
import time
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from test_runner.agents import LangCoopAgent
from test_runner.evaluator.metrics import MetricsCalculator
from test_runner.evaluator.scenario_manager import ScenarioManager, Scenario, Route
from test_runner.visualization import MetricsVisualizer, LiveMetricsTracer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LangCoopTestRunner:
    """
    Full independent test module following LangCoop architecture.
    """
    
    def __init__(
        self,
        carla_host: str = 'carla-rpc',
        carla_port: int = 2000,
        agent_config: str = 'configs/langcoop_agent_config.yaml',
        results_dir: str = 'test_results_langcoop'
    ):
        """
        Initialize LangCoop test runner.
        
        Args:
            carla_host: CARLA server hostname (e.g., 'carla-rpc')
            carla_port: CARLA server port
            agent_config: Path to agent configuration
            results_dir: Directory for results
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.agent_config = agent_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for images
        self.images_dir = self.results_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.world = None
        self.agent = None
        self.vehicle = None
        
        self.metrics_calculator = MetricsCalculator()
        # ScenarioManager will use test_runner/scenarios/ by default
        self.scenario_manager = ScenarioManager()
        self.visualizer = MetricsVisualizer()
        
        logger.info("LangCoop Test Runner initialized")
    
    def connect_to_carla(self):
        """Connect to CARLA server."""
        try:
            logger.info(f"Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()
            logger.info(f"✓ Connected to CARLA successfully")
            
            # Log server info
            server_version = self.client.get_server_version()
            logger.info(f"CARLA Server version: {server_version}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to CARLA: {e}")
            logger.error(f"Make sure CARLA server is running on {self.carla_host}:{self.carla_port}")
            return False
    
    def load_map(self, map_name: str):
        """Load CARLA map."""
        try:
            logger.info(f"Loading map: {map_name}")
            self.world = self.client.load_world(map_name)
            time.sleep(2)  # Wait for map to load
            logger.info(f"✓ Map loaded: {map_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load map {map_name}: {e}")
            raise
    
    def setup_environment(self, scenario: Scenario):
        """
        Setup simulation environment (weather, traffic, etc).
        
        Args:
            scenario: Scenario object with environmental parameters
        """
        logger.info(f"Setting up scenario: {scenario.scenario_id}")
        
        # Set weather
        weather = carla.WeatherParameters()
        if scenario.weather:
            weather.cloudiness = scenario.weather.get('cloudiness', 30)
            weather.precipitation = scenario.weather.get('precipitation', 0)
            weather.wind_intensity = scenario.weather.get('wind_intensity', 0)
            weather.sun_altitude_angle = scenario.weather.get('sun_altitude_angle', 45)
            weather.fog_density = scenario.weather.get('fog_density', 0)
        
        self.world.set_weather(weather)
        
        # Set simulation settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)
        
        logger.info(f"✓ Environment setup complete")
    
    def spawn_vehicle(self, spawn_point: carla.Transform):
        """
        Spawn ego vehicle at specified point.
        
        Args:
            spawn_point: CARLA Transform for spawn location
            
        Returns:
            Vehicle actor
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # 1) Try requested spawn point with small z-offset retries.
        z_offsets = [0.0, 0.3, 0.6, 1.0]
        for z_offset in z_offsets:
            candidate = carla.Transform(
                location=carla.Location(
                    x=spawn_point.location.x,
                    y=spawn_point.location.y,
                    z=spawn_point.location.z + z_offset,
                ),
                rotation=spawn_point.rotation,
            )
            vehicle = self.world.try_spawn_actor(vehicle_bp, candidate)
            if vehicle is not None:
                logger.info(
                    f"✓ Vehicle spawned at ({candidate.location.x:.1f}, {candidate.location.y:.1f}, {candidate.location.z:.1f})"
                )
                return vehicle

        # 2) Fallback to map spawn points if requested point is occupied.
        fallback_points = self.world.get_map().get_spawn_points()
        for fallback in fallback_points:
            vehicle = self.world.try_spawn_actor(vehicle_bp, fallback)
            if vehicle is not None:
                logger.warning(
                    "Requested spawn point occupied; spawned at fallback "
                    f"({fallback.location.x:.1f}, {fallback.location.y:.1f}, {fallback.location.z:.1f})"
                )
                return vehicle

        raise RuntimeError("Failed to spawn vehicle: all candidate spawn points are occupied")
    
    def setup_agent(self):
        """Setup LangCoop VLM agent."""
        try:
            logger.info("Initializing LangCoop VLM Agent...")
            self.agent = LangCoopAgent(agent_config_path=self.agent_config)
            self.agent.setup(self.vehicle)
            logger.info("✓ LangCoop VLM Agent initialized with:")
            logger.info(f"  - VLM Planner: VLMPlannerSpeedCurvature")
            logger.info(f"  - Chain-of-Thought (CoT) prompting")
            logger.info(f"  - Local vLLM endpoint")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to setup agent: {e}")
            logger.error("Make sure vLLM server is running and accessible")
            return False
    
    def run_scenario(
        self,
        scenario: Scenario,
        route: Route,
        max_steps: int = 1000,
        skip_frames: int = 4
    ):
        """
        Run test scenario with LangCoop agent.
        
        Args:
            scenario: Scenario configuration
            route: Route to follow
            max_steps: Maximum simulation steps
            skip_frames: Frame skip for control (LangCoop uses 4)
            
        Returns:
            Metrics dictionary
        """
        logger.info(f"Starting scenario: {scenario.scenario_id}")
        route_distance = route.compute_distance()
        route_metric_id = f"{scenario.scenario_id}_agent_0"
        logger.info(f"Route: {len(route.waypoints)} waypoints, {route_distance:.1f}m")

        # Reset this route's metrics explicitly so live traces start from 0.
        self.metrics_calculator.set_route_completion(route_metric_id, 0.0, route_distance, 0.0)
        self.metrics_calculator.set_speed_metrics(route_metric_id, 0.0, 0.0)
        
        # Setup environment
        self.setup_environment(scenario)
        
        # Spawn vehicle at first waypoint
        spawn_point = route.waypoints[0].to_transform()
        self.vehicle = self.spawn_vehicle(spawn_point)
        
        # Setup agent
        if not self.setup_agent():
            return None
        
        # Set waypoints for agent
        world_map = self.world.get_map()
        for wp in route.waypoints[1:]:  # Skip first (spawn point)
            wp_location = carla.Location(
                x=wp.location['x'],
                y=wp.location['y'],
                z=wp.location['z']
            )
            waypoint = world_map.get_waypoint(wp_location)
            if waypoint:
                self.agent.set_target_waypoint(waypoint)
        
        # Simulation loop
        logger.info("Starting simulation loop...")
        start_time = time.time()
        completed_distance = 0.0
        prev_location = self.vehicle.get_location()
        speed_samples = []
        last_control = carla.VehicleControl()
        last_target_speed = 0.0
        last_planner_debug = {}
        hazard_debug_stats = {
            'sampled_frames': 0,
            'sampled_static_hazard_frames': 0,
            'sampled_proximity_frames': 0,
            'sampled_guard_applied_frames': 0,
            'sampled_wall_barrier_mentions': 0,
            'saved_debug_frames': 0,
        }
        first_distance_sample = True
        live_trace_interval = 10
        live_tracer = LiveMetricsTracer(
            output_dir=str(self.results_dir),
            scenario_id=scenario.scenario_id,
            route_id=route_metric_id
        )
        
        collision_sensor = self._setup_collision_sensor(route_metric_id)
        last_live_snapshot = None
        
        aborted_reason = None
        for step in range(max_steps):
            try:
                self.world.tick()
            except RuntimeError as e:
                aborted_reason = str(e)
                logger.error(f"Simulator tick failed at step {step}: {aborted_reason}")
                self.metrics_calculator.record_event(
                    route_metric_id,
                    step,
                    'timeout',
                    description=aborted_reason
                )
                break
            
            # Agent step (follows LangCoop skip_frames=4 pattern)
            if step % skip_frames == 0:
                try:
                    control = self.agent.step()
                    self.vehicle.apply_control(control)
                    last_control = control

                    if hasattr(self.agent, 'last_planned_route') and isinstance(self.agent.last_planned_route, dict):
                        target_speed_arr = self.agent.last_planned_route.get('target_speed', [0.0])
                        if isinstance(target_speed_arr, list) and target_speed_arr:
                            last_target_speed = float(target_speed_arr[0])
                        else:
                            last_target_speed = float(target_speed_arr)
                        last_planner_debug = self._extract_planner_debug(self.agent.last_planned_route)

                    velocity_after_control = self.vehicle.get_velocity()
                    speed_after_control = np.linalg.norm([
                        velocity_after_control.x,
                        velocity_after_control.y,
                        velocity_after_control.z
                    ])
                    logger.debug(
                        "Applied control | step=%d throttle=%.3f brake=%.3f steer=%.3f speed=%.3f",
                        step,
                        float(control.throttle),
                        float(control.brake),
                        float(control.steer),
                        float(speed_after_control)
                    )
                except Exception as e:
                    logger.error(f"Agent step failed at step {step}: {e}")
                    self.metrics_calculator.record_event(
                        route_metric_id,
                        step,
                        'timeout',
                        description=str(e)
                    )
                    break
            
            # Update metrics
            current_location = self.vehicle.get_location()
            if first_distance_sample:
                # Ignore the first sample to avoid spawn/tick jitter inflating RC at t=0.
                distance_delta = 0.0
                first_distance_sample = False
            else:
                distance_delta = current_location.distance(prev_location)
            completed_distance += distance_delta
            prev_location = current_location

            # Check infractions
            velocity = self.vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
            speed_samples.append(speed)
            
            if speed > 20.5:  # Speed limit from LangCoop
                self.metrics_calculator.record_event(
                    route_metric_id,
                    step,
                    'speed_violation',
                    description=f"speed={speed:.2f}"
                )

            if step % live_trace_interval == 0:
                last_live_snapshot = self._build_live_metrics_snapshot(
                    route_metric_id=route_metric_id,
                    step=step,
                    max_steps=max_steps,
                    completed_distance=completed_distance,
                    route_distance=route_distance,
                    start_time=start_time,
                    speed_samples=speed_samples,
                    current_speed=speed,
                    current_steer=float(last_control.steer),
                    target_speed=last_target_speed,
                    planner_debug=last_planner_debug,
                    hazard_debug_stats=hazard_debug_stats
                )

                hazard_debug_stats['sampled_frames'] += 1
                if last_live_snapshot.get('static_hazard_detected', False):
                    hazard_debug_stats['sampled_static_hazard_frames'] += 1
                if last_live_snapshot.get('hazard_proximity_detected', False):
                    hazard_debug_stats['sampled_proximity_frames'] += 1
                if last_live_snapshot.get('static_guard_applied', False):
                    hazard_debug_stats['sampled_guard_applied_frames'] += 1
                if last_live_snapshot.get('wall_barrier_mentioned', False):
                    hazard_debug_stats['sampled_wall_barrier_mentions'] += 1

                live_tracer.update(last_live_snapshot)

                logger.info(
                    "Live Trace | step=%d/%d sim=%.1f%% route=%.1f%% DS=%.1f speed=%.1f m/s collisions=%d violations=%d",
                    step,
                    max_steps,
                    last_live_snapshot['simulation_progress_pct'],
                    last_live_snapshot['rs'],
                    last_live_snapshot['ds'],
                    last_live_snapshot['speed_mps'],
                    last_live_snapshot['collisions'],
                    last_live_snapshot['violations']
                )

            # Save camera image every 10 steps (every ~0.5s at 20 Hz)
            if step % 10 == 0 and 'camera' in self.agent.sensor_data:
                self._save_camera_image(
                    self.agent.sensor_data['camera'],
                    scenario.scenario_id,
                    step,
                    metrics_snapshot=last_live_snapshot
                )
                hazard_debug_stats['saved_debug_frames'] += 1
            
            # Progress logging
            if step % 100 == 0:
                progress = (completed_distance / route_distance) * 100 if route_distance > 0 else 0.0
                logger.info(f"Step {step}/{max_steps} | Progress: {progress:.1f}% | Speed: {speed:.1f} m/s")
        
        # Cleanup
        if collision_sensor:
            try:
                collision_sensor.stop()
            except Exception:
                pass
            try:
                collision_sensor.destroy()
            except Exception:
                pass
        if self.agent:
            self.agent.destroy()
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        # Calculate final metrics
        elapsed_time = time.time() - start_time
        self.metrics_calculator.set_route_completion(
            route_metric_id,
            completed_distance,
            route_distance,
            elapsed_time
        )
        if speed_samples:
            self.metrics_calculator.set_speed_metrics(
                route_metric_id,
                float(np.mean(speed_samples)),
                float(np.max(speed_samples))
            )
        final_ds = self.metrics_calculator.calculate_driving_score(route_metric_id)
        final_route_metrics = self.metrics_calculator.get_route_metrics(route_metric_id)
        final_violations = final_route_metrics.lane_departures + final_route_metrics.speed_violations
        final_speed = speed_samples[-1] if speed_samples else 0.0

        final_snapshot = {
            'step': int(max_steps),
            'simulation_progress_pct': 100.0,
            'elapsed_wall_time_sec': float(elapsed_time),
            'distance_m': float(completed_distance),
            'route_distance_m': float(route_distance),
            'rs': float(final_route_metrics.completion_percentage),
            'rc': float(final_route_metrics.completion_percentage),
            'ds': float(final_ds),
            'speed_mps': float(final_speed),
            'target_speed_mps': float(last_target_speed),
            'steer': float(last_control.steer),
            'steer_deg': float(last_control.steer * 70.0),
            'collisions': int(final_route_metrics.collisions),
            'violations': int(final_violations),
            'route_id': route_metric_id,
            'static_hazard_detected': bool(last_planner_debug.get('hazard_signals', {}).get('has_static_hazard', False)),
            'hazard_proximity_detected': bool(last_planner_debug.get('hazard_signals', {}).get('has_proximity_cue', False)),
            'static_guard_applied': bool(last_planner_debug.get('static_guard_applied', False)),
            'wall_barrier_mentioned': bool(last_planner_debug.get('wall_barrier_mentioned', False)),
            'hazard_side': str(last_planner_debug.get('hazard_signals', {}).get('hazard_side', 'none')),
            'static_guard_reason': str(last_planner_debug.get('static_guard_reason', '')),
            'planner_scene_excerpt': str(last_planner_debug.get('scene_excerpt', '')),
            'planner_objects_excerpt': str(last_planner_debug.get('objects_excerpt', '')),
            'planner_intent_excerpt': str(last_planner_debug.get('intent_excerpt', '')),
            'hazard_debug_stats': dict(hazard_debug_stats),
        }
        live_tracer.update(final_snapshot)
        self._write_scenario_debug_summary(
            scenario_id=scenario.scenario_id,
            final_snapshot=final_snapshot,
            hazard_debug_stats=hazard_debug_stats
        )

        metrics = self.metrics_calculator.get_summary()
        route_metrics = metrics.get('routes', {}).get(route_metric_id, {})
        
        logger.info(f"Scenario complete:")
        logger.info(f"  Route Completion: {route_metrics.get('rc', 0.0):.1f}%")
        logger.info(f"  Driving Score: {route_metrics.get('ds', 0.0):.1f}/100")
        logger.info(f"  Collisions: {metrics['total_collisions']}")
        logger.info(f"  Total Violations: {metrics['total_violations']}")
        if aborted_reason:
            logger.warning(f"  Scenario aborted early due to simulator timeout: {aborted_reason}")
        
        return metrics

    def _build_live_metrics_snapshot(
        self,
        route_metric_id: str,
        step: int,
        max_steps: int,
        completed_distance: float,
        route_distance: float,
        start_time: float,
        speed_samples: list,
        current_speed: float,
        current_steer: float,
        target_speed: float,
        planner_debug: dict,
        hazard_debug_stats: dict
    ) -> dict:
        """Compute a consistent live metrics snapshot for logging, tracing, and frame overlays."""
        elapsed_time = time.time() - start_time
        self.metrics_calculator.set_route_completion(
            route_metric_id,
            completed_distance,
            route_distance,
            elapsed_time
        )
        if speed_samples:
            self.metrics_calculator.set_speed_metrics(
                route_metric_id,
                float(np.mean(speed_samples)),
                float(np.max(speed_samples))
            )

        live_ds = self.metrics_calculator.calculate_driving_score(route_metric_id)
        live_route_metrics = self.metrics_calculator.get_route_metrics(route_metric_id)
        live_violations = live_route_metrics.lane_departures + live_route_metrics.speed_violations
        progress_pct = ((step + 1) / max_steps) * 100 if max_steps > 0 else 0.0

        return {
            'step': int(step),
            'simulation_progress_pct': float(progress_pct),
            'elapsed_wall_time_sec': float(elapsed_time),
            'distance_m': float(completed_distance),
            'route_distance_m': float(route_distance),
            'rs': float(live_route_metrics.completion_percentage),
            'rc': float(live_route_metrics.completion_percentage),
            'ds': float(live_ds),
            'speed_mps': float(current_speed),
            'target_speed_mps': float(target_speed),
            'steer': float(current_steer),
            'steer_deg': float(current_steer * 70.0),
            'collisions': int(live_route_metrics.collisions),
            'violations': int(live_violations),
            'route_id': route_metric_id,
            'static_hazard_detected': bool(planner_debug.get('hazard_signals', {}).get('has_static_hazard', False)),
            'hazard_proximity_detected': bool(planner_debug.get('hazard_signals', {}).get('has_proximity_cue', False)),
            'static_guard_applied': bool(planner_debug.get('static_guard_applied', False)),
            'wall_barrier_mentioned': bool(planner_debug.get('wall_barrier_mentioned', False)),
            'hazard_side': str(planner_debug.get('hazard_signals', {}).get('hazard_side', 'none')),
            'static_guard_reason': str(planner_debug.get('static_guard_reason', '')),
            'planner_scene_excerpt': str(planner_debug.get('scene_excerpt', '')),
            'planner_objects_excerpt': str(planner_debug.get('objects_excerpt', '')),
            'planner_intent_excerpt': str(planner_debug.get('intent_excerpt', '')),
            'hazard_debug_stats': dict(hazard_debug_stats),
        }

    def _extract_planner_debug(self, planned_route: dict) -> dict:
        """Extract compact planner diagnostics for logging overlays and sidecar files."""
        debug = planned_route.get('_debug', {}) if isinstance(planned_route, dict) else {}

        scene = str(debug.get('scene_description', '') or '')
        objects = str(debug.get('object_description', '') or '')
        intent = str(debug.get('intent_description', '') or '')
        merged = f"{scene} {objects}".lower()

        return {
            'zero_deadlock_streak': int(debug.get('zero_deadlock_streak', 0) or 0),
            'zero_override_candidate': bool(debug.get('zero_override_candidate', False)),
            'static_guard_applied': bool(debug.get('static_guard_applied', False)),
            'static_guard_reason': str(debug.get('static_guard_reason', '')),
            'hazard_signals': debug.get('hazard_signals', {}),
            'scene_excerpt': scene[:260],
            'objects_excerpt': objects[:260],
            'intent_excerpt': intent[:260],
            'wall_barrier_mentioned': any(tok in merged for tok in ('wall', 'barrier', 'guardrail', 'curb', 'fence')),
        }
    
    def _setup_collision_sensor(self, route_metric_id: str):
        """Setup collision detection sensor."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        last_collision_frame = {'value': -999999}
        
        def on_collision(event):
            frame = int(event.frame)
            if frame - last_collision_frame['value'] < 15:
                return
            last_collision_frame['value'] = frame
            logger.warning(f"Collision detected with {event.other_actor.type_id}")
            self.metrics_calculator.record_event(
                route_metric_id,
                frame,
                'collision',
                description=event.other_actor.type_id
            )
        
        collision_sensor.listen(on_collision)
        return collision_sensor
    
    def _save_camera_image(self, image: np.ndarray, scenario_id: str, step: int, metrics_snapshot: dict | None = None):
        """Save camera image to disk for visualization."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            scenario_img_dir = self.images_dir / scenario_id
            scenario_img_dir.mkdir(parents=True, exist_ok=True)
            
            img_path = scenario_img_dir / f"frame_{step:06d}.jpg"
            pil_image = Image.fromarray(image.astype(np.uint8))
            if metrics_snapshot:
                draw = ImageDraw.Draw(pil_image, 'RGBA')
                font = ImageFont.load_default()

                header_lines = [
                    f"Step {metrics_snapshot['step']}  |  Sim {metrics_snapshot['simulation_progress_pct']:.1f}%  |  Route {metrics_snapshot['rs']:.1f}%  |  DS {metrics_snapshot['ds']:.1f}",
                    f"Speed {metrics_snapshot['speed_mps']:.1f} m/s  |  Target {metrics_snapshot.get('target_speed_mps', 0.0):.1f} m/s  |  Steer {metrics_snapshot.get('steer', 0.0):+.3f} ({metrics_snapshot.get('steer_deg', 0.0):+.1f} deg)",
                    f"Dist {metrics_snapshot['distance_m']:.1f}/{metrics_snapshot['route_distance_m']:.1f} m  |  Collisions {metrics_snapshot['collisions']}  |  Violations {metrics_snapshot['violations']}",
                    f"Hazard static={metrics_snapshot.get('static_hazard_detected', False)} proximity={metrics_snapshot.get('hazard_proximity_detected', False)} guard={metrics_snapshot.get('static_guard_applied', False)} side={metrics_snapshot.get('hazard_side', 'none')}",
                    f"Route ID: {metrics_snapshot['route_id']}"
                ]

                wrapped_lines = []
                for line in header_lines:
                    wrapped_lines.extend(textwrap.wrap(line, width=70) or [line])

                line_height = 16
                padding = 10
                box_height = padding * 2 + line_height * len(wrapped_lines)
                box_width = min(pil_image.width - 20, 760)

                draw.rounded_rectangle(
                    [(10, 10), (10 + box_width, 10 + box_height)],
                    radius=12,
                    fill=(0, 0, 0, 170)
                )

                text_y = 10 + padding
                for line in wrapped_lines:
                    draw.text((20, text_y), line, fill=(255, 255, 255, 255), font=font)
                    text_y += line_height

            pil_image.save(img_path, quality=85)
            if metrics_snapshot:
                self._write_frame_debug_log(
                    scenario_id=scenario_id,
                    step=step,
                    image_path=img_path,
                    metrics_snapshot=metrics_snapshot
                )
        except Exception as e:
            logger.debug(f"Failed to save image at step {step}: {e}")

    def _write_frame_debug_log(self, scenario_id: str, step: int, image_path: Path, metrics_snapshot: dict):
        """Write sidecar JSON and append JSONL trace next to saved images for easy copying."""
        scenario_img_dir = self.images_dir / scenario_id
        scenario_img_dir.mkdir(parents=True, exist_ok=True)

        frame_payload = dict(metrics_snapshot)
        frame_payload['scenario_id'] = scenario_id
        frame_payload['frame_step'] = int(step)
        frame_payload['image_file'] = image_path.name

        sidecar_path = scenario_img_dir / f"frame_{step:06d}.debug.json"
        with sidecar_path.open('w', encoding='utf-8') as f:
            json.dump(frame_payload, f, indent=2)

        trace_path = scenario_img_dir / 'debug_trace.jsonl'
        with trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(frame_payload, ensure_ascii=True) + "\n")

    def _write_scenario_debug_summary(self, scenario_id: str, final_snapshot: dict, hazard_debug_stats: dict):
        """Write compact scenario-level hazard/perception summary into image folder."""
        scenario_img_dir = self.images_dir / scenario_id
        scenario_img_dir.mkdir(parents=True, exist_ok=True)

        sampled = max(int(hazard_debug_stats.get('sampled_frames', 0)), 1)
        summary_payload = {
            'scenario_id': scenario_id,
            'route_id': final_snapshot.get('route_id', ''),
            'final_route_completion_pct': float(final_snapshot.get('rc', 0.0)),
            'final_driving_score': float(final_snapshot.get('ds', 0.0)),
            'final_collisions': int(final_snapshot.get('collisions', 0)),
            'hazard_debug_stats': dict(hazard_debug_stats),
            'hazard_recall_indicators': {
                'static_hazard_detection_rate': float(hazard_debug_stats.get('sampled_static_hazard_frames', 0)) / sampled,
                'proximity_detection_rate': float(hazard_debug_stats.get('sampled_proximity_frames', 0)) / sampled,
                'guard_application_rate': float(hazard_debug_stats.get('sampled_guard_applied_frames', 0)) / sampled,
                'wall_barrier_mention_rate': float(hazard_debug_stats.get('sampled_wall_barrier_mentions', 0)) / sampled,
            },
            'latest_scene_excerpt': str(final_snapshot.get('planner_scene_excerpt', '')),
            'latest_objects_excerpt': str(final_snapshot.get('planner_objects_excerpt', '')),
            'latest_intent_excerpt': str(final_snapshot.get('planner_intent_excerpt', '')),
        }

        summary_path = scenario_img_dir / 'hazard_debug_summary.json'
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary_payload, f, indent=2)
    
    def run_tests(
        self,
        scenario_ids: list = None,
        max_steps: int = 100
    ):
        """
        Run multiple test scenarios.
        
        Args:
            scenario_ids: List of scenario IDs to run (None = all)
            max_steps: Maximum steps per scenario
        """
        if not self.connect_to_carla():
            return
        
        if scenario_ids:
            selected_ids = scenario_ids
        else:
            selected_ids = self.scenario_manager.list_scenarios()

        scenarios = []
        for sid in selected_ids:
            scenario = self.scenario_manager.get_scenario(sid)
            if scenario is not None:
                scenarios.append(scenario)
        
        logger.info(f"Running {len(scenarios)} scenarios")
        if not scenarios:
            logger.warning("No scenarios found. Add scenario JSON files under test_runner/scenarios")
            return
        
        for scenario in scenarios:
            route = self.scenario_manager.get_route(scenario.route_id)
            if route is None:
                logger.warning(f"Skipping scenario {scenario.scenario_id}: missing route {scenario.route_id}")
                continue
            
            # Load appropriate map
            map_name = route.map_name
            self.load_map(map_name)
            
            # Run scenario
            self.run_scenario(scenario, route, max_steps=max_steps)
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self):
        """Generate evaluation report with plots."""
        logger.info("Generating evaluation report...")
        summary = self.metrics_calculator.get_summary()
        self.visualizer.add_metrics(summary)
        
        self.visualizer.export_json(output_dir=str(self.results_dir), filename='metrics.json')
        self.visualizer.save_detailed_report(output_dir=str(self.results_dir), filename='detailed_report.txt')
        self.visualizer.plot_summary(output_dir=str(self.results_dir))

        logger.info(f"✓ Results saved in: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='LangCoop Test Runner - Full Integration')
    parser.add_argument('--host', type=str, default='carla-rpc',
                        help='CARLA server hostname (default: carla-rpc)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--agent-config', type=str,
                        default='configs/langcoop_agent_config.yaml',
                        help='Agent configuration file')
    parser.add_argument('--scenario-ids', type=str, nargs='+',
                        default=['town05_clear_easy'],
                        help='Scenario IDs to run (default: town05_clear_easy)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per scenario (default: 500 = 25 seconds simulation)')
    parser.add_argument('--results-dir', type=str, default='test_results_langcoop',
                        help='Results directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("LangCoop Test Runner - Full Integration")
    logger.info("=" * 70)
    logger.info(f"CARLA Server: {args.host}:{args.port}")
    logger.info(f"Agent Config: {args.agent_config}")
    logger.info("=" * 70)
    
    runner = LangCoopTestRunner(
        carla_host=args.host,
        carla_port=args.port,
        agent_config=args.agent_config,
        results_dir=args.results_dir
    )
    
    runner.run_tests(
        scenario_ids=args.scenario_ids,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
