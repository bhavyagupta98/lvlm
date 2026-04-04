#!/usr/bin/env python3
"""
Separate multi-view LangCoop test runner.

Keeps the original run_langcoop_test.py flow untouched while adding:
- Front / left / right / rear / BEV camera capture
- Composite image logging with live metrics overlay
- Optional 2-ego simple scenario execution
"""

import argparse
import json
import logging
import sys
import textwrap
import time
from pathlib import Path

import carla
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from test_runner.agents.multiview_langcoop_agent import MultiViewLangCoopAgent
from test_runner.evaluator.metrics import MetricsCalculator
from test_runner.evaluator.scenario_manager import ScenarioManager
from test_runner.visualization import MetricsVisualizer, LiveMetricsTracer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiViewLangCoopTestRunner:
    """Dedicated runner for multi-view logging and optional dual-ego tests."""

    def __init__(
        self,
        carla_host: str = 'carla-rpc',
        carla_port: int = 2000,
        agent_config: str = 'configs/langcoop_agent_config.yaml',
        results_dir: str = 'test_results_langcoop_multiview',
        ego_num: int = 1,
    ):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.agent_config = agent_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.results_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        self.world = None
        self.ego_num = max(1, int(ego_num))

        self.agents = []
        self.vehicles = []
        self.metrics_calculator = MetricsCalculator()
        self.scenario_manager = ScenarioManager()
        self.visualizer = MetricsVisualizer()

    def connect_to_carla(self):
        try:
            logger.info("Connecting to CARLA at %s:%s...", self.carla_host, self.carla_port)
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()
            logger.info("Connected to CARLA successfully")
            logger.info("CARLA Server version: %s", self.client.get_server_version())
            return True
        except Exception as exc:
            logger.error("Failed to connect to CARLA: %s", exc)
            return False

    def load_map(self, map_name: str):
        logger.info("Loading map: %s", map_name)
        self.world = self.client.load_world(map_name)
        time.sleep(2)
        logger.info("Map loaded: %s", map_name)

    def setup_environment(self, scenario):
        weather = carla.WeatherParameters()
        if scenario.weather:
            weather.cloudiness = scenario.weather.get('cloudiness', 30)
            weather.precipitation = scenario.weather.get('precipitation', 0)
            weather.wind_intensity = scenario.weather.get('wind_intensity', 0)
            weather.sun_altitude_angle = scenario.weather.get('sun_altitude_angle', 45)
            weather.fog_density = scenario.weather.get('fog_density', 0)

        self.world.set_weather(weather)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _spawn_vehicle(self, spawn_point: carla.Transform):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

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
                return vehicle
        return None

    def _spawn_vehicles_for_route(self, route):
        self.vehicles = []
        spawn_indices = []
        for agent_idx in range(self.ego_num):
            route_idx = min(agent_idx, max(0, len(route.waypoints) - 2))
            spawn_indices.append(route_idx)

        for agent_idx, route_idx in enumerate(spawn_indices):
            spawn_point = route.waypoints[route_idx].to_transform()
            vehicle = self._spawn_vehicle(spawn_point)
            if vehicle is None:
                raise RuntimeError(f"Failed to spawn ego vehicle {agent_idx} at route index {route_idx}")
            self.vehicles.append(vehicle)
            logger.info(
                "Spawned ego_%d at route waypoint %d -> (%.1f, %.1f, %.1f)",
                agent_idx,
                route_idx,
                spawn_point.location.x,
                spawn_point.location.y,
                spawn_point.location.z,
            )
        return spawn_indices

    def _setup_agents(self):
        self.agents = []
        for idx, vehicle in enumerate(self.vehicles):
            agent = MultiViewLangCoopAgent(agent_config_path=self.agent_config)
            agent.setup(vehicle)
            self.agents.append(agent)
            logger.info("Initialized multiview agent_%d", idx)

    def _assign_route_to_agents(self, route, spawn_indices):
        world_map = self.world.get_map()
        for agent_idx, agent in enumerate(self.agents):
            start_idx = spawn_indices[agent_idx]
            for wp in route.waypoints[start_idx + 1:]:
                wp_location = carla.Location(
                    x=wp.location['x'],
                    y=wp.location['y'],
                    z=wp.location['z']
                )
                waypoint = world_map.get_waypoint(wp_location)
                if waypoint:
                    agent.set_target_waypoint(waypoint)

    def _setup_collision_sensor(self, vehicle, route_metric_id: str):
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        last_collision_frame = {'value': -999999}

        def on_collision(event):
            frame = int(event.frame)
            if frame - last_collision_frame['value'] < 15:
                return
            last_collision_frame['value'] = frame
            self.metrics_calculator.record_event(
                route_metric_id,
                frame,
                'collision',
                description=event.other_actor.type_id
            )

        collision_sensor.listen(on_collision)
        return collision_sensor

    def _extract_planner_debug(self, planned_route: dict) -> dict:
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
        hazard_debug_stats: dict,
        current_control,
    ) -> dict:
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
            'throttle': float(getattr(current_control, 'throttle', 0.0)),
            'brake': float(getattr(current_control, 'brake', 0.0)),
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

    def _compose_multiview_panel(self, agent, metrics_snapshot: dict | None):
        from PIL import Image, ImageDraw, ImageFont

        view_order = [
            ('front_camera', 'Front'),
            ('left_camera', 'Left'),
            ('right_camera', 'Right'),
            ('rear_camera', 'Rear'),
            ('bev_camera', 'BEV'),
        ]

        available = []
        for key, label in view_order:
            image = agent.sensor_data.get(key)
            if image is None:
                continue
            available.append((key, label, Image.fromarray(image.astype(np.uint8))))

        if not available:
            return None

        cell_w = 480
        cell_h = 320
        cols = 2
        rows = (len(available) + cols - 1) // cols
        header_h = 110 if metrics_snapshot else 24
        panel = Image.new('RGB', (cols * cell_w, rows * cell_h + header_h), (18, 22, 30))
        draw = ImageDraw.Draw(panel, 'RGBA')
        font = ImageFont.load_default()

        if metrics_snapshot:
            header_lines = [
                f"Step {metrics_snapshot['step']} | Sim {metrics_snapshot['simulation_progress_pct']:.1f}% | RC {metrics_snapshot['rc']:.1f}% | DS {metrics_snapshot['ds']:.1f}",
                f"Speed {metrics_snapshot['speed_mps']:.1f} m/s | Target {metrics_snapshot['target_speed_mps']:.1f} m/s | Steer {metrics_snapshot['steer']:+.3f} ({metrics_snapshot['steer_deg']:+.1f} deg)",
                f"Throttle {metrics_snapshot['throttle']:.2f} | Brake {metrics_snapshot['brake']:.2f} | Collisions {metrics_snapshot['collisions']} | Violations {metrics_snapshot['violations']}",
                f"Route {metrics_snapshot['route_id']} | Hazard static={metrics_snapshot.get('static_hazard_detected', False)} proximity={metrics_snapshot.get('hazard_proximity_detected', False)} guard={metrics_snapshot.get('static_guard_applied', False)}",
            ]
            draw.rounded_rectangle([(10, 10), (panel.width - 10, header_h - 10)], radius=12, fill=(0, 0, 0, 170))
            y = 18
            for line in header_lines:
                for wrapped in textwrap.wrap(line, width=110) or [line]:
                    draw.text((20, y), wrapped, fill=(255, 255, 255, 255), font=font)
                    y += 16

        for idx, (_key, label, image) in enumerate(available):
            row = idx // cols
            col = idx % cols
            x0 = col * cell_w
            y0 = header_h + row * cell_h
            tile = image.resize((cell_w, cell_h))
            panel.paste(tile, (x0, y0))
            draw.rectangle([(x0 + 10, y0 + 10), (x0 + 110, y0 + 34)], fill=(0, 0, 0, 160))
            draw.text((x0 + 18, y0 + 17), label, fill=(255, 255, 255, 255), font=font)

        return panel

    def _write_frame_debug_log(self, scenario_id: str, agent_idx: int, step: int, image_path: Path, metrics_snapshot: dict):
        scenario_img_dir = self.images_dir / scenario_id / f'agent_{agent_idx}'
        scenario_img_dir.mkdir(parents=True, exist_ok=True)

        frame_payload = dict(metrics_snapshot)
        frame_payload['scenario_id'] = scenario_id
        frame_payload['agent_idx'] = int(agent_idx)
        frame_payload['frame_step'] = int(step)
        frame_payload['image_file'] = image_path.name

        sidecar_path = scenario_img_dir / f"frame_{step:06d}.debug.json"
        with sidecar_path.open('w', encoding='utf-8') as f:
            json.dump(frame_payload, f, indent=2)

        trace_path = scenario_img_dir / 'debug_trace.jsonl'
        with trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(frame_payload, ensure_ascii=True) + "\n")

    def _save_multiview_image(self, scenario_id: str, agent_idx: int, step: int, agent, metrics_snapshot: dict | None = None):
        try:
            scenario_img_dir = self.images_dir / scenario_id / f'agent_{agent_idx}'
            scenario_img_dir.mkdir(parents=True, exist_ok=True)
            img_path = scenario_img_dir / f'frame_{step:06d}.jpg'
            panel = self._compose_multiview_panel(agent, metrics_snapshot)
            if panel is None:
                return
            panel.save(img_path, quality=85)
            if metrics_snapshot:
                self._write_frame_debug_log(scenario_id, agent_idx, step, img_path, metrics_snapshot)
        except Exception as exc:
            logger.debug("Failed to save multiview frame for agent_%d step %d: %s", agent_idx, step, exc)

    def _write_scenario_debug_summary(self, scenario_id: str, agent_idx: int, final_snapshot: dict, hazard_debug_stats: dict):
        scenario_img_dir = self.images_dir / scenario_id / f'agent_{agent_idx}'
        scenario_img_dir.mkdir(parents=True, exist_ok=True)

        sampled = max(int(hazard_debug_stats.get('sampled_frames', 0)), 1)
        summary_payload = {
            'scenario_id': scenario_id,
            'agent_idx': int(agent_idx),
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

    def run_scenario(self, scenario, route, max_steps: int = 500, skip_frames: int = 4):
        logger.info("Starting multiview scenario: %s", scenario.scenario_id)
        route_distance = route.compute_distance()

        self.setup_environment(scenario)
        spawn_indices = self._spawn_vehicles_for_route(route)
        self._setup_agents()
        self._assign_route_to_agents(route, spawn_indices)

        route_metric_ids = [f"{scenario.scenario_id}_agent_{idx}" for idx in range(len(self.agents))]
        for route_metric_id in route_metric_ids:
            self.metrics_calculator.set_route_completion(route_metric_id, 0.0, route_distance, 0.0)
            self.metrics_calculator.set_speed_metrics(route_metric_id, 0.0, 0.0)

        collision_sensors = [
            self._setup_collision_sensor(vehicle, route_metric_ids[idx])
            for idx, vehicle in enumerate(self.vehicles)
        ]

        start_time = time.time()
        prev_locations = [vehicle.get_location() for vehicle in self.vehicles]
        first_distance_samples = [True for _ in self.vehicles]
        completed_distances = [0.0 for _ in self.vehicles]
        speed_samples = [[] for _ in self.vehicles]
        last_controls = [carla.VehicleControl() for _ in self.vehicles]
        last_target_speeds = [0.0 for _ in self.vehicles]
        last_planner_debugs = [{} for _ in self.vehicles]
        hazard_debug_stats = [
            {
                'sampled_frames': 0,
                'sampled_static_hazard_frames': 0,
                'sampled_proximity_frames': 0,
                'sampled_guard_applied_frames': 0,
                'sampled_wall_barrier_mentions': 0,
                'saved_debug_frames': 0,
            }
            for _ in self.vehicles
        ]
        live_trace_interval = 10
        live_tracers = [
            LiveMetricsTracer(str(self.results_dir), scenario.scenario_id, route_metric_ids[idx])
            for idx in range(len(self.agents))
        ]
        latest_snapshots = [None for _ in self.vehicles]

        try:
            for step in range(max_steps):
                self.world.tick()

                if step % skip_frames == 0:
                    for idx, (agent, vehicle) in enumerate(zip(self.agents, self.vehicles)):
                        control = agent.step()
                        vehicle.apply_control(control)
                        last_controls[idx] = control
                        if isinstance(agent.last_planned_route, dict):
                            target_speed_arr = agent.last_planned_route.get('target_speed', [0.0])
                            if isinstance(target_speed_arr, list) and target_speed_arr:
                                last_target_speeds[idx] = float(target_speed_arr[0])
                            else:
                                last_target_speeds[idx] = float(target_speed_arr)
                            last_planner_debugs[idx] = self._extract_planner_debug(agent.last_planned_route)

                for idx, vehicle in enumerate(self.vehicles):
                    current_location = vehicle.get_location()
                    if first_distance_samples[idx]:
                        distance_delta = 0.0
                        first_distance_samples[idx] = False
                    else:
                        distance_delta = current_location.distance(prev_locations[idx])
                    completed_distances[idx] += distance_delta
                    prev_locations[idx] = current_location

                    velocity = vehicle.get_velocity()
                    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                    speed_samples[idx].append(speed)

                    if speed > 20.5:
                        self.metrics_calculator.record_event(
                            route_metric_ids[idx],
                            step,
                            'speed_violation',
                            description=f"speed={speed:.2f}"
                        )

                    if step % live_trace_interval == 0:
                        snapshot = self._build_live_metrics_snapshot(
                            route_metric_id=route_metric_ids[idx],
                            step=step,
                            max_steps=max_steps,
                            completed_distance=completed_distances[idx],
                            route_distance=route_distance,
                            start_time=start_time,
                            speed_samples=speed_samples[idx],
                            current_speed=speed,
                            current_steer=float(last_controls[idx].steer),
                            target_speed=last_target_speeds[idx],
                            planner_debug=last_planner_debugs[idx],
                            hazard_debug_stats=hazard_debug_stats[idx],
                            current_control=last_controls[idx],
                        )
                        latest_snapshots[idx] = snapshot

                        hazard_debug_stats[idx]['sampled_frames'] += 1
                        if snapshot.get('static_hazard_detected', False):
                            hazard_debug_stats[idx]['sampled_static_hazard_frames'] += 1
                        if snapshot.get('hazard_proximity_detected', False):
                            hazard_debug_stats[idx]['sampled_proximity_frames'] += 1
                        if snapshot.get('static_guard_applied', False):
                            hazard_debug_stats[idx]['sampled_guard_applied_frames'] += 1
                        if snapshot.get('wall_barrier_mentioned', False):
                            hazard_debug_stats[idx]['sampled_wall_barrier_mentions'] += 1

                        live_tracers[idx].update(snapshot)

                    if step % 10 == 0:
                        self._save_multiview_image(
                            scenario.scenario_id,
                            idx,
                            step,
                            self.agents[idx],
                            metrics_snapshot=latest_snapshots[idx]
                        )
                        hazard_debug_stats[idx]['saved_debug_frames'] += 1

                if step % 100 == 0:
                    progress = [
                        (completed_distances[idx] / route_distance) * 100 if route_distance > 0 else 0.0
                        for idx in range(len(self.vehicles))
                    ]
                    logger.info("Step %d/%d | Progress: %s", step, max_steps, ", ".join(f"ego_{i}={p:.1f}%" for i, p in enumerate(progress)))
        finally:
            for sensor in collision_sensors:
                if sensor:
                    try:
                        sensor.stop()
                    except Exception:
                        pass
                    try:
                        sensor.destroy()
                    except Exception:
                        pass
            for agent in self.agents:
                agent.destroy()
            for vehicle in self.vehicles:
                if vehicle and vehicle.is_alive:
                    vehicle.destroy()

        elapsed_time = time.time() - start_time
        for idx, route_metric_id in enumerate(route_metric_ids):
            self.metrics_calculator.set_route_completion(
                route_metric_id,
                completed_distances[idx],
                route_distance,
                elapsed_time
            )
            if speed_samples[idx]:
                self.metrics_calculator.set_speed_metrics(
                    route_metric_id,
                    float(np.mean(speed_samples[idx])),
                    float(np.max(speed_samples[idx]))
                )
            final_ds = self.metrics_calculator.calculate_driving_score(route_metric_id)
            final_route_metrics = self.metrics_calculator.get_route_metrics(route_metric_id)
            final_violations = final_route_metrics.lane_departures + final_route_metrics.speed_violations
            final_speed = speed_samples[idx][-1] if speed_samples[idx] else 0.0
            final_snapshot = {
                'step': int(max_steps),
                'simulation_progress_pct': 100.0,
                'elapsed_wall_time_sec': float(elapsed_time),
                'distance_m': float(completed_distances[idx]),
                'route_distance_m': float(route_distance),
                'rs': float(final_route_metrics.completion_percentage),
                'rc': float(final_route_metrics.completion_percentage),
                'ds': float(final_ds),
                'speed_mps': float(final_speed),
                'target_speed_mps': float(last_target_speeds[idx]),
                'steer': float(last_controls[idx].steer),
                'steer_deg': float(last_controls[idx].steer * 70.0),
                'throttle': float(last_controls[idx].throttle),
                'brake': float(last_controls[idx].brake),
                'collisions': int(final_route_metrics.collisions),
                'violations': int(final_violations),
                'route_id': route_metric_id,
                'static_hazard_detected': bool(last_planner_debugs[idx].get('hazard_signals', {}).get('has_static_hazard', False)),
                'hazard_proximity_detected': bool(last_planner_debugs[idx].get('hazard_signals', {}).get('has_proximity_cue', False)),
                'static_guard_applied': bool(last_planner_debugs[idx].get('static_guard_applied', False)),
                'wall_barrier_mentioned': bool(last_planner_debugs[idx].get('wall_barrier_mentioned', False)),
                'hazard_side': str(last_planner_debugs[idx].get('hazard_signals', {}).get('hazard_side', 'none')),
                'static_guard_reason': str(last_planner_debugs[idx].get('static_guard_reason', '')),
                'planner_scene_excerpt': str(last_planner_debugs[idx].get('scene_excerpt', '')),
                'planner_objects_excerpt': str(last_planner_debugs[idx].get('objects_excerpt', '')),
                'planner_intent_excerpt': str(last_planner_debugs[idx].get('intent_excerpt', '')),
                'hazard_debug_stats': dict(hazard_debug_stats[idx]),
            }
            live_tracers[idx].update(final_snapshot)
            self._write_scenario_debug_summary(scenario.scenario_id, idx, final_snapshot, hazard_debug_stats[idx])

        metrics = self.metrics_calculator.get_summary()
        logger.info("Scenario complete: %s", scenario.scenario_id)
        for route_metric_id in route_metric_ids:
            route_metrics = metrics.get('routes', {}).get(route_metric_id, {})
            logger.info(
                "  %s | RC %.1f%% | DS %.1f | Collisions %s | Violations %s",
                route_metric_id,
                route_metrics.get('rc', 0.0),
                route_metrics.get('ds', 0.0),
                route_metrics.get('collisions', 0),
                route_metrics.get('violations', 0),
            )
        return metrics

    def run_tests(self, scenario_ids=None, max_steps: int = 500):
        if not self.connect_to_carla():
            return

        selected_ids = scenario_ids if scenario_ids else self.scenario_manager.list_scenarios()
        scenarios = []
        for sid in selected_ids:
            scenario = self.scenario_manager.get_scenario(sid)
            if scenario is not None:
                scenarios.append(scenario)

        if not scenarios:
            logger.warning("No scenarios found.")
            return

        for scenario in scenarios:
            route = self.scenario_manager.get_route(scenario.route_id)
            if route is None:
                logger.warning("Skipping scenario %s: missing route %s", scenario.scenario_id, scenario.route_id)
                continue
            self.load_map(route.map_name)
            self.run_scenario(scenario, route, max_steps=max_steps)

        summary = self.metrics_calculator.get_summary()
        self.visualizer.add_metrics(summary)
        self.visualizer.export_json(output_dir=str(self.results_dir), filename='metrics.json')
        self.visualizer.save_detailed_report(output_dir=str(self.results_dir), filename='detailed_report.txt')
        self.visualizer.plot_summary(output_dir=str(self.results_dir))
        logger.info("Results saved in: %s", self.results_dir)


def main():
    parser = argparse.ArgumentParser(description='LangCoop MultiView Test Runner')
    parser.add_argument('--host', type=str, default='carla-rpc')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--agent-config', type=str, default='configs/langcoop_agent_config_32b.yaml')
    parser.add_argument('--scenario-ids', type=str, nargs='+', default=['town05_cloudy_medium'])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--results-dir', type=str, default='test_results_langcoop_multiview')
    parser.add_argument('--ego-num', type=int, default=1, help='1 for single ego, 2 for simple dual-ego run')
    args = parser.parse_args()

    runner = MultiViewLangCoopTestRunner(
        carla_host=args.host,
        carla_port=args.port,
        agent_config=args.agent_config,
        results_dir=args.results_dir,
        ego_num=args.ego_num,
    )
    runner.run_tests(scenario_ids=args.scenario_ids, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
