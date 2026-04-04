"""Shared runner utilities for single-view and multiview LangCoop execution."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import carla
import numpy as np

from test_runner.evaluator.metrics import MetricsCalculator
from test_runner.evaluator.scenario_manager import ScenarioManager
from test_runner.visualization import MetricsVisualizer


logger = logging.getLogger(__name__)


class BaseLangCoopRunner:
    """Common CARLA runner functionality shared by multiple entrypoints."""

    def __init__(
        self,
        carla_host: str,
        carla_port: int,
        agent_config: str,
        results_dir: str,
    ):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.agent_config = agent_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.results_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        self.world = None

        self.metrics_calculator = MetricsCalculator()
        self.scenario_manager = ScenarioManager()
        self.visualizer = MetricsVisualizer()

    def _snap_location_to_driving_waypoint(self, location: carla.Location):
        """Project an arbitrary location onto the nearest drivable lane waypoint."""
        world_map = self.world.get_map()
        try:
            return world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except TypeError:
            # Older CARLA builds may not expose keyword-only variants consistently.
            return world_map.get_waypoint(location)

    def _snap_transform_to_driving_lane(self, transform: carla.Transform) -> carla.Transform:
        """Snap a transform onto a valid driving-lane transform while preserving a safe spawn height."""
        snapped_waypoint = self._snap_location_to_driving_waypoint(transform.location)
        if snapped_waypoint is None:
            return transform

        snapped_transform = snapped_waypoint.transform
        snapped_transform.location.z = max(snapped_transform.location.z, transform.location.z)
        return snapped_transform

    def _resolve_spawn_indices(self, route, ego_route_spacing: int, ego_num: int) -> list[int]:
        """Choose safer spawn indices for multi-ego scenarios."""
        route_len = len(route.waypoints)
        if route_len == 0:
            return []

        max_start_idx = max(0, route_len - 2)
        if ego_num <= 1:
            return [0]

        min_stride = 2
        requested_stride = max(1, int(ego_route_spacing))
        stride = max(requested_stride, min_stride)

        spawn_indices = []
        for agent_idx in range(ego_num):
            route_idx = min(agent_idx * stride, max_start_idx)
            spawn_indices.append(route_idx)

        return spawn_indices

    def _route_transform_at_index(self, route, route_idx: int) -> carla.Transform:
        """Get a route transform snapped to the map's driving lane."""
        route_idx = max(0, min(int(route_idx), len(route.waypoints) - 1))
        requested_transform = route.waypoints[route_idx].to_transform()
        return self._snap_transform_to_driving_lane(requested_transform)

    def connect_to_carla(self) -> bool:
        """Connect to the CARLA server."""
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
            logger.error("Make sure CARLA server is running on %s:%s", self.carla_host, self.carla_port)
            return False

    def load_map(self, map_name: str):
        """Load the requested CARLA map."""
        logger.info("Loading map: %s", map_name)
        self.world = self.client.load_world(map_name)
        time.sleep(2.0)
        logger.info("Map loaded: %s", map_name)

    def setup_environment(self, scenario):
        """Apply weather and synchronous simulation settings."""
        weather = carla.WeatherParameters()
        if scenario.weather:
            weather.cloudiness = scenario.weather.get("cloudiness", 30)
            weather.precipitation = scenario.weather.get("precipitation", 0)
            weather.wind_intensity = scenario.weather.get("wind_intensity", 0)
            weather.sun_altitude_angle = scenario.weather.get("sun_altitude_angle", 45)
            weather.fog_density = scenario.weather.get("fog_density", 0)

        self.world.set_weather(weather)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _spawn_vehicle_with_retries(self, spawn_point: carla.Transform):
        """Spawn an ego vehicle with small z-offset retries, then map fallbacks."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = self._snap_transform_to_driving_lane(spawn_point)

        for z_offset in [0.0, 0.3, 0.6, 1.0]:
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
                return vehicle, candidate, False

        for fallback in self.world.get_map().get_spawn_points():
            vehicle = self.world.try_spawn_actor(vehicle_bp, fallback)
            if vehicle is not None:
                return vehicle, fallback, True

        raise RuntimeError("Failed to spawn vehicle: all candidate spawn points are occupied")

    def _setup_collision_sensor(self, vehicle, route_metric_id: str):
        """Attach a collision sensor and record debounced collision events."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        last_collision_frame = {"value": -999999}

        def on_collision(event):
            frame = int(event.frame)
            if frame - last_collision_frame["value"] < 15:
                return
            last_collision_frame["value"] = frame
            self.metrics_calculator.record_event(
                route_metric_id,
                frame,
                "collision",
                description=event.other_actor.type_id,
            )

        collision_sensor.listen(on_collision)
        return collision_sensor

    def _extract_planner_debug(self, planned_route: dict) -> dict:
        """Extract compact planner diagnostics for overlays and sidecar logs."""
        debug = planned_route.get("_debug", {}) if isinstance(planned_route, dict) else {}
        scene = str(debug.get("scene_description", "") or "")
        objects = str(debug.get("object_description", "") or "")
        intent = str(debug.get("intent_description", "") or "")
        merged = f"{scene} {objects}".lower()

        return {
            "zero_deadlock_streak": int(debug.get("zero_deadlock_streak", 0) or 0),
            "zero_override_candidate": bool(debug.get("zero_override_candidate", False)),
            "static_guard_applied": bool(debug.get("static_guard_applied", False)),
            "static_guard_reason": str(debug.get("static_guard_reason", "")),
            "hazard_signals": debug.get("hazard_signals", {}),
            "scene_full": scene,
            "objects_full": objects,
            "intent_full": intent,
            "target_description": str(debug.get("target_description", "") or ""),
            "scene_length": int(debug.get("scene_description_length", len(scene))),
            "objects_length": int(debug.get("object_description_length", len(objects))),
            "intent_length": int(debug.get("intent_description_length", len(intent))),
            "scene_excerpt": scene[:260],
            "objects_excerpt": objects[:260],
            "intent_excerpt": intent[:260],
            "wall_barrier_mentioned": any(
                token in merged for token in ("wall", "barrier", "guardrail", "curb", "fence")
            ),
        }

    @staticmethod
    def _bucket_distance(distance_m: float) -> str:
        if distance_m < 10.0:
            return "near"
        if distance_m < 25.0:
            return "mid"
        return "far"

    @staticmethod
    def _bucket_side(lateral_m: float) -> str:
        if lateral_m > 1.5:
            return "right"
        if lateral_m < -1.5:
            return "left"
        return "center"

    def _collect_dynamic_ground_truth(self, ego_vehicle, max_distance_m: float = 40.0, front_fov_deg: float = 100.0) -> dict:
        """Collect simple CARLA ground-truth actor data relative to one ego vehicle."""
        if ego_vehicle is None or self.world is None:
            return {
                "visible_actor_count": 0,
                "all_nearby_actor_count": 0,
                "visible_vehicle_count": 0,
                "visible_walker_count": 0,
                "dominant_side": "none",
                "actors": [],
            }

        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(-ego_yaw)
        sin_yaw = math.sin(-ego_yaw)
        half_fov = front_fov_deg / 2.0

        actors_payload = []
        visible_actors = []

        for actor in self.world.get_actors():
            if actor.id == ego_vehicle.id:
                continue

            actor_type = str(getattr(actor, "type_id", ""))
            if actor_type.startswith("vehicle."):
                actor_class = "vehicle"
            elif actor_type.startswith("walker.pedestrian."):
                actor_class = "walker"
            else:
                continue

            actor_loc = actor.get_location()
            dx = actor_loc.x - ego_loc.x
            dy = actor_loc.y - ego_loc.y
            distance_m = float(math.hypot(dx, dy))
            if distance_m > max_distance_m:
                continue

            lateral_m = sin_yaw * dx + cos_yaw * dy
            local_y = -cos_yaw * dx + sin_yaw * dy
            longitudinal_m = -local_y
            bearing_deg = float(math.degrees(math.atan2(lateral_m, max(longitudinal_m, 1e-6))))
            visible_front = longitudinal_m > 0.0 and abs(bearing_deg) <= half_fov

            actor_payload = {
                "actor_id": int(actor.id),
                "actor_type": actor_type,
                "actor_class": actor_class,
                "distance_m": round(distance_m, 3),
                "distance_bucket": self._bucket_distance(distance_m),
                "side": self._bucket_side(lateral_m),
                "lateral_m": round(float(lateral_m), 3),
                "longitudinal_m": round(float(longitudinal_m), 3),
                "bearing_deg": round(bearing_deg, 3),
                "visible_front": bool(visible_front),
            }
            actors_payload.append(actor_payload)
            if visible_front:
                visible_actors.append(actor_payload)

        side_counts = {"left": 0, "center": 0, "right": 0}
        visible_vehicle_count = 0
        visible_walker_count = 0
        for actor in visible_actors:
            side_counts[actor["side"]] = side_counts.get(actor["side"], 0) + 1
            if actor["actor_class"] == "vehicle":
                visible_vehicle_count += 1
            elif actor["actor_class"] == "walker":
                visible_walker_count += 1

        dominant_side = "none"
        if any(count > 0 for count in side_counts.values()):
            dominant_side = max(side_counts, key=side_counts.get)

        return {
            "visible_actor_count": len(visible_actors),
            "all_nearby_actor_count": len(actors_payload),
            "visible_vehicle_count": visible_vehicle_count,
            "visible_walker_count": visible_walker_count,
            "dominant_side": dominant_side,
            "actors": visible_actors,
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
        current_control=None,
        gt_dynamic_perception: dict | None = None,
    ) -> dict:
        """Build one consistent metrics payload used by logs, traces, and frame overlays."""
        elapsed_time = time.time() - start_time
        self.metrics_calculator.set_route_completion(
            route_metric_id,
            completed_distance,
            route_distance,
            elapsed_time,
        )
        if speed_samples:
            self.metrics_calculator.set_speed_metrics(
                route_metric_id,
                float(np.mean(speed_samples)),
                float(np.max(speed_samples)),
            )

        live_ds = self.metrics_calculator.calculate_driving_score(route_metric_id)
        live_route_metrics = self.metrics_calculator.get_route_metrics(route_metric_id)
        live_violations = live_route_metrics.lane_departures + live_route_metrics.speed_violations
        progress_pct = ((step + 1) / max_steps) * 100 if max_steps > 0 else 0.0

        snapshot = {
            "step": int(step),
            "simulation_progress_pct": float(progress_pct),
            "elapsed_wall_time_sec": float(elapsed_time),
            "distance_m": float(completed_distance),
            "route_distance_m": float(route_distance),
            "rs": float(live_route_metrics.completion_percentage),
            "rc": float(live_route_metrics.completion_percentage),
            "ds": float(live_ds),
            "speed_mps": float(current_speed),
            "target_speed_mps": float(target_speed),
            "steer": float(current_steer),
            "steer_deg": float(current_steer * 70.0),
            "throttle": float(getattr(current_control, "throttle", 0.0)),
            "brake": float(getattr(current_control, "brake", 0.0)),
            "collisions": int(live_route_metrics.collisions),
            "violations": int(live_violations),
            "route_id": route_metric_id,
            "static_hazard_detected": bool(planner_debug.get("hazard_signals", {}).get("has_static_hazard", False)),
            "hazard_proximity_detected": bool(planner_debug.get("hazard_signals", {}).get("has_proximity_cue", False)),
            "static_guard_applied": bool(planner_debug.get("static_guard_applied", False)),
            "wall_barrier_mentioned": bool(planner_debug.get("wall_barrier_mentioned", False)),
            "hazard_side": str(planner_debug.get("hazard_signals", {}).get("hazard_side", "none")),
            "static_guard_reason": str(planner_debug.get("static_guard_reason", "")),
            "planner_scene_excerpt": str(planner_debug.get("scene_excerpt", "")),
            "planner_objects_excerpt": str(planner_debug.get("objects_excerpt", "")),
            "planner_intent_excerpt": str(planner_debug.get("intent_excerpt", "")),
            "planner_scene_full": str(planner_debug.get("scene_full", "")),
            "planner_objects_full": str(planner_debug.get("objects_full", "")),
            "planner_intent_full": str(planner_debug.get("intent_full", "")),
            "planner_target_description": str(planner_debug.get("target_description", "")),
            "planner_scene_length": int(planner_debug.get("scene_length", 0)),
            "planner_objects_length": int(planner_debug.get("objects_length", 0)),
            "planner_intent_length": int(planner_debug.get("intent_length", 0)),
            "hazard_debug_stats": dict(hazard_debug_stats),
        }
        if gt_dynamic_perception is not None:
            snapshot["gt_dynamic_perception"] = dict(gt_dynamic_perception)
        return snapshot

    @staticmethod
    def _safe_stop_and_destroy(actor):
        """Stop and destroy a CARLA actor if it exists."""
        if actor is None:
            return
        try:
            actor.stop()
        except Exception:
            pass
        try:
            actor.destroy()
        except Exception:
            pass

    @staticmethod
    def _safe_destroy(actor):
        """Destroy a CARLA actor if it exists and is alive."""
        if actor is None:
            return
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception:
            pass
