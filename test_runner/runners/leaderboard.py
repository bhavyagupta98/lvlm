"""Leaderboard-style collaborative multiview runner."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from test_runner.agents.multiview_langcoop_agent import MultiViewLangCoopAgent
from test_runner.visualization import LiveMetricsTracer

from .common import BaseLangCoopRunner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LeaderboardLangCoopRunner(BaseLangCoopRunner):
    """Collaborative runner with multiview logging and leaderboard-style background actors."""

    def __init__(
        self,
        carla_host: str = "carla-rpc",
        carla_port: int = 2000,
        agent_config: str = "configs/langcoop_agent_config_32b.yaml",
        results_dir: str = "test_results_langcoop_leaderboard",
        ego_num: int = 2,
        traffic_manager_port: int = 8000,
    ):
        super().__init__(carla_host, carla_port, agent_config, results_dir)
        self.ego_num = max(1, int(ego_num))
        self.traffic_manager_port = int(traffic_manager_port)
        self.agents = []
        self.vehicles = []
        self.background_vehicles = []
        self.walker_actors = []
        self.walker_controller_actors = []
        self.shared_memory_bank = []
        self.max_history_frames = 10
        self._active_traffic_manager_port = self.traffic_manager_port

    def _collect_compact_spawn_plan(self, route, scenario):
        base_idx = max(
            0,
            min(
                int(getattr(scenario, "ego_spawn_route_index", 0)),
                max(0, len(route.waypoints) - 1),
            ),
        )
        base_transform = self._route_transform_at_index(route, base_idx)
        base_waypoint = self._snap_location_to_driving_waypoint(base_transform.location)
        if base_waypoint is None:
            return None

        def _same_direction(candidate_wp):
            if candidate_wp is None:
                return False
            if candidate_wp.lane_type != carla.LaneType.Driving:
                return False
            yaw_delta = abs((candidate_wp.transform.rotation.yaw - base_waypoint.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
            return yaw_delta <= 35.0

        candidates = []

        def _append_candidate(route_idx, wp, longitudinal_offset=0.0):
            if wp is None:
                return
            transform = carla.Transform(
                location=carla.Location(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=max(wp.transform.location.z, base_transform.location.z),
                ),
                rotation=wp.transform.rotation,
            )
            key = (
                round(transform.location.x, 1),
                round(transform.location.y, 1),
                round(transform.rotation.yaw, 1),
            )
            if any(existing_key == key for _, _, existing_key in candidates):
                return
            candidates.append((route_idx, transform, key))

        _append_candidate(base_idx, base_waypoint)

        for neighbor_fn in (base_waypoint.get_left_lane, base_waypoint.get_right_lane):
            neighbor_wp = neighbor_fn()
            if _same_direction(neighbor_wp):
                _append_candidate(base_idx, neighbor_wp)

        for distance in (8.0, 14.0, 20.0):
            previous_wps = base_waypoint.previous(distance)
            if previous_wps:
                prev_wp = previous_wps[0]
                if _same_direction(prev_wp):
                    _append_candidate(base_idx, prev_wp)
                for neighbor_fn in (prev_wp.get_left_lane, prev_wp.get_right_lane):
                    neighbor_wp = neighbor_fn()
                    if _same_direction(neighbor_wp):
                        _append_candidate(base_idx, neighbor_wp)

        if len(candidates) < self.ego_num:
            return None

        return [(route_idx, transform) for route_idx, transform, _ in candidates[: self.ego_num]]

    def _spawn_ego_vehicles(self, route, scenario):
        self.vehicles = []
        spawn_plan = None
        if getattr(scenario, "ego_spawn_mode", "route_spacing") == "compact":
            spawn_plan = self._collect_compact_spawn_plan(route, scenario)

        if spawn_plan is None:
            spawn_indices = self._resolve_spawn_indices(route, getattr(scenario, "ego_route_spacing", 1), self.ego_num)
            spawn_plan = [(route_idx, self._route_transform_at_index(route, route_idx)) for route_idx in spawn_indices]
        else:
            spawn_indices = [route_idx for route_idx, _ in spawn_plan]

        for agent_idx, (route_idx, spawn_point) in enumerate(spawn_plan):
            vehicle, actual_transform, used_fallback = self._spawn_vehicle_with_retries(spawn_point)
            self.vehicles.append(vehicle)
            logger.info(
                "Spawned ego_%d at route waypoint %d -> (%.1f, %.1f, %.1f | yaw=%.1f)%s",
                agent_idx,
                route_idx,
                actual_transform.location.x,
                actual_transform.location.y,
                actual_transform.location.z,
                actual_transform.rotation.yaw,
                " [fallback]" if used_fallback else "",
            )

        return spawn_indices

    def _setup_agents(self):
        self.agents = []
        for idx, vehicle in enumerate(self.vehicles):
            agent = MultiViewLangCoopAgent(agent_config_path=self.agent_config)
            agent.agent_idx = idx
            agent.setup(vehicle)
            self.agents.append(agent)
            logger.info("Initialized collaborative multiview agent_%d", idx)

    def _assign_route_to_agents(self, route, spawn_indices):
        for agent_idx, agent in enumerate(self.agents):
            start_idx = spawn_indices[agent_idx]
            for waypoint_data in route.waypoints[start_idx + 1:]:
                wp_location = carla.Location(
                    x=waypoint_data.location["x"],
                    y=waypoint_data.location["y"],
                    z=waypoint_data.location["z"],
                )
                waypoint = self._snap_location_to_driving_waypoint(wp_location)
                if waypoint:
                    agent.set_target_waypoint(waypoint)

    def _setup_background_actors(self, scenario):
        """Spawn autopilot vehicles and walkers to better match leaderboard-style runs."""
        traffic_manager = None
        candidate_ports = [self.traffic_manager_port] + [
            self.traffic_manager_port + offset for offset in range(1, 6)
        ]
        last_error = None
        for candidate_port in candidate_ports:
            try:
                traffic_manager = self.client.get_trafficmanager(candidate_port)
                self._active_traffic_manager_port = candidate_port
                if candidate_port != self.traffic_manager_port:
                    logger.warning(
                        "Traffic Manager port %d was busy; using fallback port %d",
                        self.traffic_manager_port,
                        candidate_port,
                    )
                break
            except RuntimeError as exc:
                last_error = exc

        if traffic_manager is None:
            raise RuntimeError(
                f"Failed to acquire Traffic Manager on ports {candidate_ports}: {last_error}"
            )

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(10.0)

        ego_locations = [vehicle.get_location() for vehicle in self.vehicles]
        focus_near_egos = bool(getattr(scenario, "focus_background_near_egos", False))
        background_radius = float(getattr(scenario, "background_spawn_radius", 60.0))
        ego_center = None
        if ego_locations:
            ego_center = carla.Location(
                x=float(np.mean([loc.x for loc in ego_locations])),
                y=float(np.mean([loc.y for loc in ego_locations])),
                z=float(np.mean([loc.z for loc in ego_locations])),
            )
        spawn_points = self.world.get_map().get_spawn_points()
        if focus_near_egos and ego_center is not None:
            focused_spawn_points = [
                spawn_point
                for spawn_point in spawn_points
                if spawn_point.location.distance(ego_center) <= background_radius
            ]
            if focused_spawn_points:
                spawn_points = sorted(
                    focused_spawn_points,
                    key=lambda sp: sp.location.distance(ego_center),
                )
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter("vehicle.*")

        background_vehicle_count = int(getattr(scenario, "background_vehicle_count", 0) or 0)
        spawned_vehicle_count = 0
        for spawn_point in spawn_points:
            if spawned_vehicle_count >= background_vehicle_count:
                break
            too_close = any(spawn_point.location.distance(loc) < 12.0 for loc in ego_locations)
            if too_close:
                continue
            blueprint = np.random.choice(vehicle_blueprints)
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", np.random.choice(colors))
            if blueprint.has_attribute("driver_id"):
                drivers = blueprint.get_attribute("driver_id").recommended_values
                if drivers:
                    blueprint.set_attribute("driver_id", np.random.choice(drivers))

            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is None:
                continue
            vehicle.set_autopilot(True, self._active_traffic_manager_port)
            self.background_vehicles.append(vehicle)
            spawned_vehicle_count += 1

        walker_spawn_points = []
        background_walker_count = int(getattr(scenario, "background_walker_count", 0) or 0)
        attempts = 0
        max_attempts = max(background_walker_count * 12, 24)
        while len(walker_spawn_points) < background_walker_count and attempts < max_attempts:
            attempts += 1
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            if focus_near_egos and ego_center is not None and location.distance(ego_center) > background_radius:
                continue
            walker_spawn_points.append(carla.Transform(location))

        walker_blueprints = blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = blueprint_library.find("controller.ai.walker")
        batch = []
        for spawn_point in walker_spawn_points:
            walker_bp = np.random.choice(walker_blueprints)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        walker_results = self.client.apply_batch_sync(batch, True) if batch else []
        walker_ids = [result.actor_id for result in walker_results if not result.error]
        self.walker_actors = self.world.get_actors(walker_ids)

        controller_batch = [
            carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker.id)
            for walker in self.walker_actors
        ]
        controller_results = self.client.apply_batch_sync(controller_batch, True) if controller_batch else []
        controller_ids = [result.actor_id for result in controller_results if not result.error]
        self.walker_controller_actors = self.world.get_actors(controller_ids)

        for controller in self.walker_controller_actors:
            controller.start()
            destination = self.world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(1.2 + float(np.random.rand()))

        logger.info(
            "Spawned background actors: %d vehicles, %d walkers",
            len(self.background_vehicles),
            len(self.walker_actors),
        )

    def _update_shared_memory_bank(self):
        frame_timestamp = None
        detmap_pose = []
        ego_yaw = []
        front_images = []
        targets = []
        ego_speed = []

        for agent in self.agents:
            frame_data = agent.build_frame_data()
            if frame_data is None:
                return False
            frame_timestamp = frame_data["timestamp"]
            detmap_pose.append(frame_data["detmap_pose"][0])
            ego_yaw.append(frame_data["ego_yaw"][0])
            front_images.append(frame_data["front_image"])
            targets.append(frame_data["target"][0])
            ego_speed.append(frame_data["ego_speed"][0])

        shared_frame = {
            "timestamp": frame_timestamp,
            "detmap_pose": detmap_pose,
            "ego_yaw": ego_yaw,
            "front_image": front_images,
            "target": targets,
            "ego_speed": ego_speed,
        }
        self.shared_memory_bank.append(shared_frame)
        if len(self.shared_memory_bank) > self.max_history_frames:
            self.shared_memory_bank.pop(0)

        for agent in self.agents:
            agent.perception_memory_bank = list(self.shared_memory_bank)
            agent.frame_count = len(self.shared_memory_bank)
        return True

    def _compose_multiview_panel(self, agent, metrics_snapshot: dict | None):
        view_order = [
            ("front_camera", "Front"),
            ("left_camera", "Left"),
            ("right_camera", "Right"),
            ("rear_camera", "Rear"),
            ("bev_camera", "BEV"),
        ]

        available = []
        for key, label in view_order:
            image = agent.sensor_data.get(key)
            if image is None:
                continue
            available.append((label, Image.fromarray(image.astype(np.uint8))))

        if not available:
            return None

        cell_w = 480
        cell_h = 320
        cols = 2
        rows = (len(available) + cols - 1) // cols
        header_h = 126 if metrics_snapshot else 24
        panel = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), (18, 22, 30))
        draw = ImageDraw.Draw(panel, "RGBA")
        font = ImageFont.load_default()

        if metrics_snapshot:
            header_lines = [
                f"Step {metrics_snapshot['step']} | Sim {metrics_snapshot['simulation_progress_pct']:.1f}% | RC {metrics_snapshot['rc']:.1f}% | DS {metrics_snapshot['ds']:.1f}",
                f"Speed {metrics_snapshot['speed_mps']:.1f} m/s | Target {metrics_snapshot['target_speed_mps']:.1f} m/s | Steer {metrics_snapshot['steer']:+.3f} ({metrics_snapshot['steer_deg']:+.1f} deg)",
                f"Throttle {metrics_snapshot['throttle']:.2f} | Brake {metrics_snapshot['brake']:.2f} | Collisions {metrics_snapshot['collisions']} | Violations {metrics_snapshot['violations']}",
                f"Route {metrics_snapshot['route_id']} | static={metrics_snapshot.get('static_hazard_detected', False)} proximity={metrics_snapshot.get('hazard_proximity_detected', False)} guard={metrics_snapshot.get('static_guard_applied', False)}",
                f"Background actors | vehicles={metrics_snapshot.get('background_vehicle_count', 0)} walkers={metrics_snapshot.get('background_walker_count', 0)}",
            ]
            draw.rounded_rectangle([(10, 10), (panel.width - 10, header_h - 10)], radius=12, fill=(0, 0, 0, 170))
            y = 18
            for line in header_lines:
                draw.text((20, y), line, fill=(255, 255, 255, 255), font=font)
                y += 16

        for idx, (label, image) in enumerate(available):
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
        scenario_img_dir = self.images_dir / scenario_id / f"agent_{agent_idx}"
        scenario_img_dir.mkdir(parents=True, exist_ok=True)

        frame_payload = dict(metrics_snapshot)
        frame_payload["scenario_id"] = scenario_id
        frame_payload["agent_idx"] = int(agent_idx)
        frame_payload["frame_step"] = int(step)
        frame_payload["image_file"] = image_path.name

        sidecar_path = scenario_img_dir / f"frame_{step:06d}.debug.json"
        with sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(frame_payload, handle, indent=2)

        trace_path = scenario_img_dir / "debug_trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(frame_payload, ensure_ascii=True) + "\n")

    def _save_multiview_image(self, scenario_id: str, agent_idx: int, step: int, agent, metrics_snapshot: dict | None = None):
        try:
            scenario_img_dir = self.images_dir / scenario_id / f"agent_{agent_idx}"
            scenario_img_dir.mkdir(parents=True, exist_ok=True)
            img_path = scenario_img_dir / f"frame_{step:06d}.jpg"
            panel = self._compose_multiview_panel(agent, metrics_snapshot)
            if panel is None:
                return
            panel.save(img_path, quality=85)
            if metrics_snapshot:
                self._write_frame_debug_log(scenario_id, agent_idx, step, img_path, metrics_snapshot)
        except Exception as exc:
            logger.debug("Failed to save multiview frame for agent_%d step %d: %s", agent_idx, step, exc)

    def _write_scenario_debug_summary(self, scenario_id: str, agent_idx: int, final_snapshot: dict, hazard_debug_stats: dict):
        scenario_img_dir = self.images_dir / scenario_id / f"agent_{agent_idx}"
        scenario_img_dir.mkdir(parents=True, exist_ok=True)

        sampled = max(int(hazard_debug_stats.get("sampled_frames", 0)), 1)
        summary_payload = {
            "scenario_id": scenario_id,
            "agent_idx": int(agent_idx),
            "route_id": final_snapshot.get("route_id", ""),
            "final_route_completion_pct": float(final_snapshot.get("rc", 0.0)),
            "final_driving_score": float(final_snapshot.get("ds", 0.0)),
            "final_collisions": int(final_snapshot.get("collisions", 0)),
            "hazard_debug_stats": dict(hazard_debug_stats),
            "hazard_recall_indicators": {
                "static_hazard_detection_rate": float(hazard_debug_stats.get("sampled_static_hazard_frames", 0)) / sampled,
                "proximity_detection_rate": float(hazard_debug_stats.get("sampled_proximity_frames", 0)) / sampled,
                "guard_application_rate": float(hazard_debug_stats.get("sampled_guard_applied_frames", 0)) / sampled,
                "wall_barrier_mention_rate": float(hazard_debug_stats.get("sampled_wall_barrier_mentions", 0)) / sampled,
            },
            "latest_scene_excerpt": str(final_snapshot.get("planner_scene_excerpt", "")),
            "latest_objects_excerpt": str(final_snapshot.get("planner_objects_excerpt", "")),
            "latest_intent_excerpt": str(final_snapshot.get("planner_intent_excerpt", "")),
        }

        summary_path = scenario_img_dir / "hazard_debug_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)

    def _cleanup_background_actors(self):
        for controller in self.walker_controller_actors:
            try:
                controller.stop()
            except Exception:
                pass
        for actor in list(self.walker_controller_actors) + list(self.walker_actors) + list(self.background_vehicles):
            self._safe_destroy(actor)
        self.walker_controller_actors = []
        self.walker_actors = []
        self.background_vehicles = []

    def run_scenario(self, scenario, route, max_steps: int = 500, skip_frames: int = 4):
        logger.info("Starting leaderboard-style collaborative scenario: %s", scenario.scenario_id)
        route_distance = route.compute_distance()

        self.shared_memory_bank = []
        self.setup_environment(scenario)
        spawn_indices = self._spawn_ego_vehicles(route, scenario)
        self._setup_agents()
        self._assign_route_to_agents(route, spawn_indices)
        self._setup_background_actors(scenario)

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
                "sampled_frames": 0,
                "sampled_static_hazard_frames": 0,
                "sampled_proximity_frames": 0,
                "sampled_guard_applied_frames": 0,
                "sampled_wall_barrier_mentions": 0,
                "saved_debug_frames": 0,
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
                memory_ready = self._update_shared_memory_bank()

                if step % skip_frames == 0 and memory_ready:
                    if len(self.shared_memory_bank) >= 2 and self.agents and self.agents[0].vlm_planner:
                        model_config = {
                            "planning": {
                                "prompt_template": self.agents[0].config.get("planning", {}).get("prompt_template", {}),
                                "prompt_usage": self.agents[0].config.get("planning", {}).get("prompt_usage", {}),
                            },
                            "collab": self.agents[0].config.get("collab", {"sharing_modalities": ["image", "intent"]}),
                        }
                        try:
                            planned_routes = self.agents[0].vlm_planner.forward_collaborative(
                                self.shared_memory_bank,
                                model_config,
                            )
                        except Exception as exc:
                            logger.warning("Collaborative VLM planning failed: %s", exc)
                            planned_routes = None
                    else:
                        planned_routes = None

                    for idx, (agent, vehicle) in enumerate(zip(self.agents, self.vehicles)):
                        if planned_routes and idx < len(planned_routes):
                            planned_route = planned_routes[idx]
                        else:
                            planned_route = {
                                "target_speed": [agent.target_speed],
                                "curvature": [0.0],
                                "dt": 0.5,
                            }
                        control = agent.compute_control_from_plan(planned_route)
                        vehicle.apply_control(control)
                        last_controls[idx] = control

                        target_speed_arr = planned_route.get("target_speed", [0.0])
                        if isinstance(target_speed_arr, list) and target_speed_arr:
                            last_target_speeds[idx] = float(target_speed_arr[0])
                        else:
                            last_target_speeds[idx] = float(target_speed_arr)
                        last_planner_debugs[idx] = self._extract_planner_debug(planned_route)

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
                            "speed_violation",
                            description=f"speed={speed:.2f}",
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
                            gt_dynamic_perception=self._collect_dynamic_ground_truth(vehicle),
                        )
                        snapshot["background_vehicle_count"] = len(self.background_vehicles)
                        snapshot["background_walker_count"] = len(self.walker_actors)
                        latest_snapshots[idx] = snapshot

                        hazard_debug_stats[idx]["sampled_frames"] += 1
                        if snapshot.get("static_hazard_detected", False):
                            hazard_debug_stats[idx]["sampled_static_hazard_frames"] += 1
                        if snapshot.get("hazard_proximity_detected", False):
                            hazard_debug_stats[idx]["sampled_proximity_frames"] += 1
                        if snapshot.get("static_guard_applied", False):
                            hazard_debug_stats[idx]["sampled_guard_applied_frames"] += 1
                        if snapshot.get("wall_barrier_mentioned", False):
                            hazard_debug_stats[idx]["sampled_wall_barrier_mentions"] += 1

                        live_tracers[idx].update(snapshot)

                    if step % 10 == 0:
                        self._save_multiview_image(
                            scenario.scenario_id,
                            idx,
                            step,
                            self.agents[idx],
                            metrics_snapshot=latest_snapshots[idx],
                        )
                        hazard_debug_stats[idx]["saved_debug_frames"] += 1

                if step % 100 == 0:
                    progress = [
                        (completed_distances[idx] / route_distance) * 100 if route_distance > 0 else 0.0
                        for idx in range(len(self.vehicles))
                    ]
                    logger.info(
                        "Step %d/%d | Progress: %s",
                        step,
                        max_steps,
                        ", ".join(f"ego_{i}={progress_i:.1f}%" for i, progress_i in enumerate(progress)),
                    )
        finally:
            for sensor in collision_sensors:
                self._safe_stop_and_destroy(sensor)
            for agent in self.agents:
                agent.destroy()
            for vehicle in self.vehicles:
                self._safe_destroy(vehicle)
            self._cleanup_background_actors()

        elapsed_time = time.time() - start_time
        for idx, route_metric_id in enumerate(route_metric_ids):
            self.metrics_calculator.set_route_completion(
                route_metric_id,
                completed_distances[idx],
                route_distance,
                elapsed_time,
            )
            if speed_samples[idx]:
                self.metrics_calculator.set_speed_metrics(
                    route_metric_id,
                    float(np.mean(speed_samples[idx])),
                    float(np.max(speed_samples[idx])),
                )
            final_ds = self.metrics_calculator.calculate_driving_score(route_metric_id)
            final_route_metrics = self.metrics_calculator.get_route_metrics(route_metric_id)
            final_violations = final_route_metrics.lane_departures + final_route_metrics.speed_violations
            final_speed = speed_samples[idx][-1] if speed_samples[idx] else 0.0

            final_snapshot = {
                "step": int(max_steps),
                "simulation_progress_pct": 100.0,
                "elapsed_wall_time_sec": float(elapsed_time),
                "distance_m": float(completed_distances[idx]),
                "route_distance_m": float(route_distance),
                "rs": float(final_route_metrics.completion_percentage),
                "rc": float(final_route_metrics.completion_percentage),
                "ds": float(final_ds),
                "speed_mps": float(final_speed),
                "target_speed_mps": float(last_target_speeds[idx]),
                "steer": float(last_controls[idx].steer),
                "steer_deg": float(last_controls[idx].steer * 70.0),
                "throttle": float(last_controls[idx].throttle),
                "brake": float(last_controls[idx].brake),
                "collisions": int(final_route_metrics.collisions),
                "violations": int(final_violations),
                "route_id": route_metric_id,
                "static_hazard_detected": bool(last_planner_debugs[idx].get("hazard_signals", {}).get("has_static_hazard", False)),
                "hazard_proximity_detected": bool(last_planner_debugs[idx].get("hazard_signals", {}).get("has_proximity_cue", False)),
                "static_guard_applied": bool(last_planner_debugs[idx].get("static_guard_applied", False)),
                "wall_barrier_mentioned": bool(last_planner_debugs[idx].get("wall_barrier_mentioned", False)),
                "hazard_side": str(last_planner_debugs[idx].get("hazard_signals", {}).get("hazard_side", "none")),
                "static_guard_reason": str(last_planner_debugs[idx].get("static_guard_reason", "")),
                "planner_scene_excerpt": str(last_planner_debugs[idx].get("scene_excerpt", "")),
                "planner_objects_excerpt": str(last_planner_debugs[idx].get("objects_excerpt", "")),
                "planner_intent_excerpt": str(last_planner_debugs[idx].get("intent_excerpt", "")),
                "hazard_debug_stats": dict(hazard_debug_stats[idx]),
                "background_vehicle_count": len(self.background_vehicles),
                "background_walker_count": len(self.walker_actors),
            }
            live_tracers[idx].update(final_snapshot)
            self._write_scenario_debug_summary(scenario.scenario_id, idx, final_snapshot, hazard_debug_stats[idx])

        metrics = self.metrics_calculator.get_summary()
        logger.info("Scenario complete: %s", scenario.scenario_id)
        for route_metric_id in route_metric_ids:
            route_metrics = metrics.get("routes", {}).get(route_metric_id, {})
            logger.info(
                "  %s | RC %.1f%% | DS %.1f | Collisions %s | Violations %s",
                route_metric_id,
                route_metrics.get("rc", 0.0),
                route_metrics.get("ds", 0.0),
                route_metrics.get("collisions", 0),
                route_metrics.get("violations", 0),
            )
        return metrics

    def run_tests(self, scenario_ids=None, max_steps: int = 500):
        if not self.connect_to_carla():
            return

        selected_ids = scenario_ids if scenario_ids else self.scenario_manager.list_scenarios()
        scenarios = []
        for scenario_id in selected_ids:
            scenario = self.scenario_manager.get_scenario(scenario_id)
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
        self.visualizer.export_json(output_dir=str(self.results_dir), filename="metrics.json")
        self.visualizer.save_detailed_report(output_dir=str(self.results_dir), filename="detailed_report.txt")
        self.visualizer.plot_summary(output_dir=str(self.results_dir))
        logger.info("Results saved in: %s", self.results_dir)


def main():
    parser = argparse.ArgumentParser(description="LangCoop Leaderboard-Style Collaborative Runner")
    parser.add_argument("--host", type=str, default="carla-rpc")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--agent-config", type=str, default="configs/langcoop_agent_config_32b.yaml")
    parser.add_argument("--scenario-ids", type=str, nargs="+", default=["town05_langcoop_collab_eval"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--results-dir", type=str, default="test_results_langcoop_leaderboard")
    parser.add_argument("--ego-num", type=int, default=2, help="Number of collaborative ego vehicles, typically 2 or 3")
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    args = parser.parse_args()

    runner = LeaderboardLangCoopRunner(
        carla_host=args.host,
        carla_port=args.port,
        agent_config=args.agent_config,
        results_dir=args.results_dir,
        ego_num=args.ego_num,
        traffic_manager_port=args.traffic_manager_port,
    )
    runner.run_tests(scenario_ids=args.scenario_ids, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
