#!/usr/bin/env python3

import carla
import json
import math
import random
import time
from pathlib import Path

# ============================================================
# CONFIGURATION (EDIT HERE)
# ============================================================

HOST = "carla-rpc"     # Kubernetes service name
PORT = 2000
TM_PORT = 8000

TICKS = 2000
FIXED_DT = 0.05
SEED = 7

SAVE_DIR = "sim_output"
SAVE_IMAGES = False     # Set True if you want PNG frames

EGO1_MODEL = "vehicle.tesla.model3"
EGO2_MODEL = "vehicle.audi.a2"

# ============================================================


def speed_mps(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def main():

    random.seed(SEED)

    output_dir = Path(SAVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_file = output_dir / "ticks.jsonl"
    collisions_file = output_dir / "collisions.jsonl"

    print("Connecting to CARLA...")
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    print("Loading Town05...")
    world = client.load_world("Town05")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(SEED)

    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()

    actors = []
    sensors = []
    collisions = []

    def collision_callback(event):
        collisions.append({
            "frame": event.frame,
            "other_actor": event.other_actor.type_id if event.other_actor else None
        })

    try:
        spawn_points = carla_map.get_spawn_points()
        sp1, sp2 = random.sample(spawn_points, 2)

        v1 = world.try_spawn_actor(bp_lib.find(EGO1_MODEL), sp1)
        v2 = world.try_spawn_actor(bp_lib.find(EGO2_MODEL), sp2)

        actors.extend([v1, v2])

        v1.set_autopilot(True, TM_PORT)
        v2.set_autopilot(True, TM_PORT)

        # Collision sensors
        col_bp = bp_lib.find("sensor.other.collision")
        col1 = world.spawn_actor(col_bp, carla.Transform(), attach_to=v1)
        col2 = world.spawn_actor(col_bp, carla.Transform(), attach_to=v2)

        col1.listen(collision_callback)
        col2.listen(collision_callback)

        sensors.extend([col1, col2])

        print("Starting simulation loop...")

        with logs_file.open("w") as f:

            for i in range(TICKS):

                frame = world.tick()
                snapshot = world.get_snapshot()

                def vehicle_state(v):
                    tr = v.get_transform()
                    vel = v.get_velocity()
                    ctrl = v.get_control()

                    return {
                        "id": v.id,
                        "x": tr.location.x,
                        "y": tr.location.y,
                        "yaw": tr.rotation.yaw,
                        "speed_mps": speed_mps(vel),
                        "throttle": ctrl.throttle,
                        "steer": ctrl.steer,
                        "brake": ctrl.brake,
                    }

                record = {
                    "frame": frame,
                    "sim_time": snapshot.timestamp.elapsed_seconds,
                    "vehicles": [
                        vehicle_state(v1),
                        vehicle_state(v2)
                    ]
                }

                f.write(json.dumps(record) + "\n")

                if i % 50 == 0:
                    print(f"Frame {frame} | time={snapshot.timestamp.elapsed_seconds:.2f}")

        with collisions_file.open("w") as f:
            for c in collisions:
                f.write(json.dumps(c) + "\n")

        print("Simulation complete.")

    finally:
        for s in sensors:
            s.destroy()
        for a in actors:
            a.destroy()

        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)

        print("Cleaned up actors.")


if __name__ == "__main__":
    main()