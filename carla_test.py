# Minimal CARLA sanity script: load Town05, spawn 1 car, attach RGB cam, tick a few steps.
# Prereq: CARLA server running, e.g. ./CarlaUE4.sh -quality-level=Low -world-port=2000

import time
import queue
import carla

HOST = "localhost"
PORT = 2000
TIMEOUT_S = 10.0
TOWN = "Town05"

STEPS = 20
DT = 0.05  # fixed sim step (s)

CAM_W, CAM_H = 800, 600
CAM_FOV = 90

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)

    world = client.load_world(TOWN)
    original_settings = world.get_settings()

    vehicles = []
    sensors = []

    try:
        # Sync mode = deterministic ticks.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        # Spawn a vehicle.
        veh_bp = bp_lib.filter("vehicle.*model3*")
        veh_bp = veh_bp[0] if veh_bp else bp_lib.filter("vehicle.*")[0]
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points in map.")

        vehicle = world.spawn_actor(veh_bp, spawn_points[0])
        vehicles.append(vehicle)

        # Put it in autopilot (optional sanity).
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        # Attach RGB camera.
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_W))
        cam_bp.set_attribute("image_size_y", str(CAM_H))
        cam_bp.set_attribute("fov", str(CAM_FOV))

        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))  # hood-ish
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        sensors.append(camera)

        img_q: "queue.Queue[carla.Image]" = queue.Queue()
        camera.listen(img_q.put)

        # Warm up one tick so sensors start streaming.
        world.tick()

        for step in range(STEPS):
            frame = world.tick()  # advance sim by DT

            # Grab latest camera frame (blocking with timeout).
            try:
                img = img_q.get(timeout=2.0)
            except queue.Empty:
                print(f"[{step}] No camera image received.")
                continue

            # Read minimal state.
            tf = vehicle.get_transform()
            vel = vehicle.get_velocity()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            print(
                f"[{step:02d}] frame={frame} img_frame={img.frame} "
                f"pos=({tf.location.x:.1f},{tf.location.y:.1f}) "
                f"yaw={tf.rotation.yaw:.1f} speed={speed:.2f} m/s"
            )

            # Optional: save first couple images to verify camera works.
            if step < 3:
                img.save_to_disk(f"_debug_rgb_{step:02d}.png")

        print("Done. Saved _debug_rgb_00..02.png")

    finally:
        # Cleanup.
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass
            s.destroy()

        for v in vehicles:
            v.destroy()

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
