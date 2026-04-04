import time
import carla



HOST = "carla-rpc"
PORT = 2000
RPC_TIMEOUT = 2.0
MAX_WAIT = 180.0


def wait_for_server():
    """
    Wait until CARLA server becomes available.
    """
    client = carla.Client(HOST, PORT)
    client.set_timeout(RPC_TIMEOUT)

    start = time.time()
    last_err = None

    while time.time() - start < MAX_WAIT:
        try:
            world = client.get_world()
            return client, world
        except Exception as e:
            last_err = e
            print(f"[wait] CARLA not ready yet: {e}")
            time.sleep(2)

    raise RuntimeError(
        f"CARLA server not reachable after {MAX_WAIT} seconds. "
        f"Last error: {last_err}"
    )


def main():
    client, world = wait_for_server()

    print("[ok] Connected to CARLA")
    print("[ok] Current map:", world.get_map().name)


if __name__ == "__main__":
    main()