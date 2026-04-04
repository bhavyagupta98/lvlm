#!/usr/bin/env python3
"""
Debug analysis script to check:
1. If images are being captured properly
2. If VLM is receiving valid images  
3. If CARLA is receiving control commands
4. Synchronization between sensor updates and control commands
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_image_encoding():
    """Test image encoding/decoding pipeline."""
    logger.info("=" * 60)
    logger.info("CHECKING IMAGE ENCODING PIPELINE")
    logger.info("=" * 60)
    
    import numpy as np
    from test_runner.vlm.vlm_planner_speed_curvature import VLMPlannerSpeedCurvature
    
    # Create a dummy VLM planner instance
    try:
        vlm = VLMPlannerSpeedCurvature(
            api_model_name='Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
            api_base_url='http://localhost:8000/v1',
            api_key='EMPTY'
        )
        logger.info("✓ VLM Planner initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize VLM: {e}")
        return False
    
    # Create test image
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    logger.info(f"Created test image: shape={test_image.shape}, dtype={test_image.dtype}")
    
    # Test encoding
    try:
        encoded = vlm._encode_image(test_image)
        logger.info(f"✓ Image encoded successfully, base64 length: {len(encoded)}")
        logger.info(f"  First 50 chars: {encoded[:50]}...")
        return True
    except Exception as e:
        logger.error(f"✗ Image encoding failed: {e}")
        return False

def check_carla_connection():
    """Test CARLA server connection and control."""
    logger.info("=" * 60)
    logger.info("CHECKING CARLA SERVER CONNECTION")
    logger.info("=" * 60)
    
    import carla
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        server_version = client.get_server_version()
        logger.info(f"✓ Connected to CARLA: {server_version}")
        
        # Check if synchronous mode is enabled
        settings = world.get_settings()
        logger.info(f"  Synchronous mode: {settings.synchronous_mode}")
        logger.info(f"  Fixed delta seconds: {settings.fixed_delta_seconds}")
        
        return True
    except Exception as e:
        logger.error(f"✗ CARLA connection failed: {e}")
        logger.error("  Make sure CARLA server is running on localhost:2000")
        return False

def check_sensor_data_flow():
    """Test sensor data flow in a spawned vehicle."""
    logger.info("=" * 60)
    logger.info("CHECKING SENSOR DATA FLOW")
    logger.info("=" * 60)
    
    import carla
    import time
    import numpy as np
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Load map
        world = client.load_world('Town05')
        time.sleep(2)
        
        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("✗ No spawn points available in Town05")
            return False
        
        spawn_point = spawn_points[0]
        logger.info(f"Using spawn point: {spawn_point.location}")
        
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"✓ Vehicle spawned: {vehicle.type_id}")
        
        # Setup camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        
        camera_transform = carla.Transform(
            location=carla.Location(x=1.3, y=0.0, z=2.3)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Capture image
        image_captured = False
        captured_image = None
        
        def on_image(image):
            nonlocal image_captured, captured_image
            image_array = np.array(image.raw_data).reshape((image.height, image.width, 4))
            captured_image = image_array[:, :, :3]
            image_captured = True
            logger.debug(f"  Camera callback: frame={image.frame}, shape={captured_image.shape}")
        
        camera.listen(on_image)
        
        # Tick world a few times
        for i in range(5):
            world.tick()
            time.sleep(0.05)
            logger.debug(f"  Tick {i}: image_captured={image_captured}")
        
        if image_captured and captured_image is not None:
            logger.info(f"✓ Image captured: shape={captured_image.shape}, "
                       f"min={captured_image.min()}, max={captured_image.max()}, "
                       f"mean={captured_image.mean():.1f}")
        else:
            logger.error("✗ No image captured after 5 ticks")
            return False
        
        # Test control command
        control = carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0)
        vehicle.apply_control(control)
        logger.info(f"✓ Control applied: throttle=0.5")
        
        # Check velocity after control
        world.tick()
        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        logger.info(f"  Vehicle speed after control: {speed:.3f} m/s")
        
        # Cleanup
        camera.destroy()
        vehicle.destroy()
        logger.info("✓ Cleanup complete")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Sensor data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_vlm_api_connection():
    """Test vLLM API connection."""
    logger.info("=" * 60)
    logger.info("CHECKING VLLM API CONNECTION")
    logger.info("=" * 60)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key='EMPTY', base_url='http://localhost:8000/v1')
        
        # Try to list models
        models = client.models.list()
        logger.info(f"✓ Connected to vLLM API")
        
        # Get model details
        model_list = [m.id for m in models.data]
        logger.info(f"  Available models: {model_list}")
        
        if not model_list:
            logger.warning("  ⚠ No models loaded on vLLM server!")
        
        # Check if expected model is there
        if 'qwen2.5-vl-7b-instruct-awq' not in ' '.join(model_list).lower():
            logger.warning("  ⚠ Expected 7B model not found!")
            logger.warning("  Make sure vLLM is serving: Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        else:
            logger.info("  ✓ 7B model is loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ vLLM API connection failed: {e}")
        logger.error("  Make sure vLLM is running on http://localhost:8000/v1")
        return False

def main():
    logger.info("\n" + "=" * 60)
    logger.info("LANGCOOP DEBUG ANALYSIS")
    logger.info("=" * 60 + "\n")
    
    checks = [
        ("Image Encoding", check_image_encoding),
        ("CARLA Connection", check_carla_connection),
        ("vLLM API", check_vlm_api_connection),
        ("Sensor Data Flow", check_sensor_data_flow),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"CRITICAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
        
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    all_pass = all(results.values())
    logger.info("")
    if all_pass:
        logger.info("✓ All checks passed!")
    else:
        logger.error("✗ Some checks failed. See details above.")
    
    return all_pass

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
