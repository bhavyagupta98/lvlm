"""
Main test runner for CARLA simulations.
Handles simulation setup, execution, and metrics collection.
"""

import carla
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from .agents import SimpleAutonomousAgent

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Orchestrates CARLA test simulations.
    Creates agents, manages simulation, collects metrics.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 2000, timeout: float = 60.0):
        """
        Initialize test runner.
        
        Args:
            host: CARLA server host
            port: CARLA server port
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.client = None
        self.world = None
        self.agents: List[SimpleAutonomousAgent] = []
        self.vehicles: List[carla.Actor] = []
        
        self.metrics = {
            'num_steps': 0,
            'total_distance': 0.0,
            'collisions': 0,
            'start_time': None,
            'end_time': None,
            'agent_data': {}
        }
        
        self._connect_to_carla()
    
    def _connect_to_carla(self):
        """Connect to CARLA server."""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            logger.info(f"Connected to CARLA server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            raise
    
    def load_map(self, map_name: str = 'Town05'):
        """
        Load CARLA town map.
        
        Args:
            map_name: Name of map (e.g., 'Town05', 'Town10')
        """
        try:
            self.world = self.client.load_world(map_name)
            logger.info(f"Loaded map: {map_name}")
        except Exception as e:
            logger.error(f"Failed to load map {map_name}: {e}")
            raise
    
    def setup_weather(self, weather_preset: str = 'ClearNoon'):
        """
        Set world weather.
        
        Args:
            weather_preset: Weather preset name
        """
        weather = getattr(carla.WeatherParameters, weather_preset)
        self.world.set_weather(weather)
        logger.info(f"Set weather to: {weather_preset}")
    
    def spawn_vehicle(self, agent_config_path: str, 
                     start_location: Optional[carla.Location] = None) -> SimpleAutonomousAgent:
        """
        Spawn a vehicle and attach an agent.
        
        Args:
            agent_config_path: Path to agent config file
            start_location: Starting location (random if None)
            
        Returns:
            Initialized SimpleAutonomousAgent
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        if start_location is None:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in map")
            spawn_point = spawn_points[0]
        else:
            spawn_point = carla.Transform(location=start_location)
        
        try:
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.vehicles.append(vehicle)
            logger.info(f"Spawned vehicle at {spawn_point.location}")
            
            # Create and setup agent
            agent = SimpleAutonomousAgent(agent_config_path)
            agent.setup(vehicle)
            self.agents.append(agent)
            
            # Initialize metrics for this agent
            agent_id = len(self.agents) - 1
            self.metrics['agent_data'][agent_id] = {
                'distance': 0.0,
                'collisions': 0,
                'frames': 0
            }
            
            return agent
        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            raise
    
    def run_simulation(self, num_steps: int = 1000, frame_skip: int = 1):
        """
        Run simulation for specified number of steps.
        
        Args:
            num_steps: Number of simulation steps
            frame_skip: Number of frames to skip between control updates
        """
        if not self.agents or not self.vehicles:
            raise RuntimeError("No agents spawned. Call spawn_vehicle() first.")
        
        self.metrics['start_time'] = datetime.now().isoformat()
        logger.info(f"Starting simulation for {num_steps} steps with {len(self.agents)} agents")
        
        try:
            for step in range(num_steps):
                # Get spectator view
                spectator = self.world.get_spectator()
                
                # Process each agent
                for agent_id, (agent, vehicle) in enumerate(zip(self.agents, self.vehicles)):
                    # Compute control
                    control = agent.step()
                    
                    # Apply control every frame_skip frames
                    if step % frame_skip == 0:
                        vehicle.apply_control(control)
                    
                    # Update spectator to follow first vehicle
                    if agent_id == 0:
                        transform = vehicle.get_transform()
                        spectator.set_transform(
                            carla.Transform(
                                location=transform.location + carla.Location(x=-10, z=5),
                                rotation=carla.Rotation(pitch=-30, yaw=transform.rotation.yaw)
                            )
                        )
                    
                    # Update metrics
                    velocity = vehicle.get_velocity()
                    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    self.metrics['agent_data'][agent_id]['distance'] += speed * 0.05  # ~50ms per frame
                    self.metrics['agent_data'][agent_id]['frames'] += 1
                
                # Advance simulation
                self.world.tick()
                
                # Log progress
                if (step + 1) % 100 == 0:
                    distances = [d['distance'] for d in self.metrics['agent_data'].values()]
                    avg_distance = np.mean(distances)
                    logger.info(f"Step {step + 1}/{num_steps}, Avg Distance: {avg_distance:.2f}m")
                
                self.metrics['num_steps'] = step + 1
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.metrics['end_time'] = datetime.now().isoformat()
    
    def save_metrics(self, output_path: str = 'results/metrics.json'):
        """Save collected metrics to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def cleanup(self):
        """Cleanup and destroy all actors."""
        logger.info("Cleaning up simulation...")
        
        for agent in self.agents:
            agent.destroy()
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        self.agents.clear()
        self.vehicles.clear()
        logger.info("Cleanup complete")
    
    def print_metrics(self):
        """Print collected metrics."""
        print("\n" + "="*50)
        print("SIMULATION METRICS")
        print("="*50)
        print(f"Total Steps: {self.metrics['num_steps']}")
        print(f"Start Time: {self.metrics['start_time']}")
        print(f"End Time: {self.metrics['end_time']}")
        print("\nPer-Agent Metrics:")
        for agent_id, data in self.metrics['agent_data'].items():
            print(f"  Agent {agent_id}:")
            print(f"    Distance: {data['distance']:.2f}m")
            print(f"    Frames: {data['frames']}")
            print(f"    Collisions: {data['collisions']}")
        print("="*50 + "\n")
