"""
Comprehensive scenario evaluator integrating metrics, scenarios, and visualization.
This is the main entry point for evaluation.
"""

import carla
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time

from ..agents import SimpleAutonomousAgent
from ..controllers import PIDController
from .metrics import MetricsCalculator
from .scenario_manager import ScenarioManager
from ..visualization import MetricsVisualizer

logger = logging.getLogger(__name__)


class ScenarioEvaluator:
    """
    Main evaluation framework that combines:
    - Scenario management (routes + environmental conditions)
    - Metrics calculation (RC, DS, collisions, etc)
    - Visualization (plots and reports)
    """
    
    def __init__(self, carla_host: str = 'carla-rpc', carla_port: int = 2000,
                 timeout: float = 60.0):
        """
        Initialize scenario evaluator.
        
        Args:
            carla_host: CARLA server host (default: 'carla-rpc')
            carla_port: CARLA server port
            timeout: Connection timeout
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.timeout = timeout
        
        self.client = None
        self.world = None
        
        self.scenario_manager = ScenarioManager()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = MetricsVisualizer()
        
        self.agents: List[SimpleAutonomousAgent] = []
        self.vehicles: List[carla.Actor] = []
        
        self._connect_to_carla()
    
    def _connect_to_carla(self):
        """Connect to CARLA server."""
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            logger.info(f"Connected to CARLA at {self.carla_host}:{self.carla_port}")
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            raise
    
    def load_map(self, map_name: str):
        """Load CARLA map."""
        try:
            self.world = self.client.load_world(map_name)
            logger.info(f"Loaded map: {map_name}")
        except Exception as e:
            logger.error(f"Failed to load map {map_name}: {e}")
            raise
    
    def setup_scenario(self, scenario):
        """
        Setup world according to scenario parameters.
        
        Args:
            scenario: Scenario object with weather/traffic parameters
        """
        # Set weather
        weather = carla.WeatherParameters()
        if scenario.weather:
            weather.cloudiness = scenario.weather.get('cloudiness', 30)
            weather.precipitation = scenario.weather.get('precipitation', 0)
            weather.wind_intensity = scenario.weather.get('wind_intensity', 0)
            weather.sun_altitude_angle = scenario.weather.get('sun_altitude_angle', 45)
        
        self.world.set_weather(weather)
        logger.info(f"Scenario setup: {scenario.scenario_id}")
    
    def evaluate_scenario(self, scenario_id: str, agent_config_path: str,
                         num_agents: int = 1, num_steps: int = 1000,
                         frame_skip: int = 1, visualize: bool = True) -> Dict:
        """
        Evaluate a single scenario.
        
        Args:
            scenario_id: ID of scenario to evaluate
            agent_config_path: Path to agent config YAML
            num_agents: Number of agents to spawn
            num_steps: Number of simulation steps
            frame_skip: Skip frames between control updates
            visualize: Enable visualization
            
        Returns:
            Dictionary of evaluation metrics
        """
        scenario = self.scenario_manager.get_scenario(scenario_id)
        if not scenario:
            logger.error(f"Scenario not found: {scenario_id}")
            return {}
        
        route = self.scenario_manager.get_route(scenario.route_id)
        if not route:
            logger.error(f"Route not found: {scenario.route_id}")
            return {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {scenario_id} on route {route.route_id}")
        logger.info(f"Route distance: {route.total_distance:.1f}m, Difficulty: {scenario.difficulty}")
        logger.info(f"{'='*60}")
        
        # Setup scenario
        self.setup_scenario(scenario)
        
        # Spawn agents
        self.agents.clear()
        self.vehicles.clear()
        
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("No spawn points in map")
            return {}
        
        for i in range(min(num_agents, len(spawn_points))):
            try:
                # Spawn vehicle
                blueprint = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
                vehicle = self.world.spawn_actor(blueprint, spawn_points[i])
                self.vehicles.append(vehicle)
                
                # Attach agent
                agent = SimpleAutonomousAgent(agent_config_path)
                agent.setup(vehicle)
                self.agents.append(agent)
                
                logger.info(f"Spawned agent {i+1}/{num_agents}")
            except Exception as e:
                logger.error(f"Failed to spawn agent {i}: {e}")
        
        if not self.agents:
            logger.error("No agents spawned")
            return {}
        
        # Initialize tracking
        distances_traveled = [0.0] * len(self.agents)
        speeds = [[] for _ in range(len(self.agents))]
        agent_collisions = [0] * len(self.agents)
        
        start_positions = [v.get_location() for v in self.vehicles]
        
        # Run simulation
        logger.info(f"Running simulation for {num_steps} steps...")
        start_time = time.time()
        
        try:
            for step in range(num_steps):
                for agent_id, (agent, vehicle) in enumerate(zip(self.agents, self.vehicles)):
                    # Get control
                    control = agent.step()
                    
                    # Apply control periodically
                    if step % frame_skip == 0:
                        vehicle.apply_control(control)
                    
                    # Collect metrics
                    velocity = vehicle.get_velocity()
                    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    speeds[agent_id].append(speed)
                    
                    # Distance traveled
                    curr_loc = vehicle.get_location()
                    start_loc = start_positions[agent_id]
                    dist = curr_loc.distance(start_loc)
                    distances_traveled[agent_id] = dist
                
                # Advance simulation
                self.world.tick()
                
                # Log progress
                if (step + 1) % 200 == 0:
                    avg_dist = np.mean(distances_traveled)
                    logger.info(f"  Step {step + 1}/{num_steps}, Avg distance: {avg_dist:.1f}m")
        
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics for each agent
        for agent_id in range(len(self.agents)):
            route_id = f"{scenario_id}_agent_{agent_id}"
            
            # Record distance
            self.metrics_calculator.set_route_completion(
                route_id,
                distances_traveled[agent_id],
                route.total_distance,
                elapsed_time
            )
            
            # Record speed metrics
            if speeds[agent_id]:
                avg_speed = np.mean(speeds[agent_id])
                max_speed = np.max(speeds[agent_id])
                self.metrics_calculator.set_speed_metrics(route_id, avg_speed, max_speed)
            
            # Record events (simplified - can be extended)
            # In real scenario, would detect actual collisions via CARLA API
            if agent_collisions[agent_id] > 0:
                self.metrics_calculator.record_event(
                    route_id, num_steps, 'collision',
                    description=f"Collision detected"
                )
            
            # Calculate driving score
            self.metrics_calculator.calculate_driving_score(route_id)
        
        # Cleanup
        self._cleanup_agents()
        
        logger.info(f"Scenario evaluation complete (took {elapsed_time:.1f}s)")
        
        return self.metrics_calculator.get_summary()
    
    def evaluate_multiple_scenarios(self, scenario_ids: List[str],
                                   agent_config_path: str,
                                   output_dir: str = 'results',
                                   **kwargs) -> Dict:
        """
        Evaluate multiple scenarios sequentially.
        
        Args:
            scenario_ids: List of scenario IDs to evaluate
            agent_config_path: Path to agent config
            output_dir: Output directory for results
            **kwargs: Additional arguments for evaluate_scenario
            
        Returns:
            Combined metrics summary
        """
        logger.info(f"Starting evaluation of {len(scenario_ids)} scenarios...")
        
        for scenario_id in scenario_ids:
            try:
                self.evaluate_scenario(scenario_id, agent_config_path, **kwargs)
            except Exception as e:
                logger.error(f"Failed to evaluate scenario {scenario_id}: {e}")
        
        # Get final summary
        summary = self.metrics_calculator.get_summary()
        self.visualizer.add_metrics(summary)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.visualizer.export_json(output_dir)
        self.visualizer.save_detailed_report(output_dir)
        self.visualizer.print_summary(summary)
        
        try:
            self.visualizer.plot_summary(output_dir)
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        
        return summary
    
    def _cleanup_agents(self):
        """Cleanup and destroy all actors."""
        for agent in self.agents:
            agent.destroy()
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        self.agents.clear()
        self.vehicles.clear()
    
    def cleanup(self):
        """Final cleanup."""
        self._cleanup_agents()
        logger.info("Cleanup complete")
