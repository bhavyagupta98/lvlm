"""Scenario and route management for testing framework."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import carla


@dataclass
class RoutePoint:
    """Single point on a route."""
    location: Dict[str, float]  # {x, y, z}
    rotation: Dict[str, float]  # {pitch, yaw, roll}
    
    def to_transform(self) -> carla.Transform:
        """Convert to CARLA Transform."""
        loc = carla.Location(
            x=self.location['x'],
            y=self.location['y'],
            z=self.location['z']
        )
        rot = carla.Rotation(
            pitch=self.rotation['pitch'],
            yaw=self.rotation['yaw'],
            roll=self.rotation['roll']
        )
        return carla.Transform(loc, rot)


@dataclass
class Route:
    """A test route with multiple waypoints."""
    route_id: str
    map_name: str
    waypoints: List[RoutePoint]
    total_distance: float = 0.0  # Will be computed
    
    def compute_distance(self) -> float:
        """Compute total route distance."""
        if len(self.waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            p1 = self.waypoints[i].location
            p2 = self.waypoints[i + 1].location
            
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            dz = p2['z'] - p1['z']
            
            dist = (dx**2 + dy**2 + dz**2) ** 0.5
            total += dist
        
        self.total_distance = total
        return total


@dataclass
class Scenario:
    """A test scenario with environmental conditions."""
    scenario_id: str
    route_id: str
    weather: Dict = None  # weather parameters
    traffic_density: float = 0.5  # 0-1 scale
    pedestrian_density: float = 0.3
    background_vehicle_count: int = 0
    background_walker_count: int = 0
    ego_route_spacing: int = 1
    ego_spawn_mode: str = "route_spacing"
    ego_spawn_route_index: int = 0
    focus_background_near_egos: bool = False
    background_spawn_radius: float = 60.0
    difficulty: str = "moderate"  # easy, moderate, hard
    description: str = ""
    
    def __post_init__(self):
        if self.weather is None:
            self.weather = self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """Get default weather parameters."""
        return {
            'cloudiness': 0.3,
            'precipitation': 0.0,
            'precipitation_deposits': 0.0,
            'wind_intensity': 0.35,
            'sun_altitude_angle': 45.0,
            'sun_azimuth_angle': 0.0,
            'fog_distance': 1000.0,
            'fog_falloff': 0.1,
            'wetness': 0.0,
            'friction_scale': 1.0,
            'visibility_distance': 10000.0
        }


class ScenarioManager:
    """Manage routes and scenarios."""
    
    def __init__(self, scenarios_dir: str = None):
        """
        Initialize scenario manager.
        
        Args:
            scenarios_dir: Directory containing scenario definitions
        """
        if scenarios_dir is None:
            # Default to test_runner/scenarios/ directory
            scenarios_dir = str(Path(__file__).parent.parent / 'scenarios')
        
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        
        self.routes: Dict[str, Route] = {}
        self.scenarios: Dict[str, Scenario] = {}
        
        self._load_default_scenarios()
    
    def _load_default_scenarios(self):
        """Load default scenarios if they exist."""
        # Try to load from JSON files
        json_files = list(self.scenarios_dir.glob('*.json'))
        print(f"DEBUG: Looking for scenarios in: {self.scenarios_dir}")
        print(f"DEBUG: Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
        for json_file in json_files:
            print(f"DEBUG: Loading {json_file}")
            self._load_scenario_from_file(json_file)
    
    def _load_scenario_from_file(self, filepath: Path):
        """Load scenario from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            routes_count = len(data.get('routes', []))
            scenarios_count = len(data.get('scenarios', []))
            print(f"DEBUG: Loaded {filepath}: {routes_count} routes, {scenarios_count} scenarios")
            
            if 'routes' in data:
                for route_data in data['routes']:
                    self._register_route_from_dict(route_data)
            
            if 'scenarios' in data:
                for scenario_data in data['scenarios']:
                    self._register_scenario_from_dict(scenario_data)
                    
            print(f"DEBUG: Total scenarios now: {len(self.scenarios)}, total routes: {len(self.routes)}")
        except Exception as e:
            print(f"Warning: Could not load scenario file {filepath}: {e}")
    
    def _register_route_from_dict(self, route_dict: Dict):
        """Register a route from dictionary."""
        waypoints = [
            RoutePoint(
                location=wp['location'],
                rotation=wp.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
            )
            for wp in route_dict['waypoints']
        ]
        
        route = Route(
            route_id=route_dict['route_id'],
            map_name=route_dict['map_name'],
            waypoints=waypoints
        )
        route.compute_distance()
        
        self.routes[route.route_id] = route
    
    def _register_scenario_from_dict(self, scenario_dict: Dict):
        """Register a scenario from dictionary."""
        scenario = Scenario(
            scenario_id=scenario_dict['scenario_id'],
            route_id=scenario_dict['route_id'],
            weather=scenario_dict.get('weather'),
            traffic_density=scenario_dict.get('traffic_density', 0.5),
            pedestrian_density=scenario_dict.get('pedestrian_density', 0.3),
            background_vehicle_count=scenario_dict.get('background_vehicle_count', 0),
            background_walker_count=scenario_dict.get('background_walker_count', 0),
            ego_route_spacing=scenario_dict.get('ego_route_spacing', 1),
            ego_spawn_mode=scenario_dict.get('ego_spawn_mode', 'route_spacing'),
            ego_spawn_route_index=scenario_dict.get('ego_spawn_route_index', 0),
            focus_background_near_egos=scenario_dict.get('focus_background_near_egos', False),
            background_spawn_radius=scenario_dict.get('background_spawn_radius', 60.0),
            difficulty=scenario_dict.get('difficulty', 'moderate'),
            description=scenario_dict.get('description', '')
        )
        
        self.scenarios[scenario.scenario_id] = scenario
    
    def add_route(self, route: Route) -> None:
        """Add a route to the manager."""
        route.compute_distance()
        self.routes[route.route_id] = route
    
    def add_scenario(self, scenario: Scenario) -> None:
        """Add a scenario to the manager."""
        self.scenarios[scenario.scenario_id] = scenario
    
    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID."""
        return self.routes.get(route_id)
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID."""
        return self.scenarios.get(scenario_id)
    
    def get_routes_for_map(self, map_name: str) -> List[Route]:
        """Get all routes for a specific map."""
        return [r for r in self.routes.values() if r.map_name == map_name]
    
    def get_scenarios_for_route(self, route_id: str) -> List[Scenario]:
        """Get all scenarios for a specific route."""
        return [s for s in self.scenarios.values() if s.route_id == route_id]
    
    def list_routes(self) -> List[str]:
        """List all available route IDs."""
        return list(self.routes.keys())
    
    def list_scenarios(self) -> List[str]:
        """List all available scenario IDs."""
        return list(self.scenarios.keys())
    
    def save_scenarios(self, filepath: str):
        """Save current routes and scenarios to JSON."""
        data = {
            'routes': [],
            'scenarios': []
        }
        
        for route in self.routes.values():
            route_dict = {
                'route_id': route.route_id,
                'map_name': route.map_name,
                'waypoints': [
                    {
                        'location': wp.location,
                        'rotation': wp.rotation
                    }
                    for wp in route.waypoints
                ],
                'total_distance': route.total_distance
            }
            data['routes'].append(route_dict)
        
        for scenario in self.scenarios.values():
            scenario_dict = {
                'scenario_id': scenario.scenario_id,
                'route_id': scenario.route_id,
                'weather': scenario.weather,
                'traffic_density': scenario.traffic_density,
                'pedestrian_density': scenario.pedestrian_density,
                'background_vehicle_count': scenario.background_vehicle_count,
                'background_walker_count': scenario.background_walker_count,
                'ego_route_spacing': scenario.ego_route_spacing,
                'ego_spawn_mode': scenario.ego_spawn_mode,
                'ego_spawn_route_index': scenario.ego_spawn_route_index,
                'focus_background_near_egos': scenario.focus_background_near_egos,
                'background_spawn_radius': scenario.background_spawn_radius,
                'difficulty': scenario.difficulty,
                'description': scenario.description
            }
            data['scenarios'].append(scenario_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
