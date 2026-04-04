"""Metrics calculation for autonomous driving evaluation."""

import math
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class VehicleEvent:
    """Single event during simulation."""
    frame: int
    event_type: str  # 'collision', 'lane_departure', 'speed_violation', 'timeout'
    severity: float = 1.0  # Severity multiplier
    description: str = ""


@dataclass
class RouteCompletionMetrics:
    """Metrics for a single route completion."""
    route_id: str
    total_distance: float = 0.0  # meters traveled
    target_distance: float = 0.0  # total route distance
    completion_percentage: float = 0.0
    
    # Events and violations
    collisions: int = 0
    lane_departures: int = 0
    speed_violations: int = 0
    timeouts: int = 0
    events: List[VehicleEvent] = field(default_factory=list)
    
    # Time metrics
    elapsed_time: float = 0.0  # seconds
    route_time_reference: float = 0.0  # expected time from offline route
    
    # Speed metrics
    average_speed: float = 0.0
    max_speed: float = 0.0
    
    # Driving score
    driving_score: float = 0.0
    success: bool = False


class MetricsCalculator:
    """
    Calculate official driving metrics:
    - Route Completion (RC): Percentage of route completed
    - Driving Score (DS): 0-100 based on performance and violations
    """
    
    # Define infraction penalties (like official leaderboard)
    PENALTY_COLLISION = 0.6  # Heavy penalty
    PENALTY_LANE_DEPARTURE = 0.3
    PENALTY_SPEED_VIOLATION = 0.04
    PENALTY_TIMEOUT = 1.0  # Complete failure
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.routes: Dict[str, RouteCompletionMetrics] = {}
        self.overall_score = 0.0
    
    def record_event(self, route_id: str, frame: int, event_type: str,
                    severity: float = 1.0, description: str = ""):
        """Record an event during simulation."""
        if route_id not in self.routes:
            self.routes[route_id] = RouteCompletionMetrics(route_id=route_id)
        
        event = VehicleEvent(
            frame=frame,
            event_type=event_type,
            severity=severity,
            description=description
        )
        self.routes[route_id].events.append(event)
        
        # Update event counters
        if event_type == 'collision':
            self.routes[route_id].collisions += 1
        elif event_type == 'lane_departure':
            self.routes[route_id].lane_departures += 1
        elif event_type == 'speed_violation':
            self.routes[route_id].speed_violations += 1
        elif event_type == 'timeout':
            self.routes[route_id].timeouts += 1
    
    def set_route_completion(self, route_id: str, distance_traveled: float,
                            target_distance: float, elapsed_time: float = 0.0):
        """Set route completion data."""
        if route_id not in self.routes:
            self.routes[route_id] = RouteCompletionMetrics(route_id=route_id)
        
        metrics = self.routes[route_id]
        metrics.total_distance = distance_traveled
        metrics.target_distance = target_distance
        metrics.completion_percentage = (distance_traveled / target_distance * 100) if target_distance > 0 else 0.0
        metrics.elapsed_time = elapsed_time
        
        # Success = completed entire route
        metrics.success = distance_traveled >= target_distance
    
    def set_speed_metrics(self, route_id: str, avg_speed: float, max_speed: float):
        """Set speed-related metrics."""
        if route_id not in self.routes:
            self.routes[route_id] = RouteCompletionMetrics(route_id=route_id)
        
        self.routes[route_id].average_speed = avg_speed
        self.routes[route_id].max_speed = max_speed
    
    def calculate_driving_score(self, route_id: str) -> float:
        """
        Calculate Driving Score (0-100).
        Score = max(0, 100 - infractions)
        
        Infractions computed as:
        - Collision: -60 points
        - Lane departure: -30 points
        - Speed violation: -4 points
        - Timeout: -100 points (complete failure)
        
        Formula similar to official CARLA leaderboard.
        """
        if route_id not in self.routes:
            return 0.0
        
        metrics = self.routes[route_id]
        
        # Start with perfect score
        score = 100.0
        
        # Apply penalties
        score -= metrics.collisions * (self.PENALTY_COLLISION * 100)
        score -= metrics.lane_departures * (self.PENALTY_LANE_DEPARTURE * 100)
        score -= metrics.speed_violations * (self.PENALTY_SPEED_VIOLATION * 100)
        
        # Timeout means complete failure
        if metrics.timeouts > 0:
            score = 0.0
        
        # Incomplete route: additional penalty based on completion percentage
        if not metrics.success and metrics.completion_percentage > 0:
            incompleteness_penalty = (100 - metrics.completion_percentage) * 0.1
            score -= incompleteness_penalty
        
        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))
        
        metrics.driving_score = score
        return score
    
    def calculate_route_completion(self, route_id: str) -> float:
        """
        Calculate Route Completion percentage.
        RC = (distance_traveled / total_distance) * 100
        Max 100%.
        """
        if route_id not in self.routes:
            return 0.0
        
        metrics = self.routes[route_id]
        return min(100.0, metrics.completion_percentage)
    
    def get_route_metrics(self, route_id: str) -> RouteCompletionMetrics:
        """Get complete metrics for a route."""
        if route_id not in self.routes:
            return RouteCompletionMetrics(route_id=route_id)
        return self.routes[route_id]
    
    def get_summary(self) -> Dict:
        """Get summary of all routes."""
        if not self.routes:
            return {
                'routes_tested': 0,
                'avg_route_completion': 0.0,
                'avg_driving_score': 0.0,
                'successful_routes': 0,
                'total_collisions': 0,
                'total_violations': 0
            }
        
        route_completions = []
        driving_scores = []
        successful = 0
        total_collisions = 0
        total_violations = 0
        
        for metrics in self.routes.values():
            # Calculate scores if not done
            self.calculate_driving_score(metrics.route_id)
            
            route_completions.append(metrics.completion_percentage)
            driving_scores.append(metrics.driving_score)
            
            if metrics.success:
                successful += 1
            
            total_collisions += metrics.collisions
            total_violations += (metrics.lane_departures + metrics.speed_violations)
        
        avg_rc = np.mean(route_completions) if route_completions else 0.0
        avg_ds = np.mean(driving_scores) if driving_scores else 0.0
        
        return {
            'routes_tested': len(self.routes),
            'avg_route_completion': float(avg_rc),
            'avg_driving_score': float(avg_ds),
            'successful_routes': successful,
            'total_collisions': total_collisions,
            'total_violations': total_violations,
            'routes': {
                rid: {
                    'rc': metrics.completion_percentage,
                    'ds': metrics.driving_score,
                    'collisions': metrics.collisions,
                    'violations': metrics.lane_departures + metrics.speed_violations,
                    'distance': metrics.total_distance
                }
                for rid, metrics in self.routes.items()
            }
        }
    
    def print_summary(self):
        """Print metrics summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("ROUTE EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Routes Tested:        {summary['routes_tested']}")
        print(f"Successful Routes:    {summary['successful_routes']}")
        print(f"Avg Route Completion: {summary['avg_route_completion']:.1f}%")
        print(f"Avg Driving Score:    {summary['avg_driving_score']:.1f}/100")
        print(f"Total Collisions:     {summary['total_collisions']}")
        print(f"Total Violations:     {summary['total_violations']}")
        print("=" * 70)
        
        if summary['routes_tested'] > 0:
            print("\nPer-Route Breakdown:")
            print("-" * 70)
            print(f"{'Route ID':<20} {'RC %':<12} {'DS/100':<12} {'Collisions':<15}")
            print("-" * 70)
            for rid, data in summary['routes'].items():
                print(f"{rid:<20} {data['rc']:<12.1f} {data['ds']:<12.1f} {data['collisions']:<15}")
            print("-" * 70 + "\n")
