# LangCoop Scenarios Analysis

## Overview

This document provides a comprehensive analysis of the LangCoop official scenarios from the base repository and compares them with our current test setup.

---

## LangCoop Official Town05 Scenarios

### 1. Scenario Structure

The LangCoop repository defines **10 distinct scenario types** (Scenario1 through Scenario10), each with specific traffic challenges:

| Scenario | Primary Type | Description | Key Parameters |
|----------|-------------|-------------|----------------|
| **Scenario1** | **ControlLoss** | Ego vehicle experiences control jitter/noise | 10 jitters, noise_std=0.01 |
| **Scenario2** | **FollowLeadingVehicle** | Following lead vehicle, lane change, obstacles | 40% follow, 30% obstacle, 30% change lane |
| **Scenario3** | **DynamicObjectCrossing** | Pedestrians/cyclists crossing the road | 90% dynamic crossing, 10% free ride |
| **Scenario4** | **VehicleTurningRoute** | Vehicles turning at intersections | 50% route turns, 20% right, 20% left |
| **Scenario5** | **OtherLeadingVehicle** | Two vehicles ahead at different speeds | Lead speeds: 55 km/h, 45 km/h |
| **Scenario6** | **ManeuverOppositeDirection** | Oncoming traffic, opposite lane obstacles | Opposite speed: 5.56 m/s |
| **Scenario7** | **SignalJunctionCrossingRoute** | Crossing signalized junctions | Max velocity: 20 m/s |
| **Scenario8** | **SignalJunctionCrossingRoute** | Another signalized junction variant | Same as Scenario7 |
| **Scenario9** | **SignalJunctionCrossingRoute** | Third signalized junction variant | Same as Scenario7 |
| **Scenario10** | **NoSignalJunctionCrossingRoute** | Crossing unsignalized junctions | Max velocity: 20 m/s |

### 2. Scenario Details

#### **Scenario1: ControlLoss**
- **Purpose**: Test robustness to control noise/jitter
- **Mechanism**: Adds noise to steering/throttle commands
- **Parameters**:
  - Number of jitters: 10
  - Noise std dev: 0.01
  - Dynamic mean for steer: 0.001
  - Dynamic mean for throttle: 0.045
  - Start/end distance: 20m/30m
- **Traffic**: Background vehicles + pedestrians

#### **Scenario2: FollowLeadingVehicle (40%), ChangeLane (30%), FollowWithObstacle (30%)**
- **Purpose**: Car-following and lane change scenarios
- **Mechanism**: 
  - Lead vehicle at 10 km/h, 25m ahead
  - Fast/slow vehicles for lane change (70 km/h vs 0 km/h)
  - Obstacle on lane requiring lane change
- **Parameters**:
  - Trigger distance: 20-30m
  - Lead vehicle speed: 10 m/s
  - Fast vehicle speed: 70 km/h
- **Traffic**: Active scenarios with multiple vehicles

#### **Scenario3: DynamicObjectCrossing (90%), FreeRide (10%)**
- **Purpose**: Pedestrian/cyclist interactions
- **Mechanism**: 
  - Pedestrians/cyclists cross road dynamically
  - Target velocity: 3 m/s
  - Time to reach: 10s
- **Parameters**:
  - Trigger distance: 10m
  - Ego drive distance: 40m
  - Number of lane changes: 1
  - Walker yaw: 0°
- **Traffic**: Background + crossing pedestrians

#### **Scenario4: VehicleTurningRoute (50%), TurningRight (20%), TurningLeft (20%), FreeRide (10%)**
- **Purpose**: Intersection turning scenarios
- **Mechanism**: 
  - Vehicles turn right/left at intersections
  - Following route with turns
- **Parameters**:
  - Other actor velocity: 10 m/s
  - Pass distance: 30m
  - Number of attempts: 100
- **Traffic**: Turning vehicles at intersections

#### **Scenario5: OtherLeadingVehicle (100%)**
- **Purpose**: Two-vehicle following scenario
- **Mechanism**: 
  - Two vehicles ahead at different speeds
  - First at 55 km/h, second at 45 km/h
- **Parameters**:
  - First vehicle location: 35m
  - Second vehicle location: 36m
  - Ego drive distance: 140m
  - Max brake: 1.0
- **Traffic**: Two lead vehicles

#### **Scenario6: ManeuverOppositeDirection (100%)**
- **Purpose**: Oncoming traffic and opposite lane obstacles
- **Mechanism**: 
  - Vehicles in opposite lane
  - Ego must navigate around obstacles
- **Parameters**:
  - First vehicle: 8m
  - Second vehicle: 16m
  - Opposite speed: 5.56 m/s (20 km/h)
  - Ego drive distance: 32m
- **Traffic**: Oncoming vehicles

#### **Scenario7-9: SignalJunctionCrossingRoute (100%)**
- **Purpose**: Crossing signalized intersections
- **Mechanism**: 
  - Traffic lights, right-of-way rules
  - Must obey signals
- **Parameters**:
  - Max velocity: 20 m/s
  - Expected drive distance: 50m
  - Allowed drive distance: 20m
- **Traffic**: Junction traffic with signals

#### **Scenario10: NoSignalJunctionCrossingRoute (100%)**
- **Purpose**: Crossing unsignalized intersections
- **Mechanism**: 
  - No traffic lights
  - Must yield appropriately
- **Parameters**: Same as Scenario7-9
- **Traffic**: Junction traffic without signals

---

## 3. Traffic and Weather Configurations

The LangCoop repo defines **7 different parameter sets** with varying traffic densities:

| Config File | Vehicle Amount | Pedestrian Amount | CRAZY_LEVEL | Description |
|-------------|----------------|-------------------|-------------|-------------|
| `scenario_parameter.yaml` | **120** | **120** | 3 | **Dense traffic** (default) |
| `scenario_parameter_1.yaml` | **60** | **60** | 3 | **Medium traffic** |
| `scenario_parameter_2.yaml` | **80** | **80** | 3 | **Medium-high traffic** |
| `scenario_parameter_3.yaml` | **100** | **100** | 3 | **High traffic** |
| `scenario_parameter_4.yaml` | **40** | **40** | 3 | **Light traffic** |
| `scenario_parameter_5.yaml` | **20** | **20** | 3 | **Very light traffic** |
| `scenario_parameter_6.yaml` | **140** | **140** | 3 | **Very dense traffic** |

**CRAZY_LEVEL**: Controls erratic driving behavior (3 = moderate chaos, 50% of vehicles)

---

## 4. Town05 Spawn Locations

The official scenarios JSON defines **3 base Town05 scenarios** with multiple spawn configurations:

### **Town05 Base Scenarios**

| Scenario | Event Configs | Other Actors | Description |
|----------|--------------|--------------|-------------|
| **Town05-1** | Multiple positions | **No** | Basic spawn positions, no traffic |
| **Town05-2** | Multiple positions | **No** | Alternative spawn positions |
| **Town05-3** | Multiple positions | **Yes** (front, left, right) | Includes traffic vehicles at spawn |

**Sample Spawn Locations**:
- x=151.37, y=-26.18, yaw=88° (Common spawn)
- x=-5.6, y=201.85, yaw=0° (Alternative)
- x=64.13, y=187.79, yaw=178° (Another position)

**Total Event Configurations**: Each base scenario has dozens of spawn position variants (different x, y, yaw combinations).

---

## Your Current Test Setup

### Configuration Summary

| Aspect | Your Setup | Details |
|--------|-----------|---------|
| **Map** | Town05 | ✅ Same as LangCoop |
| **Routes** | 3 custom routes | 200m (short), 350m (medium), 100m (Town10) |
| **Weather** | 3 variants | Clear (easy), Cloudy (moderate), Rainy (hard) |
| **Vehicles** | **1 ego vehicle only** | ⚠️ No traffic/pedestrians |
| **Test Duration** | 500 steps (25 seconds) | Fixed duration |
| **Scenarios** | Simple waypoint following | No dynamic events |

### Your Scenario Definitions

```json
{
  "scenario_id": "town05_clear_easy",
  "route_id": "town05_short_01",
  "weather": {
    "cloudiness": 0.0,
    "precipitation": 0.0,
    "wind_intensity": 0.0,
    "sun_altitude_angle": 45.0
  },
  "traffic_density": 0.1,
  "pedestrian_density": 0.05,
  "difficulty": "easy"
}
```

**Your Routes**:
1. **town05_short_01**: Straight 200m (x: -152 to 50)
2. **town05_medium_01**: 350m with turns (includes 45° and 90° turns)
3. **town10_short_01**: Simple 100m route

---

## Comparison: Your Setup vs. LangCoop Official

### ✅ What Matches (Settings Aligned)

| Component | Status | Notes |
|-----------|--------|-------|
| **Curvature Scale** | ✅ Aligned | 0.01667 (degrees) |
| **Speed Control** | ✅ Aligned | PID with same gains (kp=5.0, kp_turn=1.0) |
| **Map** | ✅ Same | Town05 |
| **VLM Architecture** | ✅ Same | 4-stage CoT (scene/object/intent/plan) |
| **Control Pipeline** | ✅ Same | VLM → Speed/Curvature → PID → Actuators |
| **Image Resolution** | ✅ Same | 800x600 RGB |
| **Simulation Rate** | ✅ Same | 20 Hz synchronous |

### ⚠️ Key Differences (Where You Deviate)

| Component | LangCoop Official | Your Setup | Impact |
|-----------|------------------|-----------|--------|
| **Number of Vehicles** | **2 ego vehicles** (multi-agent) | **1 ego vehicle** | 🔴 Major difference |
| **Traffic Density** | **60-140 vehicles** | **0 vehicles** (config not active) | 🔴 No traffic |
| **Pedestrians** | **60-140 pedestrians** | **0 pedestrians** (config not active) | 🔴 No pedestrians |
| **Scenario Events** | **10 dynamic scenarios** (control loss, crossing, etc.) | **None** (just waypoint following) | 🔴 No events |
| **Weather Variations** | Not explicitly defined | 3 variants (clear, cloudy, rainy) | 🟡 Your addition |
| **Route Type** | Official spawn coordinates | Custom waypoints | 🟡 Different routes |
| **Test Duration** | Variable (scenario-dependent) | Fixed 500 steps (25s) | 🟡 Different |
| **Spawn Strategy** | Multiple precise locations | Custom start positions | 🟡 Different |

### 📊 Scenario Complexity Comparison

| Scenario Type | LangCoop | Your Setup |
|--------------|----------|------------|
| **Control loss/jitter** | ✅ Yes (Scenario1) | ❌ No |
| **Car following** | ✅ Yes (Scenario2) | ❌ No |
| **Pedestrian crossing** | ✅ Yes (Scenario3) | ❌ No |
| **Intersection turns** | ✅ Yes (Scenario4) | ❌ No |
| **Multi-vehicle following** | ✅ Yes (Scenario5) | ❌ No |
| **Oncoming traffic** | ✅ Yes (Scenario6) | ❌ No |
| **Signalized junctions** | ✅ Yes (Scenario7-9) | ❌ No |
| **Unsignalized junctions** | ✅ Yes (Scenario10) | ❌ No |
| **Multi-agent coordination** | ✅ Yes (2 ego vehicles) | ❌ No (1 vehicle) |
| **Background traffic** | ✅ Yes (60-140 vehicles) | ❌ No |
| **Weather variations** | ❌ No | ✅ Yes (3 variants) |

---

## What the "10 Scenarios in Town05 with 2 Cars" Means

Based on the code analysis, the paper's "10 scenarios" refers to:

1. **10 Scenario Types**: Scenario1 through Scenario10 (each with different traffic challenges)
2. **2 Ego Vehicles**: Multi-agent setup with two cooperative vehicles
3. **Multiple Runs**: Each scenario type likely tested with different:
   - Traffic densities (6 parameter sets: 20-140 vehicles)
   - Spawn positions (dozens of configurations per scenario)
   - Random seeds

**Estimated Total Test Cases**: 10 scenarios × 6 traffic densities × multiple spawn positions = **60+ unique test configurations**

---

## Summary of Differences

### 🎯 Your Approach (Simplified)
- **Focus**: VLM model capability testing
- **Environment**: Clean, controlled (no traffic/events)
- **Evaluation**: Direct assessment of VLM spatial reasoning
- **Benefit**: Clear attribution of failures to VLM weakness
- **Use Case**: Model selection and debugging phase

### 🎯 LangCoop Official (Comprehensive)
- **Focus**: Multi-agent cooperative driving
- **Environment**: Complex, realistic (dense traffic, dynamic events)
- **Evaluation**: Real-world driving competence
- **Benefit**: Rigorous evaluation of full system
- **Use Case**: Final benchmarking and paper results

---

## Recommendations

### Option 1: Keep Simplified Setup (Current)
**When to use**: Initial VLM testing and model selection

✅ **Pros**:
- Faster iteration on VLM models
- Clear failure attribution (VLM vs. environment)
- Easier debugging (no traffic interference)
- Lower computational cost

❌ **Cons**:
- Not comparable to paper results
- Doesn't test multi-agent coordination
- Misses dynamic scenario challenges

### Option 2: Upgrade to Official Scenarios
**When to use**: Final evaluation and paper comparison

✅ **Pros**:
- Direct comparison to LangCoop paper
- Tests full system capabilities
- Realistic evaluation

❌ **Cons**:
- Requires multi-agent implementation (~4-6 hours work)
- Higher computational cost
- More complex debugging

### Option 3: Hybrid Approach (Recommended)
**Phased progression**:

1. **Phase 1 (Current)**: Use simplified setup with 72B model
   - Verify 72B produces reasonable outputs
   - Confirm basic spatial reasoning works
   - Get Route Completion > 90%

2. **Phase 2**: Add background traffic (single agent)
   - Implement traffic spawning (20-60 vehicles)
   - Test traffic awareness
   - Keep single ego vehicle

3. **Phase 3**: Implement official scenarios (single agent)
   - Port Scenario1-10 logic
   - Test dynamic event handling
   - Still single ego vehicle

4. **Phase 4**: Multi-agent extension
   - Add second ego vehicle
   - Implement cooperation
   - Full LangCoop replication

---

## Next Steps

### Immediate (Current Sprint)
1. ✅ Deploy 72B model on A100-80GB
2. ✅ Run test with current simplified setup
3. ⏳ Verify VLM outputs are non-zero and reasonable
4. ⏳ Document performance on clean environment

### Near-term (If 72B Works)
1. Add background traffic spawning
2. Implement Scenario3 (pedestrian crossing) as first dynamic test
3. Port 1-2 more scenarios (e.g., Scenario2: car following)

### Long-term (If Aiming for Paper Comparison)
1. Implement full 10-scenario suite
2. Add multi-agent support (2 ego vehicles)
3. Run complete evaluation matrix (10 scenarios × 6 traffic densities)
4. Compare metrics directly to paper

---

## Conclusion

**Your current setup is NOT fully consistent with the LangCoop paper**:
- ❌ Missing: 2nd ego vehicle (multi-agent)
- ❌ Missing: Background traffic (60-140 vehicles)
- ❌ Missing: 10 dynamic scenarios
- ✅ Aligned: Control parameters, VLM architecture, simulation settings

**However**, your simplified approach is **valid for initial VLM testing**. The paper's 10 scenarios test the FULL system (multi-agent + traffic + events), while you're first validating the VLM component works at all.

**Recommendation**: Keep your simplified setup until 72B shows promising results (RC% > 90%, non-zero outputs), then progressively add complexity toward the official scenarios.
