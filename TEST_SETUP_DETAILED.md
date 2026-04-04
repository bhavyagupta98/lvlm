# LangCoop Test Runner - Detailed Test Setup & Metrics Calculation

## 1. TEST OVERVIEW

The LangCoop test is a **closed-loop autonomous driving evaluation** framework that:
- Spawns a vehicle in CARLA simulator
- Feeds camera images to a Vision Language Model (VLM)
- VLM predicts speed and steering (curvature) commands via Chain-of-Thought reasoning
- Vehicle executes predicted controls
- Metrics calculated: **Route Completion (RC%)** and **Driving Score (DS/100)**

**Entry point:** `python run_langcoop_test.py`

---

## 2. HOW THE TEST IS RUN

### 2.1 Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ CARLA Simulator (carla-rpc:2000, 20 Hz synchronous)            │
├──────────────────────────┬──────────────────────────────────────┤
│ Vehicle Sensors          │                                      │
│ - Camera (800x600)       │  Collision Detection (15-frame debounce)
│ - IMU (yaw/speed)        │  Speed monitoring (>20.5 m/s = violation)
└──────────────────────────┴──────────────────────────────────────┘
           ↑
           │ Image every 4 frames (skip_frames=4)
           │
┌──────────────────────────────────────────────────────────────────┐
│ LangCoopAgent (Local)                                           │
├──────────────────────────────────────────────────────────────────┤
│ 1. Collect camera image                                        │
│ 2. Call VLMPlannerSpeedCurvature                               │
│    - Build CoT prompt (4 stages: scene, objects, intent, prediction)
│    - Send to vLLM server (localhost:8000)                     │
│    - Parse JSON response: 5 [speed, curvature] pairs          │
│ 3. Take first prediction [speed₁, curvature₁]                  │
│ 4. Convert to throttle/brake/steer via PID controllers         │
│ 5. Apply control to vehicle                                    │
└──────────────────────────────────────────────────────────────────┘
           ↓
        vLLM Server (Port 8000, GPU)
        Running: Qwen2.5-VL-7B-Instruct-AWQ
        (or Qwen2.5-VL-72B-Instruct-AWQ for improved)
```

### 2.2 Simulation Loop (Per Timestep)

**Clock:** 20 Hz (0.05s per frame), 500 frames = 25 seconds

```python
for step in range(max_steps):
    # 1. Core simulation tick
    world.tick()
    
    # 2. Agent decision (every 4 frames = 0.2s)
    if step % skip_frames == 0:
        control = agent.step()           # Call VLM
        vehicle.apply_control(control)   # Actuate
    
    # 3. Metric collection
    update_distance_traveled()
    record_speed_sample()
    
    # 4. Infraction detection
    if speed > 20.5 m/s:
        metrics.record_event('speed_violation')
    check_collision_sensor()     # Debounced at 15 frames (~0.75s)
    
    # 5. Data logging (every 10 frames)
    if step % 10 == 0:
        save_camera_image()   # For visualization
        print_progress()
```

---

## 3. CONFIGURATION & KNOBS

### 3.1 Deployment Config (`carla_client_dev_4.yaml`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Model** | Qwen2.5-VL-7B-Instruct-AWQ (or 72B) | Vision-Language reasoning |
| **vLLM Version** | 0.12.0 | Backend inference engine |
| **GPU Memory Utilization** | 0.80 | How much VRAM to consume |
| **Max Model Length** | 4096 tokens | Context window size |
| **Max Num Seqs** | 2 | Batch size for inference |
| **dtype** | float16 | Precision (16-bit floats) |
| **Port** | 8000 | API server port |

### 3.2 Agent Config (`langcoop_agent_config.yaml` or `langcoop_agent_config_32b.yaml`)

#### Camera Settings
```yaml
camera:
  width: 800       # Pixels (standard for CARLA)
  height: 600
  fov: 100         # Field of view degrees
```

#### Control Parameters (PID Controllers)
```yaml
control:
  max_speed: 20.0 m/s              # Hard speed limit
  speed_kp: 5.0                    # Speed controller gain
  turn_kp: 1.0                     # Steering controller gain
  clip_delta: 0.35                 # Max steering angle change per step
  brake_ratio: 1.1                 # Brake efficiency multiplier
  brake_speed: 0.1 m/s             # Speed below which to use full brake
  curvature_scale: 0.001745        # π/1800: Upstream scale (divide by 10, then deg→rad)
```

**Curvature Mapping (CRITICAL FIX - March 5, 2026):**
- VLM predicts: `-180 to +180 degrees`
- Upstream process: divide by 10 → convert to radians
  - Example: `-90° ÷ 10 = -9° → deg2rad(-9°) = -0.157 rad`
- Our implementation: `curvature * curvature_scale`
  - Example: `-90 * 0.001745 = -0.157 rad` ✓ Matches upstream
- PID controller maps to throttle/brake/steer CARLA commands

**Why this matters:**
- Old scale (0.01667): `-90 * 0.01667 = -1.5` steering input ❌ **Way too aggressive**
- New scale (0.001745): `-90 * 0.001745 = -0.157` steering input ✓ **Correct, matches upstream**

#### CoT Prompts (4-Stage Reasoning)
```yaml
planning:
  prompt_template:
    scene:       # "Describe driving scenario..."
    objects:     # "Identify important road users..."
    intent:      # "Should you turn/accelerate/decelerate?"
    prediction:  # "Generate 5 [speed, curvature] pairs"
```

### 3.3 Test Runner Knobs

```python
# run_langcoop_test.py command line
python run_langcoop_test.py \
    --host carla-rpc                                    # CARLA server
    --port 2000                                         # CARLA port
    --agent-config configs/langcoop_agent_config.yaml  # Config file
    --scenario-ids town05_clear_easy                   # Which test(s)
    --max-steps 500                                     # Simulation length (25s)
    --results-dir test_results_langcoop                # Output directory
```

**Key Scenario Configurations:**

| Scenario ID | Route | Distance | Weather | Traffic | Difficulty |
|-------------|-------|----------|---------|---------|------------|
| `town05_clear_easy` | Straight line | 200m | Clear, 0% clouds | 10% density | Easy |
| `town05_cloudy_medium` | With curves | 350m | Cloudy, 30% | 30% | Medium |
| `town10_short_01` | Straight | 100m | Clear | 5% | Easy |

---

## 4. METRICS CALCULATION

### 4.1 Route Completion (RC%)

**Definition:** Percentage of route distance successfully traversed

```
RC% = (distance_traveled / total_route_distance) × 100
Max: 100%
```

**How tracked:**
- At each frame, compute vehicle location
- Calculate delta-distance from previous location
- Accumulate over all frames
- Compare to pre-computed route distance

**Example (town05_clear_easy):**
```
Total route: 200 meters
Vehicle traveled: 177 meters
RC% = (177 / 200) × 100 = 88.5%
```

### 4.2 Driving Score (DS/100)

**Definition:** Quality metric penalizing infractions

**Base formula:**
```
DS = 100 - infractions_penalty
Clamped to [0, 100]
```

**Infraction Penalties:**

| Event Type | Penalty | When Triggered |
|-----------|---------|----------------|
| **Collision** | -60 points each | Contact with object detected |
| **Lane Departure** | -30 points each | Vehicle leaves designated lane |
| **Speed Violation** | -4 points each | Speed > 20.5 m/s |
| **Timeout** | -100 points | Agent step fails (crash) |
| **Incompleteness** | -0.1 × (100 - RC%) | If route not completed |

**Calculation Code:**
```python
score = 100.0
score -= collisions × 60
score -= lane_departures × 30
score -= speed_violations × 4

if not route_completed:
    incompleteness = (100 - rc_percentage) × 0.1
    score -= incompleteness

score = max(0, min(100, score))  # Clamp
```

**Example Scenarios:**

| Scenario | Collisions | Speed Viol. | RC | DS Calculation | DS Score |
|----------|-----------|------------|-----|---|---|
| Perfect route | 0 | 0 | 100% | 100 - 0 | **100/100** |
| 1 collision | 1 | 2 | 100% | 100 - 60 - 8 | **32/100** |
| No collision, 88.5% complete | 0 | 0 | 88.5% | 100 - (1.15 × 0.1) | **98.9/100** |
| Failed (timeout) | - | - | 0% | 0 | **0/100** |

**Test Output Example:**
```
Route Completion: 88.5%
Driving Score: 98.9/100
Collisions: 0
Speed Violations: 0
```

---

## 5. COLLISION DETECTION

**Sensor:** CARLA collision sensor attached to vehicle

**Debouncing:** 15-frame debounce (~0.75 seconds at 20 Hz)
- Prevents double-counting from bouncing/sliding
- Single rubbing contact = 1 collision event
- Hard crash with stop = 1 collision event

**Detection:**
```python
def on_collision(event):
    frame = event.frame
    if frame - last_collision_frame < 15:
        return  # Skip duplicate
    
    metrics.record_event('collision', frame, description=event.other_actor.type_id)
```

---

## 6. TEST EXECUTION TIMELINE

**Total test duration for 1 scenario (500 steps):**
- Simulation time: 25 seconds (500 frames × 0.05s)
- Wall clock time: ~60-120 seconds (depending on VLM inference speed)
  - 7B model: 2-3s per inference → ~30s total
  - 72B model: 5-7s per inference → ~60s total

**Full test sequence:**
```
1. Setup CARLA connection        (5s)
2. Load map                       (2s)
3. Spawn vehicle                  (1s)
4. Spawn sensors                  (1s)
5. Initialize agent & VLM         (30s for model load)
6. Run simulation loop:
   - 500 frames @ 20Hz            (25s simulation + VLM inference)
7. Cleanup                        (2s)
8. Calculate metrics              (1s)
9. Save results & visualizations  (5s)
─────────────────────────────
Total: ~2-3 minutes per test scenario
```

---

## 7. OUTPUT & RESULTS

### Saved Files:
```
test_results_langcoop/
├── metrics_summary.json          # Raw metrics data
├── test_log.txt                  # Full simulation log
├── images/
│   └── town05_clear_easy/
│       ├── frame_000000.jpg      # Camera frames every 0.5s
│       ├── frame_000100.jpg
│       └── ...
└── visualizations/
    └── metrics_plot.png          # RC% and DS/100 graphs
```

### Console Output Example:
```
Route Completion: 88.5%
Driving Score: 98.9/100
Collisions: 0
Speed Violations: 0
Distance traveled: 177.3m
Average speed: 7.8 m/s
Scenario complete: SUCCESS
```

---

## 8. CURRENT TEST STATE (UPDATED March 5, 2026)

### Bug Fix: Curvature Scaling

**Issue Identified:** The 72B model was predicting strong left turns (-30 to -90°) even on straight roads.

**Root Cause:** `curvature_scale` was set to **3.0** in controller and **0.01667** in configs.
- Old formula: `steer = -90 * 0.01667 = -1.5` (steering command way too aggressive)
- Should be: `steer = -90 * 0.001745 = -0.157` (matches upstream)

**Upstream behavior:** They divide curvature by 10 BEFORE converting to radians
```python
curv = np.deg2rad(route_info['curvature'][i] / 10)  # LangCoop
# For -90 degrees: deg2rad(-90/10) = deg2rad(-9) = -0.157 rad
```

**Fix Applied:**
- Updated `curvature_scale` in `langcoop_controller.py`: 3.0 → 0.001745
- Updated `curvature_scale` in both agent configs: 0.01667 → 0.001745
- Added explanatory comments about upstream scaling

### What We're Testing (Post-Fix):

| Component | Current Model | Expected | With Fix |
|-----------|---------------|----------|----------|
| VLM Reasoner | Qwen2.5-VL-72B-AWQ | Non-zero predictions | ✓ |
| Speed Prediction | CoT reasoning | 5-20 m/s | Reasonable |
| Curvature Prediction | CoT reasoning | -30 to +30 deg (on curves) | Should see ~0° on straight roads |
| Route Completion | Vehicle control | 95-100% | Should improve post-fix |
| Steering Response | PID controller | ±0.35 max | Now properly scaled |

### Next Test:
Running with **fixed curvature_scale** should resolve the excessive left-turning issue on straight roads.
