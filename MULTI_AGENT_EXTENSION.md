# Multi-Agent Extension: 2 Ego Vehicles

## OVERVIEW

Currently, the test runner supports **1 ego vehicle (1 agent)**. To extend to **2 ego vehicles with independent agents**, we need architectural changes at multiple levels. The LangCoop paper already demonstrates this with heterogeneous agent scenarios.

---

## 1. CURRENT SINGLE-AGENT ARCHITECTURE

```
┌──────────────────────────────────────┐
│ LangCoopTestRunner (main)            │
├──────────────────────────────────────┤
│                                      │
│  self.vehicle (single)     ──┐────►  │
│  self.agent (single)       ──┼────►  │
│  self.metrics (single ID)  ──┴────►  │
│                                      │
│  run_scenario():                     │
│    ├─ Spawn 1 vehicle               │
│    ├─ Create 1 agent                │
│    ├─ Run 1 loop:                   │
│    │  ├─ agent.step() → control     │
│    │  └─ vehicle.apply_control()    │
│    └─ Track metrics for "agent_0"   │
└──────────────────────────────────────┘
```

---

## 2. PROPOSED 2-AGENT ARCHITECTURE

```
┌────────────────────────────────────────────────┐
│ LangCoopTestRunner (modified)                  │
├────────────────────────────────────────────────┤
│                                                │
│  ego_num: 2                    ┌─────────┐    │
│  • vehicles: [vehicle_0, vehicle_1]  │    │    │
│  • agents: [agent_0, agent_1]      │    │    │
│  • metrics: [metrics_0, metrics_1] │    │    │
│                                    │    │    │
│  run_scenario():                   │    │    │
│  ┌─────────────────────────────┐   │    │    │
│  │ For each step:              │   │    │    │
│  │   For i in [0, 1]:          │   │    │    │
│  │   ├─ control_i = agents[i].step()    │    │
│  │   └─ vehicles[i].apply_control()    │    │
│  │                             │   │    │    │
│  │   Update metrics[0] & metrics[1]    │    │
│  │   Check collisions between vehicles │    │
│  └─────────────────────────────┘   │    │    │
│                                     └─────────┘
└────────────────────────────────────────────────┘
```

---

## 3. NEEDED CODE CHANGES

### 3.1 **Test Runner Modifications** (`run_langcoop_test.py`)

#### Change 1: Track multiple vehicles & agents
```python
class LangCoopTestRunner:
    def __init__(self, ego_num: int = 1, ...):
        self.ego_num = ego_num
        self.vehicles = {}           # {agent_id: vehicle}
        self.agents = {}             # {agent_id: agent}
        self.metrics_calculators = {} # One per vehicle
        self.vehicle_configs = []    # Config per vehicle
```

#### Change 2: Modified spawn/setup
```python
def run_scenario(self, scenario, route, ego_num=2, skip_frames=4):
    """
    Run scenario with multiple ego vehicles.
    """
    spawn_points = self._get_spawn_points(route, ego_num)
    # spawn_points = [point_0, point_1]
    
    # Spawn all vehicles first
    for agent_id in range(ego_num):
        vehicle = self.spawn_vehicle(spawn_points[agent_id])
        agent = LangCoopAgent(self.agent_config)
        agent.setup(vehicle)
        agent.agent_idx = agent_id  # Mark which agent this is
        
        self.vehicles[agent_id] = vehicle
        self.agents[agent_id] = agent
        self.metrics_calculators[agent_id] = MetricsCalculator()
```

#### Change 3: Multi-agent simulation loop
```python
# Current (single agent):
for step in range(max_steps):
    world.tick()
    if step % skip_frames == 0:
        control = self.agent.step()
        self.vehicle.apply_control(control)

# New (multi-agent):
for step in range(max_steps):
    world.tick()
    
    if step % skip_frames == 0:
        # Get controls from all agents in parallel
        controls = {}
        for agent_id in range(self.ego_num):
            control = self.agents[agent_id].step()
            controls[agent_id] = control
        
        # Apply all controls simultaneously
        for agent_id in range(self.ego_num):
            self.vehicles[agent_id].apply_control(controls[agent_id])
    
    # Update metrics for each vehicle separately
    for agent_id in range(self.ego_num):
        vehicle = self.vehicles[agent_id]
        metrics_id = f"{scenario_id}_agent_{agent_id}"
        
        # Track distance, collisions, speed
        update_metrics(agent_id, metrics_id)
```

#### Change 4: Metrics tracking
```python
# Current:
route_metric_id = f"{scenario.scenario_id}_agent_0"
self.metrics_calculator.record_event(route_metric_id, ...)

# New:
for agent_id in range(ego_num):
    route_metric_id = f"{scenario.scenario_id}_agent_{agent_id}"
    self.metrics_calculators[agent_id].record_event(route_metric_id, ...)
```

### 3.2 **Agent Modifications** (`langcoop_agent.py`)

```python
class LangCoopAgent:
    def __init__(self, agent_config_path: str, agent_idx: int = 0):
        self.agent_idx = agent_idx  # NEW: Which agent is this? (0 or 1)
        self.agent_id = f"agent_{agent_idx}"
        # ... rest of init
    
    def _plan_with_vlm(self):
        """Modified to include other agents in scene understanding."""
        # NEW: Add other vehicles to CoT context
        other_agents_info = self._get_collaborative_agent_info()
        
        # Pass to VLM planner with:
        # - Self image
        # - Self history
        # - Other agents' positions/speeds (if available)
        planned_result = self.vlm_planner.forward(
            self.perception_memory_bank,
            model_config=self.config,
            other_agents=other_agents_info,  # NEW
            agent_idx=self.agent_idx         # NEW
        )
        
        return planned_result
    
    def _get_collaborative_agent_info(self) -> Dict:
        """Get positions/velocities of other agents."""
        # Placeholder for actual implementation
        return {
            'other_agents': [],
            'relative_positions': {},
            'collaborative_context': ""
        }
```

### 3.3 **Collision Detection Between Agents**

Currently: Collision with static objects only

New: Add inter-agent collision check

```python
def _check_agent_collisions(self, agent_0: carla.Actor, agent_1: carla.Actor):
    """Check if two vehicles collided with each other."""
    loc_0 = agent_0.get_location()
    loc_1 = agent_1.get_location()
    
    distance = loc_0.distance(loc_1)
    vehicle_width = 2.0  # Tesla Model 3 width
    
    if distance < vehicle_width:
        return True  # Collision detected
    return False
```

---

## 4. CONFIGURATION CHANGES

### 4.1 New scenario config for 2-agent scenarios

```yaml
# New file: scenarios/town05_2agent_easy.json
{
  "scenario_id": "town05_2agent_easy",
  "route_id": "town05_short_01",
  "ego_num": 2,  # NEW
  "spawn_strategy": "parallel",  # parallel, sequential, staggered
  "agent_configs": [
    "configs/langcoop_agent_config.yaml",      # Agent 0
    "configs/langcoop_agent_config_32b.yaml"   # Agent 1 (different model)
  ],
  "communication_enabled": false,  # Can agents share info?
  "waypoint_sharing": false,       # Do both follow same route?
}
```

### 4.2 Command line argument

```bash
# Single agent (current)
python run_langcoop_test.py --scenario-ids town05_clear_easy

# Two agents (new)
python run_langcoop_test.py --scenario-ids town05_2agent_easy --ego-num 2
```

---

## 5. KEY IMPLEMENTATION DECISIONS

### Decision 1: Spawn Point Strategy

**Option A: Parallel (Side-by-side)**
```
Car 0: x=-152, y=134      Car 1: x=-152, y=140
  ↓                          ↓
  Route forward
```
- **Pro:** Can test collision avoidance
- **Con:** Harder to fit on narrow roads

**Option B: Staggered (Following)**
```
Car 0: x=-152, y=134 (at t=0)
Car 1: x=-160, y=134 (5m behind, follows same route)
```
- **Pro:** Tests overtaking, merging
- **Con:** Less coordination needed

**Recommended:** Staggered for initial testing

### Decision 2: VLM Usage Per Agent

**Option A: Separate VLM calls (Current implementation)**
```
Agent 0 → VLM server (port 8000) → Prediction 0
Agent 1 → VLM server (port 8000) → Prediction 1
```
- **Sequential:** Agent 0 calls, then Agent 1 (slower)
- **Parallel:** Both call simultaneously (faster, needs threading)

**Option B: Batch processing**
```
Both agents' images → Single VLM call → Get [Pred 0, Pred 1]
```
- **Pro:** More efficient
- **Con:** Requires modified VLM interface

**Recommended:** Option A with threading (simple change)

### Decision 3: Agent Awareness

**Option A: Independent agents (simplest)**
- Each agent only sees camera, not other agent
- No communication
- Can test independent operation

**Option B: Aware agents (complex)**
- Agent 0 sees: Own camera + Agent 1's position/velocity
- Agent 1 sees: Own camera + Agent 0's position/velocity
- VLM can reason about other vehicle

**Recommended:** Start with Option A, add Option B later

---

## 6. TEST SCENARIOS FOR 2 AGENTS

### Scenario 1: Parallel Straight (Easy)
```
Agent 0: -152, 134 → (50, 134)
Agent 1: -152, 140 → (50, 140)

Expected: Both complete 100% RC, no collisions, DS=100/100
```

### Scenario 2: Staggered Curves (Medium)
```
Agent 0: Leads on same route
Agent 1: Follows 10m behind (staggered start)

Expected: 
  - Agent 0: 100% RC, 100/100 DS
  - Agent 1: 100% RC, might hit Agent 0 if not careful
```

### Scenario 3: Close Parallel Turn (Hard)
```
Agent 0 & 1 side-by-side, route curves
Test: Do they collide in curve?

Expected:
  - If agents are aware: Coordinate, both 100% RC
  - If independent: Likely collision, low DS
```

---

## 7. METRICS FOR 2-AGENT SCENARIOS

### Individual Metrics (per agent)
```
Route Completion 0: 100%
Driving Score 0: 98/100

Route Completion 1: 100%
Driving Score 1: 95/100
```

### Aggregate Metrics
```
Avg Route Completion: 100%
Avg Driving Score: 96.5/100
Inter-Agent Collisions: 0
```

---

## 8. IMPLEMENTATION ROADMAP

### Phase 1: Basic Support (1-2 hours)
- [ ] Modify test runner to spawn 2 vehicles
- [ ] Create 2 agents independently
- [ ] Track metrics separately
- [ ] Run test: `--ego-num 2 --scenario-ids town05_clear_easy`
- **Expected:** Both agents operate independently, no communication

### Phase 2: Collision Detection (30 mins)
- [ ] Add inter-agent collision sensor (bonus check beyond static collisions)
- [ ] Penalize inter-agent crashes in metrics
- **Expected:** Detect when agents hit each other

### Phase 3: Agent Awareness (1-2 hours)
- [ ] Agent sees other agent's position/velocity in camera view
- [ ] Pass collaborative agent info to VLM
- [ ] Modify CoT prompt to include other agents
- **Expected:** Agents reason about coordination

### Phase 4: Communication (Optional)
- [ ] Agents share location/speed predictions
- [ ] Can inform each other of plans
- **Expected:** Better coordination, fewer collisions

---

## 9. DATA STRUCTURES FOR 2-AGENT VERSION

```python
# New structure for test results
results = {
    'scenario_id': 'town05_2agent_easy',
    'ego_num': 2,
    'agents': {
        'agent_0': {
            'route_completion': 100.0,
            'driving_score': 98.5,
            'collisions_static': 0,
            'collisions_with_agents': 0,
            'distance': 200.0,
            'speed_avg': 8.2,
        },
        'agent_1': {
            'route_completion': 100.0,
            'driving_score': 95.2,
            'collisions_static': 0,
            'collisions_with_agents': 0,
            'distance': 200.0,
            'speed_avg': 7.9,
        }
    },
    'aggregate': {
        'avg_route_completion': 100.0,
        'avg_driving_score': 96.85,
        'total_inter_agent_collisions': 0,
        'coordination_quality': 'independent'
    }
}
```

---

## 10. COMMAND EXAMPLES

### Run 2-agent test (independent)
```bash
python run_langcoop_test.py \
    --ego-num 2 \
    --scenario-ids town05_2agent_easy \
    --agent-config configs/langcoop_agent_config_32b.yaml
```

### Run test with different models per agent
```bash
python run_langcoop_test.py \
    --ego-num 2 \
    --scenario-ids town05_2agent_heterogeneous \
    --agent-configs \
        configs/langcoop_agent_config.yaml \
        configs/langcoop_agent_config_32b.yaml
```

### Run with collision analysis
```bash
python run_langcoop_test.py \
    --ego-num 2 \
    --scenario-ids town05_2agent_easy \
    --analyze-collisions
```

---

## SUMMARY

| Aspect | Single Agent | Dual Agent |
|--------|------|------|
| **Vehicles** | 1 | 2 (parallel or staggered) |
| **Agents** | 1 config | 1-2 configs |
| **VLM Calls** | 1 per step | 2 per step (sequential or parallel) |
| **Metrics** | 1 set | 2 sets + aggregate |
| **Collisions** | Static only | Static + inter-agent |
| **Configuration** | Single scenario | Scenario with `ego_num: 2` |
| **Complexity** | Baseline | +20% code changes |
| **Expected Time** | 2-4 min per test | 3-6 min per test |

**Next Step:** Implement Phase 1 (basic spawn & separate metrics) to support 2 vehicles with independent agents.
