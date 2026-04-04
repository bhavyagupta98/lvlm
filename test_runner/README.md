# Test Runner Module - Quick Start Guide

A lightweight, clean test runner for LangCoop simulations using **Python 3.12** and **CARLA 0.9.16**.

## Quick Start

### 1. Prerequisites

- **Python 3.12+** installed
- **CARLA 0.9.16** server running
- **Minimum 2GB VRAM** for GPU, or CPU fallback

### 2. Setup Environment

```bash
# Create Python 3.12 virtual environment
python3.12 -m venv venv_test
source venv_test/bin/activate  # On Windows: venv_test\Scripts\activate

# Install dependencies
pip install -r test_runner/requirements.txt
```

### 3. Start CARLA Server

```bash
# Terminal 1: Start CARLA server
# Replace path with your CARLA installation
./CARLA_0.9.16/CarlaUE4.sh -quality-level=Low
```

### 4. Run Your First Test

```bash
# Terminal 2: Run test with default settings
python test_runner_example.py

# Or with custom parameters:
python test_runner_example.py \
  --map Town05 \
  --steps 5000 \
  --agents 2 \
  --weather ClearNoon
```

## Module Structure

```
test_runner/
├── __init__.py                 # Package initialization
├── agents/
│   ├── __init__.py
│   └── simple_agent.py        # Minimal autonomous agent
├── controllers/
│   ├── __init__.py
│   └── pid_controller.py      # PID steering/throttle control
├── configs/
│   └── default_agent.yaml     # Default agent configuration
├── test_runner.py             # Main test orchestration
└── requirements.txt           # Python dependencies
```

## Components Explained

### SimpleAutonomousAgent
Handles:
- **Perception**: Collects camera & IMU sensor data
- **Planning**: Basic waypoint-following (extensible to VLM)
- **Control**: PID controller for steering & speed

```python
from test_runner.agents import SimpleAutonomousAgent

agent = SimpleAutonomousAgent('test_runner/configs/default_agent.yaml')
agent.setup(carla_vehicle)
control = agent.step()  # Returns carla.VehicleControl
```

### PIDController
Converts:
- Target speed → throttle/brake [-1, 1]
- Target yaw → steering angle [-1, 1]

```python
from test_runner.controllers import PIDController

controller = PIDController(kp_throttle=0.5, kp_steering=0.8)
throttle, brake = controller.compute_throttle_brake(current_speed, target_speed)
steer = controller.compute_steering(current_yaw, target_yaw)
```

### TestRunner
Main orchestrator:
- Connects to CARLA server
- Spawns vehicles and agents
- Runs simulation loop
- Collects metrics

```python
from test_runner import TestRunner

runner = TestRunner(host='localhost', port=2000)
runner.load_map('Town05')
runner.spawn_vehicle('test_runner/configs/default_agent.yaml')
runner.run_simulation(num_steps=1000)
runner.print_metrics()
runner.save_metrics('results/metrics.json')
runner.cleanup()
```

## Usage Examples

### Example 1: Single Agent, 12-step Test

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from test_runner import TestRunner

runner = TestRunner()
runner.load_map('Town10')
runner.spawn_vehicle('test_runner/configs/default_agent.yaml')
runner.run_simulation(num_steps=1200)
runner.print_metrics()
runner.cleanup()
```

### Example 2: Multiple Agents

```python
runner = TestRunner()
runner.load_map('Town05')

# Spawn 3 agents
for i in range(3):
    agent = runner.spawn_vehicle('test_runner/configs/default_agent.yaml')
    agent.set_target_speed(8.0 + i)  # Different speeds

runner.run_simulation(num_steps=2000)
runner.print_metrics()
runner.save_metrics('results/multi_agent_test.json')
runner.cleanup()
```

### Example 3: Custom Configuration

Create `my_config.yaml`:
```yaml
agent:
  name: "custom_agent"

camera:
  width: 1024
  height: 768
  fov: 110.0

control:
  max_speed: 25.0
  target_speed: 10.0
  pid:
    throttle:
      kp: 0.6
      ki: 0.15
      kd: 0.12
```

Then use it:
```python
runner.spawn_vehicle('my_config.yaml')
```

## Output

### Metrics File (JSON)
```json
{
  "num_steps": 1000,
  "total_distance": 5234.5,
  "collisions": 0,
  "start_time": "2026-02-28T10:30:00",
  "end_time": "2026-02-28T10:35:00",
  "agent_data": {
    "0": {
      "distance": 5234.5,
      "collisions": 0,
      "frames": 1000
    }
  }
}
```

## Command Line Options

```
python test_runner_example.py --help

optional arguments:
  --host HOST           CARLA server host (default: localhost)
  --port PORT           CARLA server port (default: 2000)
  --map MAP             CARLA map to load (default: Town05)
  --steps STEPS         Number of simulation steps (default: 1000)
  --agents AGENTS       Number of agents to spawn (default: 1)
  --weather WEATHER     Weather preset (default: ClearNoon)
  --output OUTPUT       Output metrics file path (default: results/metrics.json)
```

## Extending with VLM

To integrate Vision-Language Models:

1. **Modify `simple_agent.py`'s `_plan_speed()` method**:

```python
def _plan_speed(self, camera_image: Optional[np.ndarray]) -> float:
    """Add VLM inference here."""
    if camera_image is not None:
        # Call your VLM API
        # response = vlm_client.inference(camera_image, prompt)
        # return parse_speed(response)
        pass
    return self.target_speed
```

2. **Add VLM dependencies** to `test_runner/requirements.txt`:
```
openai>=1.0.0
# or other VLM client libraries
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure CARLA server is running on localhost:2000 |
| Out of memory | Reduce `steps` or use CARLA with `-quality-level=Low` |
| Slow simulation | Increase `frame_skip` in `run_simulation()` |
| Module not found | Ensure you're in the correct directory and `test_runner/` exists |

## Performance

- **Single Agent**: ~50-100 FPS on modern GPU
- **2 Agents**: ~30-50 FPS
- **4 Agents**: ~15-25 FPS

Times vary based on map complexity and hardware.

## Next Steps

1. ✅ Run basic single-agent test
2. 📊 Examine metrics output
3. 🔧 Customize agent config
4. 🤖 Integrate VLM for decision-making
5. 🎯 Add scenario-based tests

## Support

For issues or questions, check:
- CARLA documentation: https://carla.readthedocs.io/
- Test output logs (enable DEBUG logging)
- Example scripts in current directory
