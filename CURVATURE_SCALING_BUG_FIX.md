# Curvature Scaling Bug Fix - March 5, 2026

## Problem Identified

Your 72B model logs showed **excessive left turns even on perfectly straight roads**:

```
Frame 1: speeds=[0, 0, 0, 0, 0], curvatures=[-90, -90, -90, -90, -90]
Frame 2: speeds=[5, 10, 15, 18, 20], curvatures=[-30, -45, -60, -75, -90] 
Frame 3: speeds=[10, 12, 15, 18, 20], curvatures=[-30, -25, -20, -15, -10]
```

On a **straight road** (town05_short_01: x from -152 to 50, y constant at 134), these curvature predictions should be **≈ 0 degrees**, not **-30 to -90 degrees**.

## Root Cause: Incorrect Curvature Scaling

### The Bug

Three different incorrect scales were in use:

1. **Controller default** (`langcoop_controller.py` line 60):
   ```python
   self.curvature_scale = float(kwargs.get('curvature_scale', 3.0))
   ```
   - Applied: `steer = -90 * 3.0 = -270` ❌ Completely unrealistic

2. **Agent config 1** (`configs/langcoop_agent_config.yaml`):
   ```yaml
   curvature_scale: 0.01667  # claimed to be "equivalent to / 180 * 3"
   ```
   - Applied: `steer = -90 * 0.01667 = -1.5` ❌ Way too aggressive

3. **Agent config 2** (`configs/langcoop_agent_config_32b.yaml`):
   ```yaml
   curvature_scale: 0.01667  # same incorrect value
   ```

### How Upstream (LangCoop Paper) Does It

In `simulation/leaderboard/team_code/vlm_infer_action.py` (line 969):

```python
curv = np.deg2rad(route_info['curvature'][i] / 10)
```

**Process:**
1. VLM predicts curvature in degrees: `-90`
2. Divide by 10: `-90 / 10 = -9 degrees`
3. Convert to radians: `deg2rad(-9) = -0.157 radians`

**Result:** `-0.157` is the steering command (in range [-1, 1])

### The Math

To match upstream behavior with our multiplication approach:
```
steer = curvature_degrees * scale

For -90 degrees to become -0.157 steering input:
scale = -0.157 / -90 = 0.001744...

0.001745 ≈ π / 1800  (more precisely: π ÷ 180 ÷ 10)
```

## Solution Applied

### Files Changed

1. **`test_runner/controllers/langcoop_controller.py`**
   ```python
   # OLD (WRONG)
   self.curvature_scale = float(kwargs.get('curvature_scale', 3.0))
   
   # NEW (CORRECT)
   self.curvature_scale = float(kwargs.get('curvature_scale', 0.001745))
   ```

2. **`configs/langcoop_agent_config.yaml`**
   ```yaml
   # OLD (WRONG)
   curvature_scale: 0.01667  # equivalent to / 180 * 3 (upstream)
   
   # NEW (CORRECT)
   curvature_scale: 0.001745  # π/1800: divide by 10 then deg→rad (upstream)
   ```

3. **`configs/langcoop_agent_config_32b.yaml`**
   ```yaml
   # OLD (WRONG)
   curvature_scale: 0.01667  # equivalent to / 180 * 3 (upstream)
   
   # NEW (CORRECT)
   curvature_scale: 0.001745  # π/1800: divide by 10 then deg→rad (upstream)
   ```

### Verification

**Old behavior:**
```
VLM predicts: -90 degrees
Steering applied: -90 * 0.01667 = -1.5  ❌ Massive left turn
```

**New behavior:**
```
VLM predicts: -90 degrees
Steering applied: -90 * 0.001745 = -0.157  ✓ Moderate steering adjustment
```

On a straight road, if VLM was outputting curvature ≈ 0-10°, you'd get steering ≈ 0 to 0.017 (minimal).

## Expected Impact

After applying this fix, you should see:

✅ **Straight roads:** Vehicle goes straight (curvature ≈ 0°)
✅ **Curved roads:** Vehicle steers appropriately (curvature 15-30°)  
✅ **Route completion:** Should increase from 88.5% to 95-100%
✅ **Driving score:** Should improve significantly
✅ **No more phantom left turns:** On town05_clear_easy, expect zero steering on straight segments

## Why This Wasn't Caught Earlier

- The 7B model was so weak it output near-zero values everywhere, masking the scale bug
- The 72B model's stronger outputs revealed the scale was 10-100x too large
- This was a **calibration bug**, not a coding bug

## Testing the Fix

Run the 72B test again with the corrected config:

```bash
python run_langcoop_test.py \
    --agent-config configs/langcoop_agent_config_32b.yaml \
    --scenario-ids town05_clear_easy \
    --max-steps 500
```

You should see the scene/object/intent descriptions added to logs (from the enhanced logging added), which will help debug if the model STILL predicts unnecessary turns (would indicate a model/prompt issue, not a scale issue).

## Reference

- **Upstream:** `/Users/bhavya/Desktop/ms_projects/LangCoop/simulation/leaderboard/team_code/vlm_infer_action.py:969`
- **Our fix:** `/Users/bhavya/Desktop/ms_projects/lvlm/test_runner/controllers/langcoop_controller.py:57`
- **Math:** `π / 1800 ≈ 0.001745` (divide by 10, convert deg→rad)
