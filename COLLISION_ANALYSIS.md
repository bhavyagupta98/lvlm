# LangCoop Collision Analysis & Fixes Applied

## Issues Found (From Log Analysis)

### 1. **CRITICAL: Curvature Unit Mismatch** ❌ FIXED ✅
**Location**: `test_runner/vlm/vlm_planner_speed_curvature.py:546`

**The Problem**:
- Prompt told VLM: "Curvature (degree/m): Range [-180, 180]"
- Code clamped to: `[-0.5, 0.5]` (radians)
- **Result**: Massive unit mismatch causing no steering!

**What Happened**:
- VLM  tried to learn but predictions got clamped to near-zero
- Vehicle got 0 curvature → **NO STEERING** → Drove straight into walls

**APPLIED FIX**:
✅ Updated prompt in both files to ask for **radians [-0.5, 0.5]**:
- `configs/langcoop_agent_config.yaml` (line 90-96)
- `test_runner/vlm/vlm_planner_speed_curvature.py` (line 373-381)

Changed from:
```
- Curvature (degree/m): Range [-180, 180]
- Negative curvature = turning left
- Positive curvature = turning right
```

Changed to:
```
- Curvature (rad/m): Range [-0.5, 0.5], decimal values (e.g., -0.3, 0.0, 0.2)
- Negative curvature = turning left (counter-clockwise)
- Positive curvature = turning right (clockwise)
- Zero curvature = going straight ahead
```

---

### 2. **Images Might Be Blank or Corrupted** ⚠️ DIAGNOSTICS ADDED ✅
**Evidence from logs**:
```
[EXTRACTION] Extracting from 5 pairs: [[2, 0], [2, 0], [2, 0], [2, 0], [2, 0]]
[EXTRACTION] Extracting from 5 pairs: [[20, 0], [20, 0], [20, 0], [20, 0], [20, 0]]
```

**Problem**:
- All curvatures are 0.0 - zero steering predicted
- VLM cannot understand the scene properly if image is blank

**APPLIED FIX**:
✅ Added image validation logging in `_encode_image()`:
```python
[IMAGE VALIDATION] Shape: (600, 800, 3), dtype: uint8
[IMAGE VALIDATION] Value range: [0, 255], mean: 127.5
[IMAGE VALIDATION] Encoded size: 350000 bytes
```

If you see:
- **std < 1.0**: Image is blank/uniform → Camera not capturing properly
- **Mean ~127, Std ~50**: Image looks normal
- **Encoded size ~350KB**: Good PNG compression = real image data

Run with DEBUG logging to see these messages:
```bash
# Add to start of run_langcoop_test.py
logging.basicConfig(level=logging.DEBUG)
```

---

### 3. **CARLA Control Synchronization Issue** ⏱️
**Location**: `run_langcoop_test.py:254-260`

```python
for step in range(max_steps):
    self.world.tick()
    
    # Agent step only every 4 frames!
    if step % skip_frames == 0:  # skip_frames = 4
        control = self.agent.step()
        self.vehicle.apply_control(control)
```

**Implication**:
- Decision every 0.2s (at 20 Hz): 5 decisions per second
- For fast-moving obstacles, might be too slow for avoidance
- Consider reducing `skip_frames` from 4 to 2 if collisions persist

---

## Diagnostic Checklist

### Run the debug script:
```bash
python debug_analysis.py
```

This tests:
- ✓ Image encoding/decoding
- ✓ CARLA connection  
- ✓ vLLM API & model loading
- ✓ Sensor data capture + control application

### Expected output after fixes:
```
[IMAGE VALIDATION]  Shape: (600, 800, 3)
[IMAGE VALIDATION] Value range: [0, 255]
[IMAGE VALIDATION] Mean: 120-130 (not uniform)
[RAW VALUES] curvatures=[-0.3, -0.2, 0.0, 0.1, ...] (should have variation!)
[Successfully parsed] curvatures=[...] rad/m (with mix of +/-)
```

### If still seeing all-zero curvatures:
1. Check if image is blank: `[IMAGE VALIDATION] std < 1.0`?
   - If yes: Camera not working, check sensor setup
2. Check vLLM logs: Is it actually receiving the image_url?
3. Try sending a simple test query to vLLM with an image
4. Check if 7B model is actually loaded: `python debug_analysis.py` → vLLM API section

---

## Summary of Changes

| File | Change | Reason |
|------|--------|--------|
| `configs/langcoop_agent_config.yaml` | Updated curvature prompt from degrees to radians [-0.5, 0.5] | Fix unit mismatch |
| `test_runner/vlm/vlm_planner_speed_curvature.py` | Same curvature prompt fix | Fix default fallback prompt |
| `test_runner/vlm/vlm_planner_speed_curvature.py` | Added image validation logging in `_encode_image()` | Detect blank/corrupt images |
| `debug_analysis.py` (new) | Created comprehensive diagnostic script | Test all components |

---

## Next Steps

1. **Test the fixes**:
   ```bash
   python debug_analysis.py
   ```

2. **Check for blank images**:
   - Run test with DEBUG logging level
   - Look for `[IMAGE VALIDATION]` messages
   - Check if `std < 1.0` (blank image) or normal distribution

3. **If curvatures still all-zero**:
   - Verify vLLM is serving 7B model (not 3B)
   - Check vLLM server logs for image processing errors
   - Try a manual test query to vLLM with an encoded image

4. **If collisions continue**:
   - Reduce `skip_frames` from 4 to 2 in `run_langcoop_test.py`
   - Lower target speed in config
   - Test on simpler map (Town03 vs Town05)
