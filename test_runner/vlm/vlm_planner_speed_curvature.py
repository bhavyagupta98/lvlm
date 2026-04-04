"""
VLM Planner - Speed Curvature (CARLA 0.9.16 + Python 3.12)
Ported from LangCoop's VLMPlannerSpeedCurvature to work with modern CARLA/Python.
"""

import re
import json
import logging
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VLMPlannerSpeedCurvature:
    """
    VLM Planner using speed-curvature prediction.
    Follows LangCoop architecture with Chain-of-Thought reasoning.
    
    Compatible with CARLA 0.9.16 and Python 3.12.
    """
    
    def __init__(self, api_model_name: str, api_base_url: str, api_key: str, **kwargs):
        """
        Initialize VLM planner.
        
        Args:
            api_model_name: Model name (e.g., 'qwen2-vl-7b')
            api_base_url: API endpoint (e.g., 'http://localhost:8000/v1')
            api_key: API key (use 'EMPTY' for local vLLM)
        """
        self.api_model_name = api_model_name
        self.api_base_url = api_base_url
        self.api_key = api_key
        
        # Initialize OpenAI-compatible client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
            logger.info(f"VLM Planner initialized: {api_model_name} @ {api_base_url}")
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise
        
        self.IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
        self._zero_deadlock_streak = 0
        self._static_hazard_streak = 0
        self._last_static_hazard_side = "none"
    
    def forward(self, perception_memory_bank: List[Dict], model_config: Dict) -> List[Dict]:
        """
        Main forward pass: predict speed and curvature from perception data.
        
        Args:
            perception_memory_bank: List of historical perception frames
            model_config: Configuration with prompts and settings
            
        Returns:
            List of predicted results (one per agent)
        """
        if len(perception_memory_bank) < 2:
            # Need at least 2 frames for history
            logger.warning("Not enough frames in perception memory bank")
            return [self._get_default_prediction()]
        
        # For single agent, agent_idx = 0
        agent_idx = 0
        
        # Step 1: Get ego vehicle history
        ego_history_json = self._get_ego_history(perception_memory_bank, agent_idx)
        
        # Step 2: Chain-of-Thought reasoning
        front_image = perception_memory_bank[-1]['front_image']
        
        planning_config = model_config.get('planning', {})
        prompt_template = planning_config.get('prompt_template', {})
        prompt_usage = planning_config.get('prompt_usage', {})

        scene_description = self._get_scene_description(
            front_image, 
            prompt_template,
            prompt_usage
        )
        
        object_description = self._get_objects_description(
            front_image,
            prompt_template,
            prompt_usage
        )
        
        intent_description = self._get_intent_description(
            front_image,
            perception_memory_bank[-1]['target'][agent_idx],
            prompt_template,
            prompt_usage
        )
        
        # Step 3: Combined prediction
        target_waypoint = perception_memory_bank[-1]['target'][agent_idx]
        target_description = self._get_target_description(target_waypoint)
        
        result = self._predict_speed_curvature(
            front_image,
            scene_description,
            object_description,
            intent_description,
            ego_history_json,
            target_description,
            prompt_template,
            prompt_usage
        )
        
        return [result]

    def forward_collaborative(self, perception_memory_bank: List[Dict], model_config: Dict) -> List[Dict]:
        """Collaborative multi-agent forward pass with shared image and intent context."""
        if len(perception_memory_bank) < 2:
            logger.warning("Not enough frames in collaborative perception memory bank")
            return [self._get_default_prediction()]

        latest = perception_memory_bank[-1]
        targets = latest.get('target', [])
        num_agents = len(targets)
        if num_agents <= 0:
            return [self._get_default_prediction()]

        planning_config = model_config.get('planning', {})
        prompt_template = planning_config.get('prompt_template', {})
        prompt_usage = planning_config.get('prompt_usage', {})
        sharing_modalities = model_config.get('collab', {}).get('sharing_modalities', [])

        agent_intents = []
        for agent_idx in range(num_agents):
            front_image = self._get_agent_front_image(perception_memory_bank, agent_idx)
            target_waypoint = latest['target'][agent_idx]
            target_description = self._get_target_description(target_waypoint)
            intent_description = self._get_intent_description(
                front_image,
                target_waypoint,
                prompt_template,
                prompt_usage
            )
            agent_intents.append({
                'idx': agent_idx,
                'position': latest['detmap_pose'][agent_idx][:2],
                'intent_description': intent_description,
                'front_image': front_image,
            })

        predictions = []
        for agent_idx in range(num_agents):
            front_image = self._get_agent_front_image(perception_memory_bank, agent_idx)
            ego_history_json = self._get_ego_history(perception_memory_bank, agent_idx)
            target_waypoint = latest['target'][agent_idx]
            target_description = self._get_target_description(target_waypoint)

            scene_description = self._get_scene_description(front_image, prompt_template, prompt_usage)
            object_description = self._get_objects_description(front_image, prompt_template, prompt_usage)
            ego_intent_description = agent_intents[agent_idx]['intent_description']

            collab_description = ""
            collab_images = [front_image]
            for other_intent in agent_intents:
                if other_intent['idx'] == agent_idx:
                    continue
                relative_position = self._get_related_pos_with_direction(
                    ego_pos=latest['detmap_pose'][agent_idx][:2],
                    ego_yaw=latest['ego_yaw'][agent_idx],
                    positions=other_intent['position']
                )
                position_list = [round(float(coord), 5) for coord in np.asarray(relative_position).tolist()]
                image_marker = self.IMAGE_PLACEHOLDER if 'image' in sharing_modalities else 'not shared'
                intent_text = other_intent['intent_description'] if 'intent' in sharing_modalities else ''
                collab_description += (
                    f"Agent {other_intent['idx']}, located at: {position_list}, "
                    f"intent description: {intent_text}, image: {image_marker}\n"
                )
                if 'image' in sharing_modalities:
                    collab_images.append(other_intent['front_image'])

            result = self._predict_speed_curvature(
                collab_images,
                scene_description,
                object_description,
                ego_intent_description,
                ego_history_json,
                target_description,
                prompt_template,
                prompt_usage,
                collab_agent_description=collab_description,
            )
            predictions.append(result)

        return predictions

    def _get_agent_front_image(self, perception_memory_bank: List[Dict], agent_idx: int) -> np.ndarray:
        """Retrieve one agent's front image from shared or single-agent memory."""
        front_image = perception_memory_bank[-1]['front_image']
        if isinstance(front_image, list):
            return front_image[agent_idx]
        if isinstance(front_image, np.ndarray) and front_image.ndim == 4:
            return front_image[agent_idx]
        return front_image

    def _get_related_pos_with_direction(self, ego_pos, ego_yaw: float, positions):
        """Convert other-agent global position into ego-relative coordinates."""
        if isinstance(ego_pos, torch.Tensor):
            ego_pos = ego_pos.detach().cpu().numpy()
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()

        ego_pos = np.asarray(ego_pos, dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)
        relative_global_pos = positions - ego_pos

        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        rotation_matrix = np.array([
            [sin_yaw, cos_yaw],
            [-cos_yaw, sin_yaw],
        ])
        return relative_global_pos @ rotation_matrix.T

    def _resolve_prompt(self, prompt_template: Dict, prompt_usage: Dict,
                        candidates: List[str], usage_candidates: List[str],
                        fallback: str) -> str:
        """Resolve prompt text from multiple possible keys and value shapes."""
        if not isinstance(prompt_template, dict):
            return fallback

        desired_variant = None
        if isinstance(prompt_usage, dict):
            for usage_key in usage_candidates:
                usage_value = prompt_usage.get(usage_key)
                if isinstance(usage_value, str) and usage_value.strip():
                    desired_variant = usage_value.strip()
                    break

        for key in candidates:
            if key not in prompt_template:
                continue

            value = prompt_template[key]

            if isinstance(value, str) and value.strip():
                return value

            if isinstance(value, dict):
                if desired_variant:
                    subval = value.get(desired_variant)
                    if isinstance(subval, str) and subval.strip():
                        return subval

                for subkey in ("concise", "default", "text"):
                    subval = value.get(subkey)
                    if isinstance(subval, str) and subval.strip():
                        return subval

                for subval in value.values():
                    if isinstance(subval, str) and subval.strip():
                        return subval

        return fallback
    
    def _get_ego_history(self, perception_memory_bank: List[Dict], agent_idx: int) -> str:
        """
        Build JSON string with ego vehicle history.
        
        Args:
            perception_memory_bank: List of perception frames
            agent_idx: Agent index
            
        Returns:
            JSON string with ego history
        """
        ego_history_list = []
        prev_position = None
        prev_yaw = None
        dt = 0.5  # 0.5 seconds between frames
        
        # Current position as reference
        curr_pose = perception_memory_bank[-1]['detmap_pose'][agent_idx]
        curr_x, curr_y = float(curr_pose[0]), float(curr_pose[1])
        
        # Process historical frames
        for frame_data in perception_memory_bank[:-1]:
            timestamp = frame_data['timestamp']
            ego_pose = frame_data['detmap_pose'][agent_idx]
            x, y = float(ego_pose[0]), float(ego_pose[1])
            yaw = float(frame_data['ego_yaw'][agent_idx])
            
            # Calculate speed
            if prev_position is not None:
                dx = x - prev_position[0]
                dy = y - prev_position[1]
                speed = np.sqrt(dx * dx + dy * dy) / dt
            else:
                speed = 0.0
            
            # Calculate curvature
            if prev_yaw is not None:
                d_yaw = yaw - prev_yaw
                distance = max(1e-6, speed * dt)
                curvature = d_yaw / distance
            else:
                curvature = 0.0
            
            record = {
                "timestamp": timestamp,
                "speed": round(speed, 3),
                "curvature": round(curvature, 3),
                "waypoints": [round(x - curr_x, 2), round(y - curr_y, 2)]
            }
            ego_history_list.append(record)
            prev_position = (x, y)
            prev_yaw = yaw
        
        ego_history_json = {"ego_history": ego_history_list}
        return json.dumps(ego_history_json, indent=2)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64."""
        if image is None:
            raise ValueError("Image is None")
        
        # Image validation
        logger.debug(f"[IMAGE VALIDATION] Shape: {image.shape}, dtype: {image.dtype}")
        logger.debug(f"[IMAGE VALIDATION] Value range: [{image.min()}, {image.max()}], mean: {image.mean():.1f}")
        
        # Check if image looks blank (all same values)
        if image.std() < 1.0:
            logger.warning(f"[IMAGE VALIDATION] WARNING: Image may be blank (std={image.std():.3f})")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.debug(f"[IMAGE VALIDATION] Encoded size: {len(img_base64)} bytes")
        
        return img_base64
    
    def _get_scene_description(self, image: np.ndarray, prompt_template: Dict,
                               prompt_usage: Dict) -> str:
        """Get scene description using CoT."""
        try:
            img_base64 = self._encode_image(image)

            scene_prompt = self._resolve_prompt(
                prompt_template,
                prompt_usage,
                ["scene", "scene_prompt_template"],
                ["scene_prompt_template", "scene", "scene_prompt"],
                (
                    "Describe the driving scenario, including lane geometry, road direction, weather, traffic, and whether the drivable path continues straight or bends. "
                    "Explicitly state lane boundaries and roadside structures (wall, barrier, curb, guardrail, fence) and whether any are intruding into the drivable path. "
                    "Classify road geometry as straight, gentle-left, gentle-right, sharp-left, or sharp-right and mention drift risk (left/right/none)."
                )
            )
            scene_prompt = scene_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": scene_prompt
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.debug(f"Scene: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Scene description failed: {e}")
            return "Clear conditions, standard road layout."
    
    def _get_objects_description(self, image: np.ndarray, prompt_template: Dict,
                                 prompt_usage: Dict) -> str:
        """Get object detection using CoT."""
        try:
            img_base64 = self._encode_image(image)

            object_prompt = self._resolve_prompt(
                prompt_template,
                prompt_usage,
                ["objects", "object_prompt_template", "default"],
                ["object_prompt_template", "objects", "object_prompt"],
                (
                    "Identify only traffic-relevant road users or obstacles that could affect the ego vehicle soon. "
                    "Include static hazards such as wall, barrier, curb, guardrail, fence, parked vehicle, cone, debris, or blocked lane edge. "
                    "Do not mark curb/guardrail/fence/wall as lane-blocking if they remain outside lane boundaries and run parallel to the road. "
                    "For each item, provide side (left/right/center), approximate distance (near/mid/far), and whether it intrudes into the ego lane. "
                    "List two or three highest-risk items."
                )
            )
            object_prompt = object_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": object_prompt
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.debug(f"Objects: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Object description failed: {e}")
            return "No significant objects detected."
    
    def _get_intent_description(self, image: np.ndarray, target: List[float], 
                                prompt_template: Dict, prompt_usage: Dict) -> str:
        """Get driving intent using CoT."""
        try:
            img_base64 = self._encode_image(image)
            
            # Format target description
            target_desc = self._get_target_description(target)

            intent_prompt = self._resolve_prompt(
                prompt_template,
                prompt_usage,
                ["intent", "intention_prompt_template", "intent_prompt_template", "default"],
                ["intention_prompt_template", "intent_prompt_template", "intent", "intention"],
                (
                    "You are planning in ego-relative coordinates. The target description states lateral offset "
                    "(left or right) and longitudinal offset (front or back). If the target is mostly ahead and the "
                    "lateral offset is small, keep going straight with only slight steering corrections. Prefer staying "
                    "centered in the lane over aggressively cutting toward the waypoint. Should you turn left, turn right, "
                    "go straight, slightly adjust direction, accelerate, or decelerate? Describe how you would navigate "
                    "the vehicle to reach the target safely."
                )
            )
            intent_prompt = intent_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            intent_prompt = intent_prompt.replace("{target_description}", target_desc)
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": intent_prompt
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.debug(f"Intent: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Intent description failed: {e}")
            return "Maintain current course and speed."
    
    def _get_target_description(self, target: List[float]) -> str:
        """Format target waypoint description."""
        x = float(target[0])
        y = -float(target[1])

        lateral_distance = abs(round(x, 5))
        longitudinal_distance = abs(round(y, 5))

        if lateral_distance < 1.0:
            lateral_phrase = "approximately centered in your lane"
        else:
            lateral_direction = "right" if x > 0 else "left"
            lateral_phrase = f"{lateral_distance:.1f} meters to your {lateral_direction}"

        if longitudinal_distance < 1.0:
            longitudinal_phrase = "roughly level with your current position"
        else:
            longitudinal_direction = "front" if y > 0 else "back"
            longitudinal_phrase = f"{longitudinal_distance:.1f} meters to your {longitudinal_direction}"

        return (
            "The target waypoint is given in ego-relative coordinates: lateral offset means left/right and "
            f"longitudinal offset means front/back. The target is {lateral_phrase} and {longitudinal_phrase}."
        )
    
    def _predict_speed_curvature(self, image, scene_desc: str,
                                 object_desc: str, intent_desc: str,
                                 ego_history: str, target_desc: str,
                                 prompt_template: Dict, prompt_usage: Dict,
                                 collab_agent_description: str = "") -> Dict:
        """
        Final prediction step: combine all context to predict speed and curvature.
        
        Returns:
            Dict with 'target_speed' and 'curvature' arrays
        """
        try:
            if isinstance(image, list):
                images = image
            else:
                images = [image]

            encoded_images = [self._encode_image(img) for img in images]

            # Combined prompt - use ORIGINAL placeholder names
            comb_prompt = self._resolve_prompt(
                prompt_template,
                prompt_usage,
                ["prediction", "comb_prompt", "combined_prompt"],
                ["comb_prompt", "prediction", "prediction_prompt"],
                (
                    "You are an autonomous driving vehicle controller. "
                    "You have access to a front-view camera image. <IMAGE_PLACEHOLDER>\n"
                    "Here is the environment description and detected objects:\n"
                    "- Scene: {scene_description}\n"
                    "- Objects and intents: {object_description}\n"
                    "- Ego vehicle history: {ego_history_prompt}\n"
                    "- Intent of the ego vehicle: {intent_description}\n"
                    "- Collaborative agents' information are described as follows: {collab_agent_description}\n"
                    "{target_description}\n"
                    "Generate the vehicle's desired speed and curvature for the next 5 timestamps, ensuring safe and efficient movement towards the target.\n"
                    "Use smooth but decisive control. First estimate road curvature from lane boundaries over the next 20-40 meters. "
                    "Map geometry to curvature magnitude: straight near zero, gentle turns small consistent curvature, sharp/intersection turns larger curvature. "
                    "On straight roads, or when the target is mostly ahead with a small lateral offset, keep curvature close to 0. "
                    "Only use large curvature for clear bends, intersection turns, or obstacle avoidance. "
                    "If drifting right on a straight road, apply slight left correction; if drifting left, apply slight right correction. "
                    "If uncertain or visually ambiguous, prefer near-zero curvature instead of persistent one-sided steering.\n"
                    "- Speed (m/s): Range [0, 20], integer or decimal values\n"
                    "- Curvature (degree/m): Range [-180, 180], integer or decimal values\n"
                    "- Negative curvature = turning left (counter-clockwise)\n"
                    "- Positive curvature = turning right (clockwise)\n"
                    "- Zero curvature = going straight ahead\n"
                    "- Keep the vehicle centered in the drivable lane and do not steer toward the shoulder or sidewalk.\n"
                    "- Ensure traffic rule compliance and avoid collisions\n"
                    "- Avoid collisions by slowing down or steering away from obstacles.\n"
                    'Output MUST be a valid JSON structure with the key "predicted_speeds_curvatures" containing a list of 5 [speed, curvature] pairs.\n'
                    "```json\n"
                    "{\n"
                    '    "predicted_speeds_curvatures": [\n'
                    "        [speed_1, curvature_1],\n"
                    "        [speed_2, curvature_2],\n"
                    "        [speed_3, curvature_3],\n"
                    "        [speed_4, curvature_4],\n"
                    "        [speed_5, curvature_5]\n"
                    "    ]\n"
                    "}\n"
                    "```\n"
                    "No additional text outside of this JSON format."
                )
            )
            comb_prompt = comb_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            
            # Log the values being replaced
            logger.info(f"[REPLACEMENT DEBUG]")
            logger.info(f"  scene_desc exists: {bool(scene_desc)}, len={len(scene_desc) if scene_desc else 0}")
            logger.info(f"  object_desc exists: {bool(object_desc)}, len={len(object_desc) if object_desc else 0}")
            logger.info(f"  ego_history exists: {bool(ego_history)}, len={len(ego_history) if ego_history else 0}")
            logger.info(f"  intent_desc exists: {bool(intent_desc)}, len={len(intent_desc) if intent_desc else 0}")
            logger.info(f"  target_desc exists: {bool(target_desc)}, len={len(target_desc) if target_desc else 0}")
            
            # Log actual descriptions for debugging VLM confusion
            logger.info(f"[SCENE DESCRIPTION] {scene_desc[:200]}")
            logger.info(f"[OBJECT DESCRIPTION] {object_desc[:200]}")
            logger.info(f"[INTENT DESCRIPTION] {intent_desc[:200]}")
            logger.info(f"[EGO HISTORY] {ego_history[:200]}")
            
            comb_prompt = comb_prompt.replace("{scene_description}", scene_desc)
            comb_prompt = comb_prompt.replace("{object_description}", object_desc)  # SINGULAR
            comb_prompt = comb_prompt.replace("{ego_history_prompt}", ego_history)    # WITH _prompt
            comb_prompt = comb_prompt.replace("{intent_description}", intent_desc)
            comb_prompt = comb_prompt.replace("{target_description}", target_desc)
            comb_prompt = comb_prompt.replace("{collab_agent_description}", collab_agent_description)
            
            # Log the prepared prompt for debugging
            logger.debug(f"[PROMPT PREPARED]\n{comb_prompt}\n[END PROMPT]")
            
            content = []
            for img_base64 in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
            content.append({
                "type": "text",
                "text": comb_prompt
            })

            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                max_tokens=512,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"VLM Response (full): {result_text}")
            
            # Parse JSON result
            parsed_result = self._parse_speed_curvature_response(result_text)

            # Track consecutive zero predictions
            speeds = parsed_result.get('target_speed', [])
            all_zero = speeds and all(abs(float(s)) < 1e-3 for s in speeds)
            
            if all_zero:
                self._zero_deadlock_streak += 1
            else:
                self._zero_deadlock_streak = 0
            
            # Guard against deadlock: if model outputs all zeros while intent is to accelerate
            # and scene appears clear, apply an escalating forward rollout.
            # Never override if intent explicitly requires stopping.
            intent_text = (intent_desc or '').lower()
            explicit_stop_intent = any(token in intent_text for token in 
                                      ("stop", "decelerate", "brake", "slow down", "halt", "wait"))
            
            should_override = self._should_override_zero_prediction(
                parsed_result, scene_desc, object_desc, intent_desc
            )
            
            # Apply override only if:
            # 1. Clear-road + accelerate conditions met, OR
            # 2. Streak >= 3 BUT no explicit stop intent
            if should_override and self._zero_deadlock_streak > 0:
                parsed_result = self._get_zero_deadlock_override(self._zero_deadlock_streak)
                logger.warning(
                    "Applied zero-speed deadlock override | streak=%d | applied_speed=%.2f | reason=clear-road",
                    self._zero_deadlock_streak,
                    float(parsed_result['target_speed'][0])
                )
            elif self._zero_deadlock_streak >= 3 and not explicit_stop_intent:
                parsed_result = self._get_zero_deadlock_override(self._zero_deadlock_streak)
                logger.warning(
                    "Applied zero-speed deadlock override | streak=%d | applied_speed=%.2f | reason=persistent-deadlock",
                    self._zero_deadlock_streak,
                    float(parsed_result['target_speed'][0])
                )

            # Generic safety guard: if text context indicates a nearby static obstacle but
            # planned control is nearly straight/fast, inject a mild evasive profile.
            guarded_result, guard_applied, guard_reason, guard_signals = self._apply_static_obstacle_safety_guard(
                parsed_result,
                scene_desc,
                object_desc,
                intent_desc
            )
            if guard_applied:
                parsed_result = guarded_result
                logger.warning(
                    "Applied static-obstacle safety guard | reason=%s | speed=%.2f | curvature=%.2f",
                    guard_reason,
                    float(parsed_result['target_speed'][0]),
                    float(parsed_result['curvature'][0])
                )

            parsed_result['_debug'] = {
                'scene_description': scene_desc,
                'object_description': object_desc,
                'intent_description': intent_desc,
                'target_description': target_desc,
                'scene_description_length': len(scene_desc or ""),
                'object_description_length': len(object_desc or ""),
                'intent_description_length': len(intent_desc or ""),
                'collab_agent_description': collab_agent_description,
                'zero_deadlock_streak': int(self._zero_deadlock_streak),
                'zero_override_candidate': bool(should_override),
                'static_guard_applied': bool(guard_applied),
                'static_guard_reason': guard_reason,
                'hazard_signals': guard_signals,
            }

            logger.info(f"Predicted: speed={parsed_result['target_speed'][0]:.2f} m/s, "
                       f"curvature={parsed_result['curvature'][0]:.3f} degrees")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Speed-curvature prediction failed: {e}")
            return self._get_default_prediction()
    
    def _parse_speed_curvature_response(self, response_text: str) -> Dict:
        """Parse VLM response to extract speed-curvature pairs."""
        try:
            # Log FULL response for debugging
            logger.debug(f"[VLM FULL RESPONSE]\n{response_text}\n[END RESPONSE]")
            
            # Extract JSON from response - try multiple patterns
            json_patterns = [
                (re.compile(r"```json\s*([\s\S]*?)\s*```", re.MULTILINE), "json_block"),
                (re.compile(r"\{[\s\S]*\}", re.MULTILINE), "json_object"),
            ]
            
            json_str = None
            for pattern, pname in json_patterns:
                json_match = pattern.search(response_text)
                if json_match:
                    if json_match.lastindex and json_match.group(1):
                        json_str = json_match.group(1)
                    else:
                        json_str = json_match.group(0)
                    logger.info(f"[JSON EXTRACTION] Method: {pname}")
                    logger.debug(f"Extracted JSON: {json_str[:200]}...")
                    break
            
            if not json_str:
                logger.warning("[JSON EXTRACTION] No JSON pattern matched!")
            
            if json_str:
                # Strip C-style comments (// ...) from JSON string
                json_str = re.sub(r'//.*?(?=[\n,\]])', '', json_str)
                
                json_result = json.loads(json_str)
                logger.info(f"[JSON PARSED] Keys: {list(json_result.keys()) if isinstance(json_result, dict) else 'list'}")
                
                # Handle various JSON response structures
                pairs = None
                if isinstance(json_result, dict):
                    if 'predicted_speeds_curvatures' in json_result:
                        pairs = json_result['predicted_speeds_curvatures']
                        logger.info(f"[PAIRS SOURCE] predicted_speeds_curvatures")
                    elif 'predictions' in json_result:
                        pairs = json_result['predictions']
                        logger.info(f"[PAIRS SOURCE] predictions")
                    elif 'speeds_curvatures' in json_result:
                        pairs = json_result['speeds_curvatures']
                        logger.info(f"[PAIRS SOURCE] speeds_curvatures")
                    else:
                        logger.warning(f"[PAIRS SOURCE] No recognized key. Available: {list(json_result.keys())}")
                elif isinstance(json_result, list):
                    pairs = json_result
                    logger.info(f"[PAIRS SOURCE] Direct list")
                
                if pairs:
                    try:
                        logger.info(f"[EXTRACTION] Extracting from {len(pairs)} pairs: {pairs}")
                        speeds = [float(pair[0]) for pair in pairs]
                        curvatures = [float(pair[1]) for pair in pairs]
                        
                        logger.info(f"[RAW VALUES] speeds={speeds}, curvatures={curvatures}")
                        
                        # Clamp to reasonable ranges
                        # Speeds: [0, 20] m/s
                        # Curvatures: [-180, 180] degrees (steering angle range)
                        speeds = [max(0.0, min(20.0, s)) for s in speeds]
                        curvatures = [max(-180.0, min(180.0, c)) for c in curvatures]
                        
                        logger.info(f"Successfully parsed: speeds={speeds}, curvatures={curvatures} degrees")
                        
                        return {
                            'target_speed': speeds,
                            'curvature': curvatures,
                            'dt': 0.5
                        }
                    except (ValueError, IndexError, TypeError) as e:
                        logger.warning(f"Failed to extract speed/curvature from pairs: {e}")
            
            # Fallback: try to extract numbers as [s0, c0, s1, c1, ...]
            logger.debug("JSON extraction failed, trying regex number extraction...")
            numbers = re.findall(r'[-+]?\d*\.?\d+', response_text)
            logger.debug(f"Extracted numbers: {numbers}")
            
            if len(numbers) >= 10:  # At least 5 pairs
                try:
                    speeds = [float(numbers[i*2]) for i in range(5)]
                    curvatures = [float(numbers[i*2+1]) for i in range(5)]
                    
                    speeds = [max(0.0, min(20.0, s)) for s in speeds]
                    curvatures = [max(-180.0, min(180.0, c)) for c in curvatures]
                    
                    logger.info(f"Extracted via regex: speeds={speeds}, curvatures={curvatures} degrees")
                    
                    return {
                        'target_speed': speeds,
                        'curvature': curvatures,
                        'dt': 0.5
                    }
                except (ValueError, IndexError) as e:
                    logger.warning(f"Regex extraction failed: {e}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing VLM response: {e}", exc_info=True)
        
        logger.warning("Returning default prediction due to parse failure")
        return self._get_default_prediction()

    def _should_override_zero_prediction(self, prediction: Dict, scene_desc: str, object_desc: str, intent_desc: str) -> bool:
        """Return True when all-zero prediction likely indicates VLM deadlock on a clear road."""
        speeds = prediction.get('target_speed', [])
        curvatures = prediction.get('curvature', [])
        if not speeds or not curvatures:
            return False

        all_zero = all(abs(float(s)) < 1e-3 for s in speeds) and all(abs(float(c)) < 1e-3 for c in curvatures)
        if not all_zero:
            return False

        intent_text = (intent_desc or '').lower()
        accelerate_intent = any(token in intent_text for token in ("accelerate", "speed up", "go straight"))
        if not accelerate_intent:
            return False

        scene_text = (scene_desc or '').lower()
        object_text = (object_desc or '').lower()

        clear_scene_markers = [
            "no visible traffic",
            "no vehicles",
            "empty road",
            "clear road",
            "no pedestrians"
        ]
        stop_markers = ["stop sign", "red light", "traffic light"]

        scene_looks_clear = any(marker in scene_text for marker in clear_scene_markers)
        hard_stop_cue = any(marker in scene_text for marker in stop_markers)

        # If object description only talks about static/non-traffic entities,
        # treat as likely hallucinated hazard content for control purposes.
        non_traffic_only = any(token in object_text for token in ("birds", "trees", "streetlights")) and not any(
            token in object_text for token in ("vehicle", "car", "truck", "pedestrian", "cyclist")
        )

        return accelerate_intent and scene_looks_clear and (non_traffic_only or not hard_stop_cue)

    def _get_zero_deadlock_override(self, streak: int) -> Dict:
        """Escalating fallback to break persistent zero-speed deadlocks."""
        if streak <= 1:
            speeds = [2.0, 3.5, 5.0, 5.0, 5.0]
        elif streak == 2:
            speeds = [3.5, 5.0, 6.5, 7.0, 7.0]
        else:
            speeds = [5.0, 6.5, 8.0, 8.0, 8.0]

        return {
            'target_speed': speeds,
            'curvature': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dt': 0.5
        }

    def _apply_static_obstacle_safety_guard(self, prediction: Dict, scene_desc: str,
                                            object_desc: str, intent_desc: str):
        """Apply a generic evasive fallback when static obstacles are likely near ego lane."""
        speeds = prediction.get('target_speed', [])
        curvatures = prediction.get('curvature', [])
        if not speeds or not curvatures:
            return prediction, False, "empty_prediction", {
                'has_static_hazard': False,
                'has_proximity_cue': False,
                'has_dynamic_conflict': False,
                'hazard_side': 'none'
            }

        scene_text = (scene_desc or '').lower()
        object_text = (object_desc or '').lower()
        intent_text = (intent_desc or '').lower()
        combined = f"{scene_text} {object_text}"

        static_tokens = (
            "wall", "barrier", "guardrail", "curb", "fence", "bollard", "concrete",
            "parked vehicle", "parked car", "cone", "debris", "block", "blocked"
        )
        proximity_tokens = ("near", "close", "immediate", "very near", "adjacent")
        dynamic_tokens = ("pedestrian", "cyclist", "vehicle", "truck", "motorcycle", "car ahead")
        lane_intrusion_tokens = (
            "intrudes into ego lane",
            "in ego lane",
            "blocks ego lane",
            "blocking ego lane",
            "blocks lane center",
            "blocking lane center",
            "occupies lane",
            "lane blocked"
        )

        has_static_hazard = any(tok in combined for tok in static_tokens)
        has_proximity_cue = any(tok in combined for tok in proximity_tokens)
        has_dynamic_conflict = any(tok in combined for tok in dynamic_tokens)
        has_lane_intrusion = any(tok in combined for tok in lane_intrusion_tokens)

        # Keep this guard focused on static obstacle misses; do not override strong
        # dynamic-object intent that is already handled by the VLM output.
        left_tokens = ("left wall", "left barrier", "left curb", "left guardrail", "left side")
        right_tokens = ("right wall", "right barrier", "right curb", "right guardrail", "right side")
        left_detected = any(tok in combined for tok in left_tokens)
        right_detected = any(tok in combined for tok in right_tokens)

        hazard_side = 'none'
        if left_detected and not right_detected:
            hazard_side = 'left'
        elif right_detected and not left_detected:
            hazard_side = 'right'
        elif left_detected and right_detected:
            hazard_side = 'ambiguous'
        elif has_static_hazard:
            hazard_side = 'unspecified'

        signals = {
            'has_static_hazard': bool(has_static_hazard),
            'has_proximity_cue': bool(has_proximity_cue),
            'has_dynamic_conflict': bool(has_dynamic_conflict),
            'has_lane_intrusion': bool(has_lane_intrusion),
            'hazard_side': hazard_side,
        }

        if hazard_side == 'ambiguous':
            self._static_hazard_streak = 0
            self._last_static_hazard_side = 'none'
            return prediction, False, "ambiguous_hazard_side", signals

        if not has_static_hazard or not has_proximity_cue or has_dynamic_conflict or not has_lane_intrusion:
            self._static_hazard_streak = 0
            self._last_static_hazard_side = 'none'
            return prediction, False, "not_applicable", signals

        if hazard_side != self._last_static_hazard_side:
            self._static_hazard_streak = 1
            self._last_static_hazard_side = hazard_side
        else:
            self._static_hazard_streak += 1

        signals['hazard_streak'] = int(self._static_hazard_streak)

        if self._static_hazard_streak < 1:
            return prediction, False, "awaiting_temporal_confirmation", signals

        current_speed = float(speeds[0])
        current_curv = float(curvatures[0])
        mostly_straight = abs(current_curv) < 1.5
        too_fast_for_hazard = current_speed > 5.0 or float(speeds[0]) > 6.0

        if not (mostly_straight and too_fast_for_hazard):
            return prediction, False, "control_already_cautious", signals

        if any(tok in combined for tok in ("left wall", "left barrier", "left curb", "left side")):
            turn_dir = 1.0   # right
            reason = "static_hazard_left"
        elif any(tok in combined for tok in ("right wall", "right barrier", "right curb", "right side")):
            turn_dir = -1.0  # left
            reason = "static_hazard_right"
        elif "turn left" in intent_text:
            turn_dir = -1.0
            reason = "intent_left_with_static_hazard"
        elif "turn right" in intent_text:
            turn_dir = 1.0
            reason = "intent_right_with_static_hazard"
        else:
            turn_dir = 1.0
            reason = "static_hazard_unspecified_side"

        guarded = {
            'target_speed': [min(current_speed, 7.0), 6.5, 6.0, 6.5, 7.0],
            'curvature': [2.0 * turn_dir, 3.0 * turn_dir, 2.5 * turn_dir, 2.0 * turn_dir, 1.5 * turn_dir],
            'dt': prediction.get('dt', 0.5)
        }
        return guarded, True, reason, signals
    
    def _get_default_prediction(self) -> Dict:
        """Default prediction when VLM fails."""
        return {
            'target_speed': [8.0, 8.5, 9.0, 9.5, 10.0],
            'curvature': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dt': 0.5
        }
    
    def to(self, device):
        """Compatibility method for device placement."""
        return self
    
    def eval(self):
        """Compatibility method for eval mode."""
        return self
