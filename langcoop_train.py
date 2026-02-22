import json, re, time, math, random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE_DTYPE = torch.float16
MAX_NEIGHBORS = 1
PACK_MAX_BYTES = 2048
STALENESS_S = 1.0

DT = 0.2
N_STEPS = 5

EGO_IMAGE_PATH = "demo_road.jpg"
NEI_IMAGE_PATH = "demo_road.jpg"

class VLM:
    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=DEVICE_DTYPE,
            trust_remote_code=True,
        ).eval()

    @torch.inference_mode()
    def gen(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 220, temperature: float = 0.2) -> str:
        msgs = [{"role": "user", "content": []}]
        if image is not None:
            msgs[0]["content"].append({"type": "image", "image": image})
        msgs[0]["content"].append({"type": "text", "text": prompt})

        inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        # mild cleanup
        if prompt in text:
            text = text.split(prompt)[-1].strip()
        return text

def m3cot(vlm: VLM, image: Image.Image, goal_text: str) -> Dict[str, str]:
    scene = vlm.gen(
        "Stage 1/4 (Scene). Describe the driving scene concisely.\n"
        "Include: road type, lanes, traffic, weather/visibility, hazards.\n"
        "Output 3-6 bullets.",
        image=image,
        max_new_tokens=160,
        temperature=0.2,
    )

    objects = vlm.gen(
        "Stage 2/4 (Objects). List interactive objects relevant for driving decisions.\n"
        "Up to 8 bullets. For each: type, relative position (L/C/R; near/mid/far), motion, risk.",
        image=image,
        max_new_tokens=220,
        temperature=0.2,
    )

    goal = vlm.gen(
        "Stage 3/4 (Goal). Convert this navigation goal into ONE clear driving objective sentence.\n"
        f"Goal input: {goal_text}\n"
        "Output exactly one sentence.",
        image=None,
        max_new_tokens=60,
        temperature=0.2,
    )

    intent = vlm.gen(
        "Stage 4/4 (Intent). Given SCENE, OBJECTS, GOAL, decide intended maneuver.\n"
        "Output exactly:\n"
        "Maneuver: <one line>\n"
        "Rationale:\n- <bullet>\n- <bullet>\n- <bullet>\n\n"
        f"SCENE:\n{scene}\n\nOBJECTS:\n{objects}\n\nGOAL:\n{goal}",
        image=None,
        max_new_tokens=220,
        temperature=0.2,
    )

    return {"scene": scene, "objects": objects, "goal": goal, "intent": intent}

@dataclass
class State:
    agent_id: str
    t: float
    x: float
    y: float
    yaw: float
    speed: float

def langpack(state: State, m3: Dict[str, str]) -> Dict[str, Any]:
    pack = {
        "state": {
            "agent_id": state.agent_id,
            "t": state.t,
            "x": state.x, "y": state.y, "yaw": state.yaw,
            "speed": state.speed,
        },
        "scene": m3["scene"],
        "objects": m3["objects"],
        "goal": m3["goal"],
        "intent": m3["intent"],
    }
    return pack

def pack_bytes(pack: Dict[str, Any]) -> int:
    return len(json.dumps(pack, ensure_ascii=False).encode("utf-8"))

def cooperative_prompt(ego_pack: Dict[str, Any], neighbor_packs: List[Dict[str, Any]]) -> str:
    payload = {"ego": ego_pack, "neighbors": neighbor_packs}
    return (
        "You are a cooperative driving planner.\n"
        "Use EGO and NEIGHBORS to choose a safe short-horizon command.\n"
        "Prefer safety and smoothness.\n"
        "CONTEXT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )

def driving_signal(vlm: VLM, ctx_prompt: str) -> Dict[str, float]:
    ask = (
        ctx_prompt
        + "\nOutput EXACTLY two lines, no extra text:\n"
          "speed_mps: <float 0..15>\n"
          "curvature: <float -0.2..0.2>\n"
    )
    text = vlm.gen(ask, image=None, max_new_tokens=60, temperature=0.0)

    m_speed = re.search(r"speed_mps:\s*([-+]?\d+(\.\d+)?)", text)
    m_curv = re.search(r"curvature:\s*([-+]?\d+(\.\d+)?)", text)
    if not (m_speed and m_curv):
        raise ValueError(f"Could not parse action.\nModel output:\n{text}")

    speed = float(m_speed.group(1))
    curv = float(m_curv.group(1))
    speed = max(0.0, min(15.0, speed))
    curv = max(-0.2, min(0.2, curv))
    return {"speed_mps": speed, "curvature": curv, "raw": text}

def step_vehicle(state: State, speed_cmd: float, curvature: float, dt: float) -> State:
    yaw_rate = speed_cmd * curvature
    yaw = state.yaw + yaw_rate * dt
    x = state.x + speed_cmd * math.cos(yaw) * dt
    y = state.y + speed_cmd * math.sin(yaw) * dt
    return State(state.agent_id, state.t + dt, x, y, yaw, speed_cmd)

def filter_neighbors(packs: List[Dict[str, Any]], now_t: float) -> List[Dict[str, Any]]:
    fresh = []
    for p in packs:
        age = now_t - p["state"]["t"]
        if age <= STALENESS_S:
            fresh.append(p)
    fresh.sort(key=lambda p: p["state"]["t"], reverse=True)
    return fresh[:MAX_NEIGHBORS]

def main():
    vlm = VLM(MODEL_NAME)

    ego_img = Image.open(EGO_IMAGE_PATH).convert("RGB")
    nei_img = Image.open(NEI_IMAGE_PATH).convert("RGB")

    ego = State("ego", t=0.0, x=0.0, y=0.0, yaw=0.0, speed=6.0)
    nei = State("nei", t=0.0, x=12.0, y=2.0, yaw=0.1, speed=5.0)

    neighbor_inbox: List[Dict[str, Any]] = []

    for k in range(N_STEPS):
        print(f"\n================ STEP {k} ================")

        ego_goal_text = "Stay in lane and proceed forward for 30 meters."
        nei_goal_text = "Maintain safe distance and be ready to yield if needed."

        ego_m3 = m3cot(vlm, ego_img, ego_goal_text)
        nei_m3 = m3cot(vlm, nei_img, nei_goal_text)

        ego_pack = langpack(ego, ego_m3)
        nei_pack = langpack(nei, nei_m3)

        if pack_bytes(ego_pack) > PACK_MAX_BYTES:
            print("WARNING: ego LangPack exceeds size limit:", pack_bytes(ego_pack))
        if pack_bytes(nei_pack) > PACK_MAX_BYTES:
            print("WARNING: nei LangPack exceeds size limit:", pack_bytes(nei_pack))

        neighbor_inbox.append(nei_pack)

        neighbors_used = filter_neighbors(neighbor_inbox, ego.t)

        ctx = cooperative_prompt(ego_pack, neighbors_used)
        action = driving_signal(vlm, ctx)

        print("ACTION RAW:\n", action["raw"])
        print("Parsed:", {"speed_mps": action["speed_mps"], "curvature": action["curvature"]})

        # Update ego state using the command (toy dynamics)
        ego = step_vehicle(ego, action["speed_mps"], action["curvature"], DT)

        # Update neighbor (toy: keep it moving forward with mild curvature)
        nei = step_vehicle(nei, speed_cmd=5.0, curvature=0.02, dt=DT)

        print("EGO state:", ego)
        print("NEI state:", nei)


if __name__ == "__main__":
    main()
