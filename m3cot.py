import base64
import mimetypes
from pathlib import Path

from openai import OpenAI
from PIL import Image


def image_file_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p.resolve()}")

    mime_type, _ = mimetypes.guess_type(str(p))
    if mime_type is None:
        mime_type = "image/png"

    image_bytes = p.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def quick_sanity_check_image(image_path: str) -> None:
    with Image.open(image_path) as img:
        img.load()
        print(f"[ok] Loaded image: {image_path}")
        print(f"     Format: {img.format}, Size: {img.size}, Mode: {img.mode}")


def call_lvlm_with_image(prompt: str, image_path: str, model: str = "gpt-4o-mini") -> str:
    """
    Common helper: takes a prompt + local image, sends to LVLM, returns text.
    """
    client = OpenAI()
    data_url = image_file_to_data_url(image_path)

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return resp.output_text.strip()


def call_llm_text_only(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Helper for prompts that are text-only (no image).
    """
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
    )
    return resp.output_text.strip()


def m3cot_stage1_scene_description(image_path: str, model: str = "gpt-4o-mini") -> str:
    """
    M3CoT Stage 1: Driving Scene Description
    Output: weather/lighting + road type/condition + traffic density/context.
    """
    prompt = (
        "Driving Scene Description:\n"
        "Describe the driving scenario in this front-view image, including:\n"
        "1) weather/lighting, 2) road type/condition, 3) traffic density and notable context.\n"
        "Be very concise (2-4 sentences). Avoid guessing invisible details."
    )
    return call_lvlm_with_image(prompt, image_path, model=model)


def m3cot_stage2_interactive_objects(image_path: str, model: str = "gpt-4o-mini") -> str:
    """
    M3CoT Stage 2: Interactive Objects Description
    Output: important road users + relative location + status + likely intent (if clear).
    """
    prompt = (
        "Interactive Objects Description:\n"
        "Identify important road users in this driving scene that could affect the ego vehicle.\n"
        "List only the most relevant objects (typically 1-6). For each object, provide:\n"
        "- Type (car/truck/motorcycle/pedestrian/cyclist/traffic light/obstacle if visible)\n"
        "- Relative location (e.g., front-center, left lane ahead, right shoulder)\n"
        "- Status (moving/stopped/merging/turning/approaching)\n"
        "- Likely intent only if obvious (otherwise say 'unclear')\n\n"
        "Return as a numbered list. Be very concise."
    )
    return call_lvlm_with_image(prompt, image_path, model=model)


def m3cot_stage3_navigation_goal(target_left_m: float, target_front_m: float) -> str:
    """
    M3CoT Stage 3: Navigation Goal Prompting

    In the paper, this is expressed as a relative target position in meters:
      "The target is X meters to your left/right and Y meters to your front."

    Convention we’ll use:
    - target_left_m > 0  => left
    - target_left_m < 0  => right
    - target_front_m > 0 => in front
    """
    lr = "left" if target_left_m >= 0 else "right"
    return f"The target is {abs(target_left_m):.3f} meters to your {lr} and {target_front_m:.1f} meters to your front."


def m3cot_stage4_future_intent(
    scene_description: str,
    objects_description: str,
    navigation_goal: str,
    image_path: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    M3CoT Stage 4: Future Intent Description

    Input: Stage 1 + Stage 2 + Stage 3 (optionally image again)
    Output: concise plan + rationale (safe, goal-oriented)
    """
    prompt = (
        "Future Intent Description:\n"
        "You are controlling the ego vehicle. Based on the context below, describe how you would "
        "navigate to reach the target safely.\n\n"
        f"Scene description:\n{scene_description}\n\n"
        f"Objects description:\n{objects_description}\n\n"
        f"Navigation goal:\n{navigation_goal}\n\n"
        "Write a concise intent as a numbered list (3-6 items). Include:\n"
        "1) speed control (maintain/slow/stop if needed),\n"
        "2) direction control (keep lane / slight left/right),\n"
        "3) safety (avoid collisions, keep safe distance),\n"
        "4) goal alignment (move toward target).\n"
        "Do not invent objects that are not mentioned. If something is uncertain, say 'unclear'."
    )

    # Including the image is optional. For now, keep it consistent with vision-based grounding.
    if image_path is not None:
        return call_lvlm_with_image(prompt, image_path, model=model)
    else:
        return call_llm_text_only(prompt, model=model)



if __name__ == "__main__":
    MODEL = "gpt-4o-mini"

    IMAGE_PATH = "/Users/bhavya/Desktop/ms_projects/lvlm/images/img1.png"
    # img = Image.open(IMAGE_PATH)
    # print(img.size)   # original size
    # img = img.resize((800, 600))
    # img.show()

    # quick_sanity_check_image(IMAGE_PATH)

    # stage1 = m3cot_stage1_scene_description(IMAGE_PATH, model=MODEL)
    # stage2 = m3cot_stage2_interactive_objects(IMAGE_PATH, model=MODEL)
    # stage3 = m3cot_stage3_navigation_goal(10, 20)
    # stage4 = m3cot_stage4_future_intent(stage1, stage2, stage3, IMAGE_PATH, model=MODEL)

    # print("\n=== M3CoT Stage 1: Scene Description ===")
    # print(stage1)

    # print("\n=== M3CoT Stage 2: Interactive Objects ===")
    # print(stage2)

    # # Optional: bundle for later LangPack / Stage 4
    # m3cot_outputs = {"scene_description": stage1, "objects_description": stage2}
    # print("\nBundle:", m3cot_outputs)


    # print("\n=== M3CoT Stage 3: Navigation Goal ===")
    # print(stage3)

    # print("\n=== M3CoT Stage 4: Future Intent ===")
    # print(m3cot_stage4_future_intent(stage1, stage2, stage3, IMAGE_PATH, model=MODEL))

    # m3cot_outputs = {
    #     "scene_description": stage1,
    #     "objects_description": stage2,
    #     "navigation_goal": stage3,
    # }


