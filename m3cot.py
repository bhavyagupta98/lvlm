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

    # print("\n=== M3CoT Stage 1: Scene Description ===")
    # print(stage1)

    # print("\n=== M3CoT Stage 2: Interactive Objects ===")
    # print(stage2)

    # # Optional: bundle for later LangPack / Stage 4
    # m3cot_outputs = {"scene_description": stage1, "objects_description": stage2}
    # print("\nBundle:", m3cot_outputs)
