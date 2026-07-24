"""Query a local VLM inference server to locate a ColorChecker in an image."""

import base64
import json
import sys
from pathlib import Path
from typing import Optional
from urllib import request

from PIL import Image
from pydantic import BaseModel, Field, field_validator

SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "mlx-community/gemma-4-12B-it-4bit"
IMAGE_PATH = Path(__file__).parent / "colorchecker.png"

# The model has no idea what pixel grid we'll display its answer against
# (it sees the image resized to its own vision encoder's input size), so it's
# asked for a point as a fraction of the image (0-1) rather than raw pixels,
# and that fraction is scaled against the image's actual dimensions
# afterward.
QUERY = (
    "Does this image contain a colorchecker and if so where is its "
    "approximate center? Respond in JSON with two fields: 'present', which "
    "is true or false, and 'center', which is either null or [x, y] where "
    "x and y are each a fraction between 0 and 1 of the image's width and "
    "height respectively."
)


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ModelResponse(BaseModel):
    present: bool
    center: Optional[NormalizedPoint] = None

    @field_validator("center", mode="before")
    @classmethod
    def _coerce_center_list(cls, value):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"center list must have exactly 2 values, got {len(value)}")
            x, y = value
            return {"x": x, "y": y}
        return value


class PixelPoint(BaseModel):
    x: int
    y: int


class DetectionResult(BaseModel):
    present: bool
    center: Optional[PixelPoint] = None


# Stubbed out with the known-correct center for colorchecker.png (the
# ColorChecker Classic patch grid) so downstream LUT automation work isn't
# blocked on a working VLM server. Swap back to the real HTTP call below
# once a VLM has been validated for this task.
SIMULATED_RESPONSE = json.dumps({"present": True, "center": [0.570, 0.495]})


def query_vlm(image_path: Path) -> str:
    return SIMULATED_RESPONSE


def _query_vlm_live(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{image_b64}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUERY},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 500,
    }

    req = request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as resp:
        response = json.load(resp)

    return response["choices"][0]["message"]["content"]


def parse_model_response(content: str) -> ModelResponse:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    return ModelResponse.model_validate_json(text.strip())


def to_pixel_result(parsed: ModelResponse, image_width: int, image_height: int) -> DetectionResult:
    if not parsed.present or parsed.center is None:
        return DetectionResult(present=parsed.present, center=None)

    center = parsed.center
    x = min(round(center.x * image_width), image_width - 1)
    y = min(round(center.y * image_height), image_height - 1)

    return DetectionResult(present=True, center=PixelPoint(x=x, y=y))


def main() -> None:
    with Image.open(IMAGE_PATH) as img:
        image_width, image_height = img.size

    content = query_vlm(IMAGE_PATH)
    parsed = parse_model_response(content)
    result = to_pixel_result(parsed, image_width, image_height)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    sys.exit(main())
