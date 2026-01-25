from pydantic import BaseModel
from typing import Optional

class FlorenceInput(BaseModel):
    shot_id: int
    image_b64: str

class FlorenceOutput(BaseModel):
    shot_id: int
    caption: str