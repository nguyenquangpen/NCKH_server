from pydantic import BaseModel
from typing import List

class LlamaInput(BaseModel):
    captions: List[str]
    audio_texts: List[str]

class LlamaOutput(BaseModel):
    analysis: str
    embedding: List[float] 