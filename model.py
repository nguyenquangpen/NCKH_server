from pydantic import BaseModel, Field
from typing import List, Optional

class ShotMetadata(BaseModel):
    shot_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

class VideoStructure(BaseModel):
    video_id: str
    fps: float
    total_frames: int
    shots: List[ShotMetadata]

class AudioSnippet(BaseModel):
    text: str
    start: float
    end: float

class PerceptionResult(BaseModel):
    shot_id: int
    visual_caption: str
    audio_context: List[AudioSnippet]

class AgentReasoningResult(BaseModel):
    shot_id: int
    aligned_text: str
    keywords: List[str]
    reasoning: str
    local_score: float = Field(..., ge=0, le=1)

class ModelInputTensor(BaseModel):
    video_id: str
    embedding_dim: int = 5120 
    n_shots: int
    
    change_points: List[List[int]]
    n_frame_per_seg: List[int]
    n_frames: int

class SummaryResult(BaseModel):
    video_id: str
    selected_shot_indices: List[int]
    importance_scores: List[float]
    summary_duration_frames: int
    proportion: float = 0.15