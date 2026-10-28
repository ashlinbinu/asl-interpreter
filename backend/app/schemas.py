"""Request/response schemas (validated, size-limited). RawFrame follows docs/FEATURE_SPEC.md §1."""
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, StringConstraints

Point3 = Annotated[list[float], Field(min_length=3, max_length=3)]
PosePoint = Annotated[list[float], Field(min_length=3, max_length=4)]
MAX_FRAMES = 900  # 30 s at 30 fps


class Hand(BaseModel):
    handedness: Optional[Literal["Left", "Right"]] = None
    score: Optional[float] = None
    lm: Annotated[list[Point3], Field(min_length=21, max_length=21)]


class RawFrame(BaseModel):
    t: float = Field(..., description="timestamp in ms")
    w: int = Field(640, gt=0, le=10000)
    h: int = Field(480, gt=0, le=10000)
    hands: Annotated[list[Hand], Field(max_length=2)] = []
    pose: Optional[Annotated[list[PosePoint], Field(min_length=17, max_length=33)]] = None


LegacyFrame = Annotated[list[Annotated[list[Point3], Field(min_length=21, max_length=21)]], Field(max_length=2)]
Frame = Union[RawFrame, LegacyFrame]
Frames = Annotated[list[Frame], Field(max_length=MAX_FRAMES)]
LearnerId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{6,64}$")]


def frames_to_dicts(frames) -> list:
    return [f.model_dump() if isinstance(f, RawFrame) else f for f in frames or []]


class RecognizeIn(BaseModel):
    frames: Optional[Frames] = None
    image: Optional[Annotated[str, Field(max_length=12_000_000)]] = Field(
        None, description="base64 or data URL; landmarks are extracted server-side")
    mode: Literal["auto", "alphabet", "word"] = "auto"
    left_handed: bool = False
    top_k: int = Field(5, ge=1, le=10)


class Match(BaseModel):
    label: str
    confidence: float
    sign_id: Optional[int]
    reference_video_url: Optional[str]
    reference_image_url: Optional[str] = None


class RecognizeOut(BaseModel):
    matches: list[Match]
    rejected: bool
    method: str
    model_version: Optional[str]
    frames_used: int


class ScoreIn(BaseModel):
    sign_id: int
    frames: Frames
    left_handed: bool = False
    learner_id: Optional[LearnerId] = None
    mode: Literal["practice", "sentence"] = "practice"
    save: bool = True


class Highlight(BaseModel):
    slot: int = Field(description="0 = dominant hand, 1 = second hand (as drawn by the camera overlay)")
    joints: list[int] = Field(description="MediaPipe hand joint indices that differ most from the reference")
    component: str


class ScoreOut(BaseModel):
    score: float
    components: dict[str, float]
    hints: list[str]
    highlight: Optional[Highlight] = None
    feedback: str
    predicted: Optional[str]
    sign_label: str
    method: str
    model_version: Optional[str]


class SignIn(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
    frames: Frames
    learner_id: Optional[LearnerId] = None


class TextIn(BaseModel):
    text: Annotated[str, Field(max_length=500)]
    normalize: bool = Field(False, description="approximate English -> gloss normalisation (flagged in the result)")


class AttemptIn(BaseModel):
    learner_id: LearnerId
    sign_id: int
    score: float = Field(ge=0, le=100)
    components: dict[str, float] = {}
    hints: list[str] = []
    mode: Literal["practice", "sentence", "recognize"] = "practice"
    duration_ms: Optional[float] = None
