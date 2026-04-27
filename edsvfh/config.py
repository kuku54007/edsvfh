from __future__ import annotations

from pathlib import Path
import os
from pydantic import BaseModel, Field



def _parse_horizons_env() -> tuple[int, ...]:
    raw = os.getenv("EDSVFH_HORIZONS") or os.getenv("HORIZONS")
    if not raw:
        return (1, 3, 5)
    values: list[int] = []
    for token in raw.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid horizon value {token!r} in EDSVFH_HORIZONS={raw!r}") from exc
        if value <= 0:
            raise ValueError(f"Horizon values must be positive integers; got {value} in EDSVFH_HORIZONS={raw!r}")
        values.append(value)
    if not values:
        return (1, 3, 5)
    return tuple(sorted(dict.fromkeys(values)))

def _default_device() -> str:
    env = os.getenv("EDSVFH_DEVICE")
    if env:
        return env
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class WatcherConfig(BaseModel):
    visual_weight: float = Field(default=0.35, ge=0.0)
    stall_weight: float = Field(default=0.20, ge=0.0)
    uncertainty_weight: float = Field(default=0.25, ge=0.0)
    high_stakes_weight: float = Field(default=0.20, ge=0.0)
    trigger_threshold: float = Field(default=0.40, ge=0.0, le=1.5)
    heartbeat_steps: int = Field(default=4, ge=1)
    stall_window: int = Field(default=4, ge=2)
    stall_delta_threshold: float = Field(default=0.03, ge=0.0)


class MemoryConfig(BaseModel):
    capacity: int = Field(default=6, ge=1)
    window: int = Field(default=4, ge=1)


class DecisionConfig(BaseModel):
    continue_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    warning_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    stop_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    warning_uncertainty: float = Field(default=0.35, ge=0.0, le=1.0)
    abstain_uncertainty: float = Field(default=0.85, ge=0.0, le=1.0)
    spread_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    confirm_count: int = Field(default=1, ge=1)
    terminal_decisions: tuple[str, ...] = ("shield", "abstain")


class EncoderConfig(BaseModel):
    name: str = Field(default_factory=lambda: os.getenv('EDSVFH_ENCODER') or os.getenv('ENCODER') or 'fallback')
    visual_dim: int = Field(default=64, ge=8)
    device: str = Field(default_factory=_default_device)
    siglip_model_id: str = Field(default="google/siglip2-base-patch16-256")
    dinov2_model_id: str = Field(default="facebook/dinov2-base")
    disable_cudnn: bool = Field(default_factory=lambda: os.getenv("EDSVFH_DISABLE_CUDNN", "0") == "1")
    hf_local_files_only: bool = Field(
        default_factory=lambda: (
            os.getenv("EDSVFH_HF_LOCAL_ONLY", "0") == "1"
            or os.getenv("HF_HUB_OFFLINE", "0") == "1"
            or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        )
    )
    convert_batch_size: int = Field(default_factory=lambda: int(os.getenv("EDSVFH_CONVERT_BATCH_SIZE", "16")), ge=1)


class TrainingConfig(BaseModel):
    horizons: tuple[int, ...] = Field(default_factory=_parse_horizons_env)
    num_subgoals: int = 5
    random_seed: int = 7
    n_estimators: int = 120
    max_depth: int = 12
    valid_ratio: float = Field(default=0.15, gt=0.0, lt=0.5)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=0.5)


class DatasetConfig(BaseModel):
    default_instruction: str = "Complete the manipulation task safely."
    image_key: str | None = None
    eef_key: str | None = None
    gripper_key: str | None = None
    object_key: str | None = None
    goal_key: str | None = None
    max_episodes: int | None = None


class AppConfig(BaseModel):
    watcher: WatcherConfig = WatcherConfig()
    memory: MemoryConfig = MemoryConfig()
    decision: DecisionConfig = DecisionConfig()
    encoder: EncoderConfig = EncoderConfig()
    training: TrainingConfig = TrainingConfig()
    dataset: DatasetConfig = DatasetConfig()


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = ROOT / "artifacts" / "public_fixture_bundle.pkl"
DEFAULT_FIXTURE_PATH = ROOT / "artifacts" / "tiny_robomimic_fixture.hdf5"
