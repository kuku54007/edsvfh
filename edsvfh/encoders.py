from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from .config import EncoderConfig
from .types import ACTION_TYPES, FeatureSnapshot, StepObservation


class ObservationEncoder(Protocol):
    def extract(self, obs: StepObservation) -> FeatureSnapshot: ...
    def extract_batch(self, observations: Sequence[StepObservation]) -> list[FeatureSnapshot]: ...


_ENCODER_CACHE: dict[tuple[str, str, str, str, bool, int], ObservationEncoder] = {}


def snapshot_from_precomputed(obs: StepObservation) -> FeatureSnapshot:
    visual_embedding = (
        np.asarray(obs.precomputed_visual_embedding, dtype=np.float32)
        if obs.precomputed_visual_embedding is not None
        else np.zeros((0,), dtype=np.float32)
    )
    action_one_hot = (
        np.asarray(obs.precomputed_action_one_hot, dtype=np.float32)
        if obs.precomputed_action_one_hot is not None
        else np.zeros((len(ACTION_TYPES),), dtype=np.float32)
    )
    return FeatureSnapshot(
        vector=np.asarray(obs.precomputed_vector, dtype=np.float32),
        visual_embedding=visual_embedding,
        object_gripper_dist=float(obs.precomputed_object_gripper_dist or 0.0),
        object_target_dist=float(obs.precomputed_object_target_dist or 0.0),
        object_height=float(obs.precomputed_object_height or 0.0),
        visibility=float(1.0 if obs.precomputed_visibility is None else obs.precomputed_visibility),
        action_one_hot=action_one_hot,
    )


def _safe_resize(image: np.ndarray, size: int = 64) -> np.ndarray:
    if image is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class FallbackVisionEncoder:
    """Deterministic CPU-friendly encoder used for validated smoke tests.

    The vector intentionally mixes simple image statistics with standardized robot
    state to remain usable on tiny fixtures and without heavyweight model
    dependencies.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self.action_to_idx = {name: idx for idx, name in enumerate(ACTION_TYPES)}

    @staticmethod
    def _state_triplets(robot_state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        rs = np.asarray(robot_state, dtype=np.float32)
        padded = np.zeros(10, dtype=np.float32)
        padded[: min(len(rs), 10)] = rs[:10]
        eef = padded[0:3]
        gripper = float(padded[3])
        obj = padded[4:7]
        goal = padded[7:10]
        return eef, gripper, obj, goal

    def _image_stats(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = _safe_resize(image, 64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150)
        hist_b = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [8], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [8], [0, 256]).flatten()
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        stats = np.array(
            [
                float(gray.mean()) / 255.0,
                float(gray.std()) / 255.0,
                float(edges.mean()) / 255.0,
                float(img[..., 0].mean()) / 255.0,
                float(img[..., 1].mean()) / 255.0,
                float(img[..., 2].mean()) / 255.0,
            ],
            dtype=np.float32,
        )
        hist = np.concatenate([hist_b, hist_g, hist_r, hist_h]).astype(np.float32)
        hist = hist / max(float(hist.sum()), 1.0)
        return stats, hist

    def _action_one_hot(self, action_type: str) -> np.ndarray:
        action_one_hot = np.zeros(len(ACTION_TYPES), dtype=np.float32)
        action_one_hot[self.action_to_idx.get(action_type, len(ACTION_TYPES) - 1)] = 1.0
        return action_one_hot

    def _nonvisual_tail(self, obs: StepObservation) -> tuple[np.ndarray, float, float, float, float, np.ndarray]:
        eef, gripper, obj, goal = self._state_triplets(obs.robot_state)
        object_gripper_dist = float(np.linalg.norm(obj - eef))
        object_target_dist = float(np.linalg.norm(obj - goal))
        object_height = float(obj[2])
        visibility = 1.0 if obs.image is not None else 0.0
        action_one_hot = self._action_one_hot(obs.action_type)
        tail = np.concatenate(
            [
                eef,
                obj,
                goal,
                np.array([gripper, object_gripper_dist, object_target_dist, object_height], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        return tail, object_gripper_dist, object_target_dist, object_height, visibility, action_one_hot

    def _from_precomputed(self, obs: StepObservation) -> FeatureSnapshot:
        return snapshot_from_precomputed(obs)

    def extract(self, obs: StepObservation) -> FeatureSnapshot:
        if obs.precomputed_vector is not None:
            return self._from_precomputed(obs)
        image_stats, image_hist = self._image_stats(obs.image)
        tail, object_gripper_dist, object_target_dist, object_height, visibility, action_one_hot = self._nonvisual_tail(obs)
        visual_embedding = np.concatenate([image_stats, image_hist, tail], dtype=np.float32)
        vector = np.concatenate(
            [visual_embedding, obs.robot_state.astype(np.float32), obs.action.astype(np.float32), obs.policy_stats.astype(np.float32), action_one_hot],
            dtype=np.float32,
        )
        return FeatureSnapshot(
            vector=vector,
            visual_embedding=visual_embedding,
            object_gripper_dist=object_gripper_dist,
            object_target_dist=object_target_dist,
            object_height=object_height,
            visibility=visibility,
            action_one_hot=action_one_hot,
        )

    def extract_batch(self, observations: Sequence[StepObservation]) -> list[FeatureSnapshot]:
        return [self.extract(obs) for obs in observations]


@contextmanager
def _temporary_cudnn_disabled(torch_module, disable: bool):
    if not disable:
        yield
        return
    prev = bool(torch_module.backends.cudnn.enabled)
    torch_module.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch_module.backends.cudnn.enabled = prev


@contextmanager
def _inference_autocast(torch_module, device: str):
    if device.startswith('cuda'):
        try:
            with torch_module.autocast(device_type='cuda', dtype=torch_module.float16):
                yield
            return
        except Exception:
            pass
    with nullcontext():
        yield


def _is_cudnn_init_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        'CUDNN_STATUS_NOT_INITIALIZED' in msg
        or 'CUDNN_STATUS_EXECUTION_FAILED' in msg
        or 'CUDNN_STATUS_INTERNAL_ERROR' in msg
    )


def _torch_cuda_summary(torch_module) -> str:
    try:
        if not torch_module.cuda.is_available():
            return 'cuda-unavailable'
        idx = torch_module.cuda.current_device()
        name = torch_module.cuda.get_device_name(idx)
        cap = torch_module.cuda.get_device_capability(idx)
        return f'{name} cc={cap[0]}.{cap[1]}'
    except Exception:
        return 'cuda-unknown'


class HFBackboneMissingError(RuntimeError):
    pass


class _HFBase(FallbackVisionEncoder):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(config=config)
        self.disable_cudnn = bool(getattr(self.config, 'disable_cudnn', False))
        self.batch_size = int(getattr(self.config, 'convert_batch_size', 16))

    def _to_pil_batch(self, images: Sequence[np.ndarray], size: int = 224) -> list[Image.Image]:
        return [Image.fromarray(cv2.cvtColor(_safe_resize(img, size), cv2.COLOR_BGR2RGB)) for img in images]

    def _assemble_snapshots(self, observations: Sequence[StepObservation], image_embeddings: Sequence[np.ndarray]) -> list[FeatureSnapshot]:
        snaps: list[FeatureSnapshot] = []
        for obs, img_emb in zip(observations, image_embeddings):
            tail, object_gripper_dist, object_target_dist, object_height, visibility, action_one_hot = self._nonvisual_tail(obs)
            visual_embedding = np.concatenate([np.asarray(img_emb, dtype=np.float32), tail], dtype=np.float32)
            vector = np.concatenate(
                [visual_embedding, obs.robot_state.astype(np.float32), obs.action.astype(np.float32), obs.policy_stats.astype(np.float32), action_one_hot],
                dtype=np.float32,
            )
            snaps.append(
                FeatureSnapshot(
                    vector=vector,
                    visual_embedding=visual_embedding,
                    object_gripper_dist=object_gripper_dist,
                    object_target_dist=object_target_dist,
                    object_height=object_height,
                    visibility=visibility,
                    action_one_hot=action_one_hot,
                )
            )
        return snaps


class HFDinoV2Encoder(_HFBase):
    """Optional DINOv2 wrapper via transformers with batched GPU extraction."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(config=config)
        try:
            from transformers import AutoImageProcessor, AutoModel
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise HFBackboneMissingError('transformers + torch with DINOv2 weights are required for HFDinoV2Encoder') from exc
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(
            self.config.dinov2_model_id,
            local_files_only=self.config.hf_local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.config.dinov2_model_id,
            local_files_only=self.config.hf_local_files_only,
        )
        self.model.to(self.config.device).eval()

    def _image_embeddings_batch(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        pil_batch = self._to_pil_batch(images, 224)
        inputs = self.processor(images=pil_batch, return_tensors='pt')
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        try:
            with _temporary_cudnn_disabled(self.torch, self.disable_cudnn):
                with self.torch.inference_mode():
                    with _inference_autocast(self.torch, self.config.device):
                        outputs = self.model(**inputs)
        except RuntimeError as exc:
            if self.config.device == 'cuda' and _is_cudnn_init_error(exc):
                self.disable_cudnn = True
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                with _temporary_cudnn_disabled(self.torch, True):
                    with self.torch.inference_mode():
                        outputs = self.model(**inputs)
            else:
                raise
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            batch = outputs.pooler_output
        else:
            batch = outputs.last_hidden_state[:, 0, :]
        return [v.detach().cpu().float().numpy() for v in batch]

    def extract(self, obs: StepObservation) -> FeatureSnapshot:
        return self.extract_batch([obs])[0]

    def extract_batch(self, observations: Sequence[StepObservation]) -> list[FeatureSnapshot]:
        if not observations:
            return []
        if all(obs.precomputed_vector is not None for obs in observations):
            return [self._from_precomputed(obs) for obs in observations]
        img_embs = self._image_embeddings_batch([obs.image for obs in observations])
        return self._assemble_snapshots(observations, img_embs)


class HFSigLIP2DinoV2FusedEncoder(_HFBase):
    """Optional fused SigLIP2 + DINOv2 encoder with batched GPU extraction."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(config=config)
        try:
            from transformers import AutoModel, AutoImageProcessor, AutoProcessor
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise HFBackboneMissingError('transformers + torch with SigLIP2/DINOv2 weights are required for HFSigLIP2DinoV2FusedEncoder') from exc
        self.torch = torch
        self.siglip_processor = AutoProcessor.from_pretrained(
            self.config.siglip_model_id,
            local_files_only=self.config.hf_local_files_only,
        )
        self.siglip_model = AutoModel.from_pretrained(
            self.config.siglip_model_id,
            local_files_only=self.config.hf_local_files_only,
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(
            self.config.dinov2_model_id,
            local_files_only=self.config.hf_local_files_only,
        )
        self.dino_model = AutoModel.from_pretrained(
            self.config.dinov2_model_id,
            local_files_only=self.config.hf_local_files_only,
        )
        self.siglip_model.to(self.config.device).eval()
        self.dino_model.to(self.config.device).eval()
        if self.config.device == 'cuda' and self.torch.cuda.is_available():
            try:
                self.torch.cuda.init()
            except Exception:
                pass

    def _image_embeddings_batch(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        pil_batch = self._to_pil_batch(images, 224)
        sig_inputs = self.siglip_processor(images=pil_batch, return_tensors='pt')
        sig_inputs = {k: v.to(self.config.device) for k, v in sig_inputs.items()}
        din_inputs = self.dino_processor(images=pil_batch, return_tensors='pt')
        din_inputs = {k: v.to(self.config.device) for k, v in din_inputs.items()}
        try:
            with _temporary_cudnn_disabled(self.torch, self.disable_cudnn):
                with self.torch.inference_mode():
                    with _inference_autocast(self.torch, self.config.device):
                        sig_out = self.siglip_model.get_image_features(**sig_inputs)
                        dino_out = self.dino_model(**din_inputs)
        except RuntimeError as exc:
            if self.config.device == 'cuda' and _is_cudnn_init_error(exc):
                self.disable_cudnn = True
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                with _temporary_cudnn_disabled(self.torch, True):
                    with self.torch.inference_mode():
                        sig_out = self.siglip_model.get_image_features(**sig_inputs)
                        dino_out = self.dino_model(**din_inputs)
            else:
                raise RuntimeError(f'HF fused encoder failed on {self.config.device} ({_torch_cuda_summary(self.torch)}): {exc}') from exc
        dino_batch = dino_out.pooler_output if getattr(dino_out, 'pooler_output', None) is not None else dino_out.last_hidden_state[:, 0, :]
        fused_batch = self.torch.cat([sig_out, dino_batch], dim=-1)
        return [v.detach().cpu().float().numpy() for v in fused_batch]

    def extract(self, obs: StepObservation) -> FeatureSnapshot:
        return self.extract_batch([obs])[0]

    def extract_batch(self, observations: Sequence[StepObservation]) -> list[FeatureSnapshot]:
        if not observations:
            return []
        if all(obs.precomputed_vector is not None for obs in observations):
            return [self._from_precomputed(obs) for obs in observations]
        img_embs = self._image_embeddings_batch([obs.image for obs in observations])
        return self._assemble_snapshots(observations, img_embs)


def build_encoder(config: EncoderConfig | None = None, *, reuse: bool = True) -> ObservationEncoder:
    config = config or EncoderConfig()
    name = config.name.lower()
    key = (
        name,
        str(config.device),
        str(config.siglip_model_id),
        str(config.dinov2_model_id),
        bool(config.hf_local_files_only),
        int(config.convert_batch_size),
    )
    if reuse and key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    if name == 'fallback':
        encoder: ObservationEncoder = FallbackVisionEncoder(config)
    elif name in {'dinov2', 'hf_dinov2'}:
        encoder = HFDinoV2Encoder(config)
    elif name in {'siglip2_dinov2', 'hf_siglip2_dinov2'}:
        encoder = HFSigLIP2DinoV2FusedEncoder(config)
    else:
        raise ValueError(f'Unsupported encoder name: {config.name}')
    if reuse:
        _ENCODER_CACHE[key] = encoder
    return encoder
