"""
Keyframe extraction wrapper for StoryMem.

Extracts keyframes from video using:
- CLIP embeddings for similarity detection
- HPSv3 for quality filtering (optional)
- Configurable similarity thresholds
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

from .tensor_utils import extract_frames_from_video

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """
    Wrapper around StoryMem's keyframe extraction logic.

    Extracts representative keyframes from video shots for memory conditioning.
    Uses CLIP + optional HPSv3 quality filtering.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        max_keyframes: int = 3,
        use_clip: bool = True,
        use_hpsv3: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize keyframe extractor.

        Args:
            similarity_threshold: CLIP similarity threshold (0-1)
            max_keyframes: Maximum keyframes to extract per shot
            use_clip: Use CLIP for similarity detection
            use_hpsv3: Use HPSv3 for quality filtering
            device: Target device for models
        """
        self.similarity_threshold = similarity_threshold
        self.max_keyframes = max_keyframes
        self.use_clip = use_clip
        self.use_hpsv3 = use_hpsv3
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model storage
        self.clip_model = None
        self.clip_preprocess = None
        self.hpsv3_model = None

        # Lazy loading flags
        self._models_loaded = False

        logger.info(
            f"KeyframeExtractor initialized: threshold={similarity_threshold}, "
            f"max_keyframes={max_keyframes}, device={self.device}"
        )

    def _load_models(self):
        """Lazy load CLIP and HPSv3 models."""
        if self._models_loaded:
            return

        try:
            if self.use_clip:
                logger.info("Loading CLIP model...")
                # TODO: Load CLIP when available
                # import clip
                # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                logger.warning("CLIP loading not yet implemented (placeholder)")

            if self.use_hpsv3:
                logger.info("Loading HPSv3 model...")
                # TODO: Load HPSv3 when available
                logger.warning("HPSv3 loading not yet implemented (placeholder)")

            self._models_loaded = True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def extract_from_video(
        self,
        video_tensor: torch.Tensor,
        force_count: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Extract keyframes from video tensor.

        Args:
            video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]
            force_count: Force exact number of keyframes (overrides max_keyframes)

        Returns:
            keyframes: List of keyframe tensors [H, W, C]
            indices: List of frame indices where keyframes were extracted
        """
        # Extract individual frames
        frames = extract_frames_from_video(video_tensor)

        if not frames:
            logger.warning("No frames to extract keyframes from")
            return [], []

        num_frames = len(frames)
        target_count = force_count if force_count is not None else self.max_keyframes

        logger.debug(f"Extracting keyframes from {num_frames} frames (target: {target_count})")

        # Simple strategy: evenly spaced frames
        # TODO: Implement CLIP-based similarity detection once models are available
        keyframe_indices = self._extract_evenly_spaced(num_frames, target_count)

        # Extract frames at selected indices
        keyframes = [frames[idx] for idx in keyframe_indices]

        logger.debug(f"Extracted {len(keyframes)} keyframes at indices: {keyframe_indices}")

        return keyframes, keyframe_indices

    def _extract_evenly_spaced(
        self,
        num_frames: int,
        target_count: int
    ) -> List[int]:
        """
        Extract evenly spaced frame indices.

        Args:
            num_frames: Total number of frames
            target_count: Desired number of keyframes

        Returns:
            List of frame indices
        """
        if target_count >= num_frames:
            # Return all frames
            return list(range(num_frames))

        # Evenly space keyframes
        step = num_frames / target_count
        indices = [int(i * step) for i in range(target_count)]

        # Always include last frame
        if indices[-1] != num_frames - 1:
            indices[-1] = num_frames - 1

        return indices

    def _extract_with_clip(
        self,
        frames: List[torch.Tensor],
        target_count: int
    ) -> List[int]:
        """
        Extract keyframes using CLIP similarity detection.

        Args:
            frames: List of frame tensors
            target_count: Desired number of keyframes

        Returns:
            List of frame indices

        TODO: Implement once CLIP is integrated
        """
        logger.warning("CLIP-based extraction not yet implemented, using evenly spaced")
        return self._extract_evenly_spaced(len(frames), target_count)

    def _filter_with_hpsv3(
        self,
        frames: List[torch.Tensor],
        indices: List[int]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Filter keyframes using HPSv3 quality scores.

        Args:
            frames: List of keyframe tensors
            indices: Corresponding frame indices

        Returns:
            Filtered keyframes and indices

        TODO: Implement once HPSv3 is integrated
        """
        logger.warning("HPSv3 filtering not yet implemented, returning unfiltered")
        return frames, indices

    def get_last_frame(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract last frame from video for MI2V conditioning.

        Args:
            video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]

        Returns:
            Last frame tensor [H, W, C]
        """
        frames = extract_frames_from_video(video_tensor)
        if not frames:
            raise ValueError("Video has no frames")
        return frames[-1]

    def get_motion_frames(
        self,
        video_tensor: torch.Tensor,
        num_frames: int = 5
    ) -> List[torch.Tensor]:
        """
        Extract last N frames for MM2V motion conditioning.

        Args:
            video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]
            num_frames: Number of frames to extract from end

        Returns:
            List of motion frame tensors [H, W, C]
        """
        frames = extract_frames_from_video(video_tensor)
        if not frames:
            raise ValueError("Video has no frames")

        # Return last N frames
        return frames[-num_frames:]

    def analyze_frame_diversity(
        self,
        video_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze diversity of frames in video.

        Args:
            video_tensor: Video tensor

        Returns:
            Dictionary with diversity metrics

        TODO: Implement once CLIP is available
        """
        frames = extract_frames_from_video(video_tensor)

        return {
            'num_frames': len(frames),
            'recommended_keyframes': min(self.max_keyframes, max(1, len(frames) // 8)),
            'clip_available': self.clip_model is not None,
        }


def extract_keyframes_simple(
    video_tensor: torch.Tensor,
    num_keyframes: int = 3
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Simple keyframe extraction without model loading.

    Convenience function for basic evenly-spaced extraction.

    Args:
        video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]
        num_keyframes: Number of keyframes to extract

    Returns:
        keyframes: List of keyframe tensors [H, W, C]
        indices: List of frame indices
    """
    extractor = KeyframeExtractor(max_keyframes=num_keyframes, use_clip=False)
    return extractor.extract_from_video(video_tensor)
