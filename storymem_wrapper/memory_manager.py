"""
Memory buffer management for StoryMem shot-to-shot continuity.

Implements a sliding window memory strategy that keeps:
- Fixed earliest frames (long-term character consistency)
- Most recent frames (smooth transitions)
"""

import torch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StoryMemMemory:
    """
    Memory buffer for shot-to-shot continuity in video generation.

    Uses a sliding window strategy:
    - Keeps first `fixed_count` frames (character consistency)
    - Fills remaining slots with most recent frames (context)
    - Never exceeds `max_size` total frames

    Example:
        >>> memory = StoryMemMemory(max_size=10, fixed_count=3)
        >>> memory.add_keyframes([frame1, frame2, frame3], shot_id=1)
        >>> memory.add_keyframes([frame4, frame5], shot_id=2)
        >>> len(memory.keyframes)  # Total frames in buffer
        5
    """

    def __init__(
        self,
        max_size: int = 10,
        fixed_count: int = 3,
    ):
        """
        Initialize memory buffer.

        Args:
            max_size: Maximum number of keyframes to retain
            fixed_count: Number of earliest frames to always keep
        """
        if fixed_count > max_size:
            raise ValueError(f"fixed_count ({fixed_count}) cannot exceed max_size ({max_size})")

        self.max_size = max_size
        self.fixed_count = fixed_count

        # Storage
        self.keyframes: List[torch.Tensor] = []  # [H, W, C] tensors

        # Metadata tracking
        self.metadata: Dict[str, List[Any]] = {
            'shot_ids': [],          # Which shot each frame came from
            'frame_indices': [],     # Original frame index in source video
            'quality_scores': [],    # Optional quality scores
            'timestamps': [],        # When frame was added
        }

        self._shot_counter = 0

        logger.info(f"StoryMemMemory initialized: max_size={max_size}, fixed_count={fixed_count}")

    def add_keyframes(
        self,
        new_keyframes: List[torch.Tensor],
        shot_id: Optional[int] = None,
        frame_indices: Optional[List[int]] = None,
        quality_scores: Optional[List[float]] = None,
    ):
        """
        Add new keyframes to the memory buffer.

        Applies sliding window strategy:
        - Keep first `fixed_count` frames
        - Fill remaining slots with most recent frames

        Args:
            new_keyframes: List of keyframe tensors [H, W, C]
            shot_id: Shot identifier (auto-increments if None)
            frame_indices: Original frame indices in source video
            quality_scores: Quality scores for each frame
        """
        if not new_keyframes:
            logger.warning("add_keyframes called with empty list")
            return

        # Auto-increment shot ID if not provided
        if shot_id is None:
            shot_id = self._shot_counter
            self._shot_counter += 1

        # Validate keyframe format
        for i, frame in enumerate(new_keyframes):
            if not isinstance(frame, torch.Tensor):
                raise TypeError(f"Keyframe {i} is not a torch.Tensor")
            if frame.ndim != 3:
                raise ValueError(f"Keyframe {i} has {frame.ndim} dimensions, expected 3 [H,W,C]")

        # Add keyframes
        self.keyframes.extend(new_keyframes)

        # Update metadata
        self.metadata['shot_ids'].extend([shot_id] * len(new_keyframes))

        if frame_indices is not None:
            self.metadata['frame_indices'].extend(frame_indices)
        else:
            # Default to sequential indices
            start_idx = len(self.metadata['frame_indices'])
            self.metadata['frame_indices'].extend(range(start_idx, start_idx + len(new_keyframes)))

        if quality_scores is not None:
            self.metadata['quality_scores'].extend(quality_scores)
        else:
            self.metadata['quality_scores'].extend([1.0] * len(new_keyframes))

        import time
        self.metadata['timestamps'].extend([time.time()] * len(new_keyframes))

        # Apply sliding window if exceeded max_size
        if len(self.keyframes) > self.max_size:
            self._apply_sliding_window()

        logger.debug(f"Added {len(new_keyframes)} keyframes from shot {shot_id}. Total: {len(self.keyframes)}")

    def _apply_sliding_window(self):
        """
        Apply sliding window strategy to trim memory buffer.

        Strategy:
        - Keep first `fixed_count` frames
        - Keep most recent (max_size - fixed_count) frames
        - Discard frames in the middle
        """
        if len(self.keyframes) <= self.max_size:
            return

        # Split into fixed and recent
        fixed_frames = self.keyframes[:self.fixed_count]
        recent_count = self.max_size - self.fixed_count
        recent_frames = self.keyframes[-recent_count:]

        # Update keyframes
        self.keyframes = fixed_frames + recent_frames

        # Update all metadata lists
        for key in self.metadata:
            fixed_meta = self.metadata[key][:self.fixed_count]
            recent_meta = self.metadata[key][-recent_count:]
            self.metadata[key] = fixed_meta + recent_meta

        logger.debug(f"Applied sliding window: {len(self.keyframes)} frames retained")

    def get_memory_frames(self) -> List[torch.Tensor]:
        """
        Get all keyframes in memory buffer.

        Returns:
            List of keyframe tensors [H, W, C]
        """
        return self.keyframes.copy()

    def get_last_frame(self) -> Optional[torch.Tensor]:
        """
        Get the most recent keyframe.

        Returns:
            Last keyframe tensor [H, W, C], or None if buffer is empty
        """
        if not self.keyframes:
            return None
        return self.keyframes[-1]

    def get_last_n_frames(self, n: int) -> List[torch.Tensor]:
        """
        Get the last N keyframes for motion conditioning.

        Args:
            n: Number of frames to retrieve

        Returns:
            List of up to N most recent keyframes
        """
        if n <= 0:
            return []
        return self.keyframes[-n:]

    def clear(self):
        """Clear all keyframes and metadata (for scene cuts)."""
        self.keyframes = []
        self.metadata = {
            'shot_ids': [],
            'frame_indices': [],
            'quality_scores': [],
            'timestamps': [],
        }
        logger.debug("Memory buffer cleared")

    def get_shot_count(self) -> int:
        """Get number of unique shots in memory."""
        if not self.metadata['shot_ids']:
            return 0
        return len(set(self.metadata['shot_ids']))

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about memory buffer state.

        Returns:
            Dictionary with buffer statistics
        """
        return {
            'total_frames': len(self.keyframes),
            'max_size': self.max_size,
            'fixed_count': self.fixed_count,
            'unique_shots': self.get_shot_count(),
            'is_full': len(self.keyframes) >= self.max_size,
            'is_empty': len(self.keyframes) == 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize memory buffer to dictionary for ComfyUI data passing.

        Returns:
            Dictionary containing all buffer state
        """
        return {
            'keyframes': self.keyframes,
            'max_size': self.max_size,
            'fixed_count': self.fixed_count,
            'metadata': self.metadata,
            'shot_counter': self._shot_counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryMemMemory':
        """
        Deserialize memory buffer from dictionary.

        Args:
            data: Dictionary containing buffer state

        Returns:
            Reconstructed StoryMemMemory instance
        """
        memory = cls(
            max_size=data['max_size'],
            fixed_count=data['fixed_count']
        )
        memory.keyframes = data['keyframes']
        memory.metadata = data['metadata']
        memory._shot_counter = data.get('shot_counter', 0)
        return memory

    def visualize_buffer(self) -> torch.Tensor:
        """
        Create a visualization of the memory buffer.

        Returns:
            Tensor [H, W, C] showing grid of keyframes

        Note:
            Creates a grid layout of all keyframes for inspection
        """
        if not self.keyframes:
            # Return black image if empty
            return torch.zeros(480, 832, 3)

        # Determine grid size
        num_frames = len(self.keyframes)
        grid_cols = min(5, num_frames)
        grid_rows = (num_frames + grid_cols - 1) // grid_cols

        # Get frame dimensions
        h, w, c = self.keyframes[0].shape

        # Create grid
        grid_h = h * grid_rows
        grid_w = w * grid_cols
        grid = torch.zeros(grid_h, grid_w, c)

        # Fill grid
        for idx, frame in enumerate(self.keyframes):
            row = idx // grid_cols
            col = idx % grid_cols
            y_start = row * h
            x_start = col * w
            grid[y_start:y_start+h, x_start:x_start+w] = frame

        return grid

    def __len__(self) -> int:
        """Return number of keyframes in buffer."""
        return len(self.keyframes)

    def __repr__(self) -> str:
        """String representation of memory buffer."""
        return (
            f"StoryMemMemory(frames={len(self.keyframes)}/{self.max_size}, "
            f"fixed={self.fixed_count}, shots={self.get_shot_count()})"
        )


def create_memory_buffer(
    max_size: int = 10,
    fixed_count: int = 3
) -> StoryMemMemory:
    """
    Factory function to create a memory buffer.

    Args:
        max_size: Maximum keyframes to retain
        fixed_count: Number of earliest frames to always keep

    Returns:
        Initialized StoryMemMemory instance
    """
    return StoryMemMemory(max_size=max_size, fixed_count=fixed_count)
