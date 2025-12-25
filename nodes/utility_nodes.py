"""
Utility nodes for StoryMem.

Helper nodes for:
- Memory visualization
- Video combination
- Format conversion
"""

import torch
from ..storymem_wrapper.tensor_utils import concatenate_videos


class StoryMemMemoryVisualizer:
    """
    Visualize memory buffer as a grid of keyframes.

    Useful for debugging and understanding what's in memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory": ("STORYMEM_MEMORY",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "StoryMem/utils"
    DESCRIPTION = "Visualize memory buffer contents as a grid of keyframes. Useful for debugging."

    def visualize(self, memory):
        """
        Create visualization of memory buffer.

        Args:
            memory: Memory buffer

        Returns:
            Tuple containing visualization image
        """
        # Get visualization from memory buffer
        grid = memory.visualize_buffer()

        # Add batch dimension [H, W, C] -> [1, H, W, C]
        grid = grid.unsqueeze(0)

        info = memory.get_info()
        print(f"\n{'='*60}")
        print("Memory Buffer Visualization")
        print(f"{'='*60}")
        print(f"Total frames: {info['total_frames']}/{info['max_size']}")
        print(f"Fixed frames: {info['fixed_count']}")
        print(f"Unique shots: {info['unique_shots']}")
        print(f"Status: {'Full' if info['is_full'] else 'Not full'}")

        return (grid,)


class StoryMemVideoCombine:
    """
    Combine multiple video shots into a single video.

    Concatenates videos along the frame dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("IMAGE",),
            },
            "optional": {
                "video2": ("IMAGE",),
                "video3": ("IMAGE",),
                "video4": ("IMAGE",),
                "video5": ("IMAGE",),
                "video6": ("IMAGE",),
                "video7": ("IMAGE",),
                "video8": ("IMAGE",),
                "video9": ("IMAGE",),
                "video10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_video",)
    FUNCTION = "combine"
    CATEGORY = "StoryMem/utils"
    DESCRIPTION = "Combine multiple video shots into a single video by concatenating frames."

    def combine(self, video1, video2=None, video3=None, video4=None, video5=None,
                video6=None, video7=None, video8=None, video9=None, video10=None):
        """
        Combine videos.

        Args:
            video1-10: Video tensors to combine

        Returns:
            Tuple containing combined video
        """
        # Collect all provided videos
        videos = [video1]
        for video in [video2, video3, video4, video5, video6, video7, video8, video9, video10]:
            if video is not None:
                videos.append(video)

        print(f"\n{'='*60}")
        print("Combining Videos")
        print(f"{'='*60}")
        print(f"Number of shots: {len(videos)}")

        # Get frame counts
        for i, video in enumerate(videos, 1):
            if video.ndim == 4:  # [F, H, W, C]
                num_frames = video.shape[0]
            elif video.ndim == 5:  # [B, F, H, W, C]
                num_frames = video.shape[1]
            else:
                num_frames = 0
            print(f"  Shot {i}: {num_frames} frames")

        # Ensure all videos have batch dimension
        videos_batched = []
        for video in videos:
            if video.ndim == 4:  # [F, H, W, C]
                video = video.unsqueeze(0)  # [1, F, H, W, C]
            videos_batched.append(video)

        # Concatenate along frame dimension (dim=1)
        combined = concatenate_videos(videos_batched)

        total_frames = combined.shape[1]
        print(f"✓ Combined into {total_frames} total frames")

        return (combined,)


class StoryMemGetMotionFrames:
    """
    Extract last N frames from video for MM2V motion conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "num_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("motion_frames",)
    FUNCTION = "extract"
    CATEGORY = "StoryMem/utils"
    DESCRIPTION = "Extract last N frames from video for MM2V motion conditioning."

    def extract(self, video, num_frames: int = 5):
        """
        Extract motion frames.

        Args:
            video: Video tensor
            num_frames: Number of frames to extract

        Returns:
            Tuple containing motion frames
        """
        from ..storymem_wrapper.tensor_utils import get_motion_frames

        motion = get_motion_frames(video, num_frames)

        actual_frames = motion.shape[1] if motion.ndim == 5 else motion.shape[0]
        print(f"✓ Extracted {actual_frames} motion frames for MM2V conditioning")

        return (motion,)


class StoryMemInfo:
    """
    Display information about StoryMem models and memory state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "models": ("STORYMEM_MODELS",),
                "memory": ("STORYMEM_MEMORY",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_info"
    CATEGORY = "StoryMem/utils"
    DESCRIPTION = "Display information about loaded models and memory state."

    def show_info(self, models=None, memory=None):
        """
        Display information.

        Args:
            models: Optional models dict
            memory: Optional memory buffer
        """
        print(f"\n{'='*60}")
        print("StoryMem Information")
        print(f"{'='*60}")

        if models is not None:
            print("\nModels:")
            print(f"  T2V: {models.get('t2v_path', 'N/A')}")
            print(f"  I2V: {models.get('i2v_path', 'N/A')}")
            print(f"  M2V LoRA: {models.get('m2v_type', 'N/A')}")
            print(f"  Device: {models.get('device', 'N/A')}")
            print(f"  Offloading: {models.get('enable_offload', False)}")

        if memory is not None:
            info = memory.get_info()
            print("\nMemory Buffer:")
            print(f"  Total frames: {info['total_frames']}/{info['max_size']}")
            print(f"  Fixed frames: {info['fixed_count']}")
            print(f"  Unique shots: {info['unique_shots']}")
            print(f"  Status: {'Full' if info['is_full'] else 'Not full'}")
            print(f"  Empty: {info['is_empty']}")

        print(f"{'='*60}\n")

        return ()
