"""
Memory management nodes for StoryMem.

Handles memory buffer creation and updates for shot-to-shot continuity.
"""

from ..storymem_wrapper.memory_manager import create_memory_buffer
from ..storymem_wrapper.keyframe_extractor import extract_keyframes_simple


class StoryMemMemoryBuffer:
    """
    Create a memory buffer for shot-to-shot continuity.

    The memory buffer maintains keyframes across shots using a sliding window strategy:
    - Keeps first `fixed_frames` (character consistency)
    - Fills remaining slots with most recent frames (context)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_memory_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider",
                }),
                "fixed_frames": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                }),
            }
        }

    RETURN_TYPES = ("STORYMEM_MEMORY",)
    RETURN_NAMES = ("memory",)
    FUNCTION = "create_buffer"
    CATEGORY = "StoryMem/memory"
    DESCRIPTION = "Create a memory buffer for maintaining consistency across video shots. Uses sliding window strategy with fixed early frames and recent context."

    def create_buffer(self, max_memory_size: int, fixed_frames: int):
        """
        Create a new memory buffer.

        Args:
            max_memory_size: Maximum keyframes to retain (e.g., 10)
            fixed_frames: Number of earliest frames to always keep (e.g., 3)

        Returns:
            Tuple containing memory buffer
        """
        if fixed_frames > max_memory_size:
            print(f"⚠ fixed_frames ({fixed_frames}) > max_memory_size ({max_memory_size}), adjusting to {max_memory_size}")
            fixed_frames = max_memory_size

        memory = create_memory_buffer(
            max_size=max_memory_size,
            fixed_count=fixed_frames
        )

        print(f"✓ Memory buffer created: max={max_memory_size}, fixed={fixed_frames}")

        return (memory,)


class StoryMemUpdateMemory:
    """
    Manually update memory buffer with keyframes from a video.

    Useful for adding specific frames to memory or rebuilding memory state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory": ("STORYMEM_MEMORY",),
                "video": ("IMAGE",),  # ComfyUI batched images = video frames
            },
            "optional": {
                "extract_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("STORYMEM_MEMORY", "IMAGE")
    RETURN_NAMES = ("updated_memory", "keyframes")
    FUNCTION = "update_memory"
    CATEGORY = "StoryMem/memory"
    DESCRIPTION = "Extract keyframes from video and add them to the memory buffer. Returns updated memory and extracted keyframes for visualization."

    def update_memory(
        self,
        memory,
        video,
        extract_count: int = 3,
        similarity_threshold: float = 0.9,
    ):
        """
        Update memory buffer with keyframes from video.

        Args:
            memory: Existing memory buffer
            video: Video frames as batched images [B, F, H, W, C] or [F, H, W, C]
            extract_count: Number of keyframes to extract
            similarity_threshold: Similarity threshold for extraction

        Returns:
            Tuple containing (updated memory, keyframes as images)
        """
        import torch

        # Extract keyframes from video
        keyframes, indices = extract_keyframes_simple(
            video,
            num_keyframes=extract_count
        )

        if not keyframes:
            print("⚠ No keyframes extracted from video")
            # Return empty image
            empty_image = torch.zeros(1, 480, 832, 3)
            return (memory, empty_image)

        # Add keyframes to memory
        memory.add_keyframes(keyframes)

        print(f"✓ Added {len(keyframes)} keyframes to memory at indices {indices}")
        print(f"  Memory now contains: {len(memory)} frames")

        # Stack keyframes for output visualization
        # Convert list of [H,W,C] to [N,H,W,C]
        keyframes_batch = torch.stack(keyframes, dim=0)

        return (memory, keyframes_batch)
