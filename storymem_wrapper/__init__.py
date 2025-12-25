"""
Wrapper classes that bridge ComfyUI and StoryMem.

These classes handle:
- Model loading and management
- Memory buffer operations
- Keyframe extraction
- Video generation
- Tensor format conversion
"""

from .model_loader import StoryMemModelManager
from .memory_manager import StoryMemMemory, create_memory_buffer
from .keyframe_extractor import KeyframeExtractor, extract_keyframes_simple
from .shot_generator import StoryMemShotGenerator, create_shot_generator
from .tensor_utils import (
    comfyui_to_storymem,
    storymem_to_comfyui,
    numpy_to_tensor,
    tensor_to_numpy,
)

__all__ = [
    'StoryMemModelManager',
    'StoryMemMemory',
    'create_memory_buffer',
    'KeyframeExtractor',
    'extract_keyframes_simple',
    'StoryMemShotGenerator',
    'create_shot_generator',
    'comfyui_to_storymem',
    'storymem_to_comfyui',
    'numpy_to_tensor',
    'tensor_to_numpy',
]
