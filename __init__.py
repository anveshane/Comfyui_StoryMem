"""
ComfyUI StoryMem Custom Nodes

Multi-shot long video storytelling with memory-conditioned video diffusion models.
Wraps the StoryMem repository (https://github.com/Kevin-thu/StoryMem) for use in ComfyUI.
"""

from .nodes.loader_nodes import StoryMemModelLoader
from .nodes.memory_nodes import StoryMemMemoryBuffer, StoryMemUpdateMemory
from .nodes.shot_nodes import StoryMemFirstShot, StoryMemContinuationShot
from .nodes.utility_nodes import (
    StoryMemMemoryVisualizer,
    StoryMemVideoCombine,
    StoryMemGetMotionFrames,
    StoryMemInfo,
)

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    # Model loading
    "StoryMemModelLoader": StoryMemModelLoader,

    # Memory management
    "StoryMemMemoryBuffer": StoryMemMemoryBuffer,
    "StoryMemUpdateMemory": StoryMemUpdateMemory,

    # Shot generation
    "StoryMemFirstShot": StoryMemFirstShot,
    "StoryMemContinuationShot": StoryMemContinuationShot,

    # Utilities
    "StoryMemMemoryVisualizer": StoryMemMemoryVisualizer,
    "StoryMemVideoCombine": StoryMemVideoCombine,
    "StoryMemGetMotionFrames": StoryMemGetMotionFrames,
    "StoryMemInfo": StoryMemInfo,
}

# Display names for nodes in ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Model loading
    "StoryMemModelLoader": "StoryMem Model Loader",

    # Memory management
    "StoryMemMemoryBuffer": "StoryMem Memory Buffer",
    "StoryMemUpdateMemory": "StoryMem Update Memory",

    # Shot generation
    "StoryMemFirstShot": "StoryMem First Shot",
    "StoryMemContinuationShot": "StoryMem Continuation Shot",

    # Utilities
    "StoryMemMemoryVisualizer": "StoryMem Memory Visualizer",
    "StoryMemVideoCombine": "StoryMem Video Combine",
    "StoryMemGetMotionFrames": "StoryMem Get Motion Frames",
    "StoryMemInfo": "StoryMem Info",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
