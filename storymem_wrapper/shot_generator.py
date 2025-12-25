"""
Shot generation wrapper for StoryMem.

Wraps StoryMem's video generation pipeline and adapts it for ComfyUI:
- Text prompt to video generation (T2V, M2V, MI2V, MM2V)
- Memory conditioning
- Shot-by-shot generation
"""

import torch
from typing import Dict, Any, List, Optional, Literal
import logging

from .memory_manager import StoryMemMemory
from .tensor_utils import storymem_to_comfyui, comfyui_to_storymem

logger = logging.getLogger(__name__)


class StoryMemShotGenerator:
    """
    Wrapper around StoryMem's generation pipeline.

    Handles:
    - Text-to-video generation (first shot)
    - Memory-conditioned generation (continuation shots)
    - Multiple generation modes (M2V, MI2V, MM2V)
    """

    def __init__(self, models_dict: Dict[str, Any]):
        """
        Initialize shot generator with loaded models.

        Args:
            models_dict: Dictionary containing model paths and configuration
                - 't2v_path': Path to T2V model
                - 'i2v_path': Path to I2V model
                - 'm2v_path': Path to M2V LoRA
                - 'm2v_type': Type of M2V LoRA ('MI2V' or 'MM2V')
                - 'device': Target device
                - 'enable_offload': Whether offloading is enabled
        """
        self.models_dict = models_dict
        self.device = torch.device(models_dict.get('device', 'cuda'))
        self.m2v_type = models_dict.get('m2v_type', 'MI2V')

        # Pipeline storage (lazy loaded)
        self.t2v_pipeline = None
        self.i2v_pipeline = None
        self.m2v_pipeline = None

        logger.info(f"StoryMemShotGenerator initialized with {self.m2v_type} LoRA")

    def generate_first_shot(
        self,
        prompt: str,
        num_frames: int = 25,
        width: int = 832,
        height: int = 480,
        seed: int = 0,
        steps: int = 50,
        cfg_scale: float = 7.5,
        use_t2v: bool = True,
        init_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate the first shot of a story using T2V or M2V (without memory).

        Args:
            prompt: Text description of the shot
            num_frames: Number of frames to generate (must be 4n+1)
            width: Video width
            height: Video height
            seed: Random seed for reproducibility
            steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            use_t2v: If True, use T2V; if False, use M2V without memory
            init_image: Optional initial image for conditioning

        Returns:
            video_tensor: Generated video in ComfyUI format [1, F, H, W, C]
        """
        logger.info(f"Generating first shot: '{prompt[:50]}...'")
        logger.info(f"  Mode: {'T2V' if use_t2v else 'M2V (no memory)'}")
        logger.info(f"  Frames: {num_frames}, Size: {width}x{height}")
        logger.info(f"  Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}")

        # Validate frame count (must be 4n+1)
        if (num_frames - 1) % 4 != 0:
            raise ValueError(f"num_frames must be 4n+1 format, got {num_frames}")

        # Check if pipeline is available
        pipeline = self.models_dict.get('pipeline')
        if pipeline is None:
            logger.warning("StoryMem pipeline not available, returning placeholder")
            video = torch.rand(1, num_frames, height, width, 3)
            return video

        # Generate with actual pipeline
        try:
            if use_t2v:
                # Use T2V pipeline
                video_storymem = pipeline.generate_t2v(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=seed,
                    steps=steps,
                    cfg_scale=cfg_scale,
                )
            else:
                # Use M2V with empty memory
                video_storymem = pipeline.generate_m2v(
                    prompt=prompt,
                    memory_frames=[],
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=seed,
                    steps=steps,
                    cfg_scale=cfg_scale,
                )

            # Convert from StoryMem format [B,C,F,H,W] to ComfyUI format [B,F,H,W,C]
            video = storymem_to_comfyui(video_storymem)

            logger.info(f"✓ Generated video: {video.shape}")
            return video

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.warning("Returning placeholder video")
            video = torch.rand(1, num_frames, height, width, 3)
            return video

    def generate_continuation(
        self,
        prompt: str,
        memory: StoryMemMemory,
        generation_mode: Literal['M2V', 'MI2V', 'MM2V'] = 'MI2V',
        num_frames: int = 25,
        seed: int = 0,
        steps: int = 50,
        cfg_scale: float = 7.5,
        is_scene_cut: bool = False,
        first_frame: Optional[torch.Tensor] = None,
        motion_frames: Optional[List[torch.Tensor]] = None,
        width: int = 832,
        height: int = 480,
    ) -> torch.Tensor:
        """
        Generate a continuation shot with memory conditioning.

        Args:
            prompt: Text description of the shot
            memory: Memory buffer with keyframes
            generation_mode: Generation mode ('M2V', 'MI2V', 'MM2V')
            num_frames: Number of frames to generate
            seed: Random seed
            steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            is_scene_cut: Whether this is a scene cut (affects memory handling)
            first_frame: First frame for MI2V conditioning
            motion_frames: Motion frames for MM2V conditioning
            width: Video width
            height: Video height

        Returns:
            video_tensor: Generated video in ComfyUI format [1, F, H, W, C]

        TODO: Implement once StoryMem is integrated
        """
        logger.info(f"Generating continuation shot: '{prompt[:50]}...'")
        logger.info(f"  Mode: {generation_mode}, Scene cut: {is_scene_cut}")
        logger.info(f"  Memory: {len(memory)} keyframes")
        logger.info(f"  Frames: {num_frames}, Seed: {seed}")

        # Validate mode against loaded LoRA
        if generation_mode == 'MM2V' and self.m2v_type != 'MM2V':
            logger.warning(f"MM2V mode requested but {self.m2v_type} LoRA loaded. Using MI2V.")
            generation_mode = 'MI2V'

        # Validate frame count
        if (num_frames - 1) % 4 != 0:
            raise ValueError(f"num_frames must be 4n+1 format, got {num_frames}")

        # Prepare memory input
        memory_frames = memory.get_memory_frames() if not is_scene_cut else []

        # Validate conditioning inputs
        if generation_mode == 'MI2V' and first_frame is None:
            logger.warning("MI2V mode requires first_frame, falling back to M2V")
            generation_mode = 'M2V'

        if generation_mode == 'MM2V' and motion_frames is None:
            logger.warning("MM2V mode requires motion_frames, falling back to MI2V")
            generation_mode = 'MI2V'

        # Check if pipeline is available
        pipeline = self.models_dict.get('pipeline')
        if pipeline is None:
            logger.warning("StoryMem pipeline not available, returning placeholder")
            video = torch.rand(1, num_frames, height, width, 3)
            return video

        # Generate with actual pipeline
        try:
            if generation_mode == 'M2V':
                # Memory-only generation
                video_storymem = pipeline.generate_m2v(
                    prompt=prompt,
                    memory_frames=memory_frames,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=seed,
                    steps=steps,
                    cfg_scale=cfg_scale,
                )
            else:
                # MI2V or MM2V (handled by pipeline)
                # Note: Actual implementation depends on StoryMem's API
                logger.warning(f"{generation_mode} mode using M2V as fallback")
                video_storymem = pipeline.generate_m2v(
                    prompt=prompt,
                    memory_frames=memory_frames,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=seed,
                    steps=steps,
                    cfg_scale=cfg_scale,
                )

            # Convert from StoryMem format to ComfyUI format
            video = storymem_to_comfyui(video_storymem)

            logger.info(f"✓ Generated continuation video: {video.shape}")
            return video

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.warning("Returning placeholder video")
            video = torch.rand(1, num_frames, height, width, 3)
            return video

    def _prepare_memory_input(
        self,
        memory_frames: List[torch.Tensor],
        mode: str
    ) -> Any:
        """
        Prepare memory frames for StoryMem pipeline input.

        Args:
            memory_frames: List of keyframe tensors [H, W, C]
            mode: Generation mode

        Returns:
            Memory input in StoryMem expected format

        TODO: Implement once StoryMem format is known
        """
        if not memory_frames:
            return None

        # Placeholder: Stack frames into batch
        # Actual format depends on StoryMem's expectations
        memory_batch = torch.stack(memory_frames, dim=0)  # [N, H, W, C]
        return memory_batch

    def _create_shot_config(
        self,
        prompt: str,
        shot_index: int = 0,
        scene_index: int = 1,
        is_cut: bool = False
    ) -> Dict[str, Any]:
        """
        Convert text prompt to StoryMem's JSON structure.

        Args:
            prompt: Text description
            shot_index: Shot number
            scene_index: Scene number
            is_cut: Whether this is a scene cut

        Returns:
            Dictionary in StoryMem's expected format
        """
        return {
            "story": "Generated story",  # Could accumulate from all prompts
            "scenes": [
                {
                    "id": scene_index,
                    "shot_id": shot_index,
                    "is_cut": is_cut,
                    "video_prompt": prompt
                }
            ]
        }

    def _load_pipelines(self):
        """
        Lazy load StoryMem pipelines.

        TODO: Implement once StoryMem is integrated
        """
        if self.t2v_pipeline is not None:
            return  # Already loaded

        logger.info("Loading StoryMem pipelines...")

        # TODO: Load pipelines from paths
        # from storymem.pipeline import WanT2V, WanI2V, WanM2V
        # self.t2v_pipeline = WanT2V(...)
        # self.i2v_pipeline = WanI2V(...)
        # self.m2v_pipeline = WanM2V(...)

        logger.warning("Pipeline loading not yet implemented")


def create_shot_generator(models_dict: Dict[str, Any]) -> StoryMemShotGenerator:
    """
    Factory function to create a shot generator.

    Args:
        models_dict: Model configuration dictionary

    Returns:
        Initialized StoryMemShotGenerator
    """
    return StoryMemShotGenerator(models_dict)
