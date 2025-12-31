"""
Bridge module for integrating StoryMem with ComfyUI.

This module provides a simplified API that wraps the StoryMem repository,
making it easier to use in ComfyUI nodes.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from easydict import EasyDict

# Add storymem_src to path
STORYMEM_PATH = Path(__file__).parent.parent / "storymem_src"
if str(STORYMEM_PATH) not in sys.path:
    sys.path.insert(0, str(STORYMEM_PATH))

try:
    # Import StoryMem/Wan modules
    from wan.text2video import WanT2V
    from wan.memory2video import WanM2V
    from wan.textimage2video import WanTI2V
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    from wan.utils.utils import save_video
    STORYMEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"StoryMem modules not available: {e}")
    STORYMEM_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_storymem_available() -> bool:
    """Check if StoryMem is available."""
    return STORYMEM_AVAILABLE


def get_wan_config(size_str: str = "832*480") -> EasyDict:
    """
    Get Wan model configuration for given size.

    Args:
        size_str: Size string like "832*480"

    Returns:
        EasyDict configuration
    """
    if not STORYMEM_AVAILABLE:
        raise RuntimeError("StoryMem is not available")

    # Parse size
    width, height = map(int, size_str.split('*'))
    size = (height, width)  # Wan uses (height, width)

    # Get config for this size
    if size in SIZE_CONFIGS:
        config_name = SIZE_CONFIGS[size]
        config = WAN_CONFIGS[config_name]
        return config
    else:
        # Use default config
        logging.warning(f"Size {size} not in SIZE_CONFIGS, using default")
        return WAN_CONFIGS['A14B']


class StoryMemPipeline:
    """
    Unified pipeline for StoryMem video generation.

    Handles T2V, M2V, MI2V, and MM2V modes.
    """

    def __init__(
        self,
        t2v_checkpoint_dir: str,
        i2v_checkpoint_dir: str,
        m2v_lora_path: Optional[str] = None,
        device: str = "cuda",
        enable_offload: bool = True,
        t5_cpu: bool = False,
    ):
        """
        Initialize StoryMem pipeline.

        Args:
            t2v_checkpoint_dir: Path to T2V model
            i2v_checkpoint_dir: Path to I2V model
            m2v_lora_path: Path to M2V LoRA weights
            device: Target device
            enable_offload: Enable model offloading
            t5_cpu: Place T5 on CPU
        """
        if not STORYMEM_AVAILABLE:
            raise RuntimeError(
                "StoryMem is not available. Please ensure the storymem_src submodule "
                "is initialized: git submodule update --init --recursive"
            )

        self.t2v_checkpoint_dir = t2v_checkpoint_dir
        self.i2v_checkpoint_dir = i2v_checkpoint_dir
        self.m2v_lora_path = m2v_lora_path
        self.device = torch.device(device)
        self.enable_offload = enable_offload
        self.t5_cpu = t5_cpu

        # Pipeline storage (lazy loaded)
        self._t2v_pipeline = None
        self._m2v_pipeline = None
        self._mi2v_pipeline = None

        logger.info(f"StoryMemPipeline initialized on {device}")

    def load_t2v_pipeline(self, config: EasyDict):
        """Load T2V pipeline if not already loaded."""
        if self._t2v_pipeline is not None:
            return self._t2v_pipeline

        logger.info("Loading T2V pipeline...")

        self._t2v_pipeline = WanT2V(
            config=config,
            checkpoint_dir=self.t2v_checkpoint_dir,
            device_id=self.device.index if self.device.type == 'cuda' else 0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.t5_cpu,
            init_on_cpu=self.enable_offload,
            convert_model_dtype=False,
        )

        logger.info("✓ T2V pipeline loaded")
        return self._t2v_pipeline

    def load_m2v_pipeline(self, config: EasyDict, lora_path: Optional[str] = None):
        """Load M2V pipeline if not already loaded."""
        if self._m2v_pipeline is not None:
            return self._m2v_pipeline

        logger.info("Loading M2V pipeline...")

        lora_path = lora_path or self.m2v_lora_path

        self._m2v_pipeline = WanM2V(
            config=config,
            checkpoint_dir=self.i2v_checkpoint_dir,
            lora_weight_path=lora_path,
            device_id=self.device.index if self.device.type == 'cuda' else 0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.t5_cpu,
            init_on_cpu=self.enable_offload,
            convert_model_dtype=False,
        )

        logger.info("✓ M2V pipeline loaded")
        return self._m2v_pipeline

    def load_mi2v_pipeline(self, config: EasyDict, lora_path: Optional[str] = None):
        """Load MI2V pipeline if not already loaded."""
        if self._mi2v_pipeline is not None:
            return self._mi2v_pipeline

        logger.info("Loading MI2V pipeline...")

        lora_path = lora_path or self.m2v_lora_path

        self._mi2v_pipeline = WanTI2V(
            config=config,
            checkpoint_dir=self.i2v_checkpoint_dir,
            lora_weight_path=lora_path,
            device_id=self.device.index if self.device.type == 'cuda' else 0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.t5_cpu,
            init_on_cpu=self.enable_offload,
            convert_model_dtype=False,
        )

        logger.info("✓ MI2V pipeline loaded")
        return self._mi2v_pipeline

    def generate_t2v(
        self,
        prompt: str,
        num_frames: int = 25,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
        steps: int = 50,
        cfg_scale: float = 3.5,
    ) -> torch.Tensor:
        """
        Generate video from text using T2V.

        Args:
            prompt: Text prompt
            num_frames: Number of frames
            height: Video height
            width: Video width
            seed: Random seed
            steps: Sampling steps
            cfg_scale: CFG scale

        Returns:
            Video tensor [B, C, F, H, W] in StoryMem format
        """
        size_str = f"{width}*{height}"
        config = get_wan_config(size_str)

        # Load pipeline
        pipeline = self.load_t2v_pipeline(config)

        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Generate
        logger.info(f"Generating T2V: {prompt[:50]}...")

        video = pipeline.generate(
            prompt=prompt,
            num_frames=num_frames,
            size=(height, width),
            sample_steps=steps,
            sample_guide_scale=cfg_scale,
            offload_model=self.enable_offload,
        )

        return video

    def generate_m2v(
        self,
        prompt: str,
        memory_frames: List[torch.Tensor],
        num_frames: int = 25,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
        steps: int = 50,
        cfg_scale: float = 3.5,
    ) -> torch.Tensor:
        """
        Generate video from text + memory using M2V.

        Args:
            prompt: Text prompt
            memory_frames: List of memory keyframes
            num_frames: Number of frames
            height: Video height
            width: Video width
            seed: Random seed
            steps: Sampling steps
            cfg_scale: CFG scale

        Returns:
            Video tensor [B, C, F, H, W]
        """
        size_str = f"{width}*{height}"
        config = get_wan_config(size_str)

        # Load pipeline
        pipeline = self.load_m2v_pipeline(config)

        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Generate
        logger.info(f"Generating M2V with {len(memory_frames)} memory frames...")

        video = pipeline.generate(
            prompt=prompt,
            memory_frames=memory_frames,
            num_frames=num_frames,
            size=(height, width),
            sample_steps=steps,
            sample_guide_scale=cfg_scale,
            offload_model=self.enable_offload,
        )

        return video

    def unload_models(self):
        """Unload all pipelines to free memory."""
        if self._t2v_pipeline is not None:
            del self._t2v_pipeline
            self._t2v_pipeline = None

        if self._m2v_pipeline is not None:
            del self._m2v_pipeline
            self._m2v_pipeline = None

        if self._mi2v_pipeline is not None:
            del self._mi2v_pipeline
            self._mi2v_pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded")
