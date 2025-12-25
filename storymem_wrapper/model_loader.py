"""
Model loading and management for StoryMem models.

Handles loading, caching, and VRAM management for:
- Wan2.2-T2V-A14B (Text-to-Video)
- Wan2.2-I2V-A14B (Image-to-Video)
- StoryMem M2V LoRA weights
"""

import os
import torch
from typing import Optional, Dict, Any, Literal
import logging

# Try to import ComfyUI's folder_paths for model location
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logging.warning("ComfyUI folder_paths not available, using fallback paths")

# Import StoryMem bridge
from .storymem_bridge import StoryMemPipeline, is_storymem_available

logger = logging.getLogger(__name__)


class StoryMemModelManager:
    """
    Manages loading and caching of StoryMem models.

    Handles:
    - Model loading with lazy initialization
    - VRAM management and offloading
    - Model caching and reuse
    - ComfyUI integration via folder_paths
    """

    def __init__(
        self,
        device: Optional[str] = None,
        enable_offload: bool = True,
        use_fsdp: bool = False,
    ):
        """
        Initialize model manager.

        Args:
            device: Target device ('cuda', 'cpu', or None for auto)
            enable_offload: Enable CPU offloading to manage VRAM
            use_fsdp: Use Fully Sharded Data Parallel for distributed inference
        """
        self.device = self._get_device(device)
        self.enable_offload = enable_offload
        self.use_fsdp = use_fsdp

        # Model storage
        self.t2v_pipeline = None
        self.i2v_pipeline = None
        self.m2v_lora = None
        self.m2v_type = None  # 'MI2V' or 'MM2V'

        # Model paths
        self.model_dir = self._get_model_dir()

        logger.info(f"StoryMemModelManager initialized on device: {self.device}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Offloading enabled: {self.enable_offload}")

    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the target device."""
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)

    def _get_model_dir(self) -> str:
        """Get the model directory, using ComfyUI's folder_paths if available."""
        if COMFYUI_AVAILABLE:
            # Register storymem model directory with ComfyUI
            storymem_dir = os.path.join(folder_paths.models_dir, "storymem")

            # Register if not already registered
            if "storymem" not in folder_paths.folder_names_and_paths:
                folder_paths.folder_names_and_paths["storymem"] = (
                    [storymem_dir],
                    {".safetensors", ".bin", ".pth", ".pt"}
                )

            return storymem_dir
        else:
            # Fallback to relative path
            return os.path.join(os.path.dirname(__file__), "..", "models")

    def check_vram_availability(self, required_gb: float = 80) -> bool:
        """
        Check if sufficient VRAM is available.

        Args:
            required_gb: Required VRAM in GB

        Returns:
            True if sufficient VRAM available

        Raises:
            RuntimeError: If insufficient VRAM and offloading is disabled
        """
        if not torch.cuda.is_available():
            if not self.enable_offload:
                raise RuntimeError(
                    "CUDA not available. Enable CPU offloading or use CPU device."
                )
            logger.warning("CUDA not available, using CPU")
            return False

        # Get available VRAM
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / (1024**3)
        free_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated(0)
        ) / (1024**3)

        logger.info(f"VRAM: {free_memory_gb:.1f}GB free / {total_memory_gb:.1f}GB total")

        if free_memory_gb < required_gb:
            if self.enable_offload:
                logger.warning(
                    f"Insufficient VRAM ({free_memory_gb:.1f}GB < {required_gb}GB). "
                    f"Using CPU offloading."
                )
                return False
            else:
                raise RuntimeError(
                    f"Insufficient VRAM: {free_memory_gb:.1f}GB available, "
                    f"{required_gb}GB required. Enable offloading or reduce model size."
                )

        return True

    def load_models(
        self,
        t2v_model: str = "Wan2.2-T2V-A14B",
        i2v_model: str = "Wan2.2-I2V-A14B",
        m2v_lora: Literal["MI2V", "MM2V"] = "MI2V",
    ) -> Dict[str, Any]:
        """
        Load all StoryMem models.

        Args:
            t2v_model: Text-to-Video model name
            i2v_model: Image-to-Video model name
            m2v_lora: M2V LoRA type (MI2V or MM2V)

        Returns:
            Dictionary containing loaded models
        """
        logger.info("Loading StoryMem models...")
        logger.info(f"  T2V: {t2v_model}")
        logger.info(f"  I2V: {i2v_model}")
        logger.info(f"  M2V LoRA: {m2v_lora}")

        # Check if StoryMem is available
        if not is_storymem_available():
            logger.warning(
                "StoryMem modules not available. Please ensure:\n"
                "  1. Git submodule is initialized: git submodule update --init --recursive\n"
                "  2. Dependencies are installed: pip install -r requirements.txt\n"
                "Returning configuration dict without actual pipelines."
            )
            # Return config without pipeline (fallback mode)
            return self._create_config_dict(t2v_model, i2v_model, m2v_lora)

        # Check VRAM
        try:
            self.check_vram_availability(required_gb=80)
        except RuntimeError as e:
            logger.error(f"VRAM check failed: {e}")
            if not self.enable_offload:
                raise

        # Build model paths
        t2v_path = os.path.join(self.model_dir, t2v_model)
        i2v_path = os.path.join(self.model_dir, i2v_model)
        m2v_path = os.path.join(self.model_dir, "StoryMem", f"Wan2.2-{m2v_lora}-A14B")

        # Validate paths exist
        self._validate_model_paths(t2v_path, i2v_path, m2v_path)

        # Create StoryMem pipeline
        try:
            pipeline = StoryMemPipeline(
                t2v_checkpoint_dir=t2v_path,
                i2v_checkpoint_dir=i2v_path,
                m2v_lora_path=m2v_path,
                device=str(self.device),
                enable_offload=self.enable_offload,
                t5_cpu=False,  # TODO: Make this configurable
            )

            self.m2v_type = m2v_lora

            models_dict = {
                'pipeline': pipeline,
                't2v_path': t2v_path,
                'i2v_path': i2v_path,
                'm2v_path': m2v_path,
                'm2v_type': m2v_lora,
                'device': str(self.device),
                'enable_offload': self.enable_offload,
                'manager': self,
            }

            logger.info("✓ StoryMem models loaded successfully")
            return models_dict

        except Exception as e:
            logger.error(f"Failed to load StoryMem pipeline: {e}")
            logger.warning("Returning configuration dict without pipeline")
            return self._create_config_dict(t2v_model, i2v_lora, m2v_lora)

    def _create_config_dict(self, t2v_model: str, i2v_model: str, m2v_lora: str) -> Dict[str, Any]:
        """Create a configuration dictionary without actual pipeline (fallback mode)."""
        t2v_path = os.path.join(self.model_dir, t2v_model)
        i2v_path = os.path.join(self.model_dir, i2v_model)
        m2v_path = os.path.join(self.model_dir, "StoryMem", f"Wan2.2-{m2v_lora}-A14B")

        return {
            'pipeline': None,
            't2v_path': t2v_path,
            'i2v_path': i2v_path,
            'm2v_path': m2v_path,
            'm2v_type': m2v_lora,
            'device': str(self.device),
            'enable_offload': self.enable_offload,
            'manager': self,
        }

    def _validate_model_paths(
        self,
        t2v_path: str,
        i2v_path: str,
        m2v_path: str
    ):
        """
        Validate that model paths exist.

        Raises:
            FileNotFoundError: If any model path doesn't exist
        """
        missing = []

        if not os.path.exists(t2v_path):
            missing.append(f"T2V model: {t2v_path}")
        if not os.path.exists(i2v_path):
            missing.append(f"I2V model: {i2v_path}")
        if not os.path.exists(m2v_path):
            missing.append(f"M2V LoRA: {m2v_path}")

        if missing:
            error_msg = (
                "Missing model files:\n" + "\n".join(f"  - {m}" for m in missing) +
                "\n\nPlease download models using: python scripts/download_models.py"
            )
            raise FileNotFoundError(error_msg)

    def load_t2v_pipeline(self, model_path: str):
        """
        Load Text-to-Video pipeline.

        Args:
            model_path: Path to T2V model

        TODO: Implement once StoryMem is integrated
        """
        logger.info(f"Loading T2V pipeline from {model_path}")

        # Placeholder: Will import and initialize StoryMem's T2V pipeline
        # from storymem.pipeline import WanT2V
        # self.t2v_pipeline = WanT2V(
        #     checkpoint_dir=model_path,
        #     device=self.device,
        #     enable_offload=self.enable_offload,
        # )

        raise NotImplementedError(
            "T2V pipeline loading will be implemented once StoryMem is integrated"
        )

    def load_i2v_pipeline(self, model_path: str):
        """
        Load Image-to-Video pipeline.

        Args:
            model_path: Path to I2V model

        TODO: Implement once StoryMem is integrated
        """
        logger.info(f"Loading I2V pipeline from {model_path}")

        # Placeholder: Will import and initialize StoryMem's I2V pipeline
        raise NotImplementedError(
            "I2V pipeline loading will be implemented once StoryMem is integrated"
        )

    def load_m2v_lora(self, model_path: str, lora_type: str):
        """
        Load M2V LoRA weights.

        Args:
            model_path: Path to LoRA weights
            lora_type: Type of LoRA (MI2V or MM2V)

        TODO: Implement once StoryMem is integrated
        """
        logger.info(f"Loading M2V LoRA ({lora_type}) from {model_path}")

        # Placeholder: Will load LoRA weights
        raise NotImplementedError(
            "M2V LoRA loading will be implemented once StoryMem is integrated"
        )

    def unload_models(self):
        """Unload all models and free VRAM."""
        logger.info("Unloading models...")

        if self.t2v_pipeline is not None:
            del self.t2v_pipeline
            self.t2v_pipeline = None

        if self.i2v_pipeline is not None:
            del self.i2v_pipeline
            self.i2v_pipeline = None

        if self.m2v_lora is not None:
            del self.m2v_lora
            self.m2v_lora = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded, VRAM freed")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            't2v_loaded': self.t2v_pipeline is not None,
            'i2v_loaded': self.i2v_pipeline is not None,
            'm2v_loaded': self.m2v_lora is not None,
            'm2v_type': self.m2v_type,
            'device': str(self.device),
            'offload_enabled': self.enable_offload,
        }


def get_available_models(model_type: Literal["t2v", "i2v", "m2v"]) -> list[str]:
    """
    Get list of available models of specified type.

    Args:
        model_type: Type of model to list

    Returns:
        List of available model names
    """
    manager = StoryMemModelManager()
    model_dir = manager.model_dir

    if not os.path.exists(model_dir):
        return []

    if model_type == "t2v":
        # Look for T2V models
        return [d for d in os.listdir(model_dir) if "T2V" in d and os.path.isdir(os.path.join(model_dir, d))]
    elif model_type == "i2v":
        # Look for I2V models
        return [d for d in os.listdir(model_dir) if "I2V" in d and os.path.isdir(os.path.join(model_dir, d))]
    elif model_type == "m2v":
        # M2V LoRA types
        return ["MI2V", "MM2V"]

    return []
