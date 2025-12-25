"""
Model loader nodes for StoryMem.

Handles loading and caching of StoryMem models:
- Wan2.2-T2V-A14B (Text-to-Video)
- Wan2.2-I2V-A14B (Image-to-Video)
- StoryMem M2V LoRA weights
"""

from ..storymem_wrapper.model_loader import StoryMemModelManager, get_available_models


class StoryMemModelLoader:
    """
    Load and cache StoryMem models for video generation.

    This node loads the three required models and returns a models dictionary
    that can be passed to shot generation nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Try to get available models, fallback to defaults
        try:
            t2v_models = get_available_models("t2v")
            i2v_models = get_available_models("i2v")
        except:
            t2v_models = ["Wan2.2-T2V-A14B"]
            i2v_models = ["Wan2.2-I2V-A14B"]

        # Ensure at least default options
        if not t2v_models:
            t2v_models = ["Wan2.2-T2V-A14B"]
        if not i2v_models:
            i2v_models = ["Wan2.2-I2V-A14B"]

        return {
            "required": {
                "t2v_model": (t2v_models, {
                    "default": t2v_models[0],
                }),
                "i2v_model": (i2v_models, {
                    "default": i2v_models[0],
                }),
                "m2v_lora": (["MI2V", "MM2V"], {
                    "default": "MI2V",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                }),
                "enable_offload": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
            },
            "optional": {
                "use_fsdp": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                }),
            }
        }

    RETURN_TYPES = ("STORYMEM_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_models"
    CATEGORY = "StoryMem/loaders"
    DESCRIPTION = "Load StoryMem models (T2V, I2V, M2V LoRA) for video generation. Models require ~40-60GB storage and 80GB VRAM (or enable offloading)."

    def load_models(
        self,
        t2v_model: str,
        i2v_model: str,
        m2v_lora: str,
        device: str,
        enable_offload: bool,
        use_fsdp: bool = False,
    ):
        """
        Load StoryMem models.

        Args:
            t2v_model: Text-to-Video model name
            i2v_model: Image-to-Video model name
            m2v_lora: M2V LoRA type (MI2V or MM2V)
            device: Target device (auto/cuda/cpu)
            enable_offload: Enable CPU offloading for VRAM management
            use_fsdp: Use Fully Sharded Data Parallel

        Returns:
            Tuple containing models dictionary
        """
        try:
            # Create model manager
            manager = StoryMemModelManager(
                device=device if device != "auto" else None,
                enable_offload=enable_offload,
                use_fsdp=use_fsdp,
            )

            # Load models
            models_dict = manager.load_models(
                t2v_model=t2v_model,
                i2v_model=i2v_model,
                m2v_lora=m2v_lora,
            )

            print(f"✓ StoryMem models loaded successfully")
            print(f"  T2V: {t2v_model}")
            print(f"  I2V: {i2v_model}")
            print(f"  M2V LoRA: {m2v_lora}")
            print(f"  Device: {models_dict['device']}")
            print(f"  Offloading: {'enabled' if enable_offload else 'disabled'}")

            return (models_dict,)

        except FileNotFoundError as e:
            print(f"\n✗ Model loading failed: {e}")
            print("\nTo download models, run:")
            print("  python custom_nodes/comfyui_storymem/scripts/download_models.py")
            raise

        except Exception as e:
            print(f"\n✗ Model loading failed: {e}")
            raise
