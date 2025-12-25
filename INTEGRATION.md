# StoryMem Integration Guide

This document explains how the StoryMem repository is integrated with ComfyUI nodes.

## Architecture Overview

The integration uses a **bridge pattern** to wrap the StoryMem repository without modifying its code:

```
ComfyUI Nodes → storymem_wrapper → storymem_bridge → storymem_src (submodule)
```

### Components

1. **storymem_src/** - Git submodule of the original StoryMem repository
2. **storymem_wrapper/storymem_bridge.py** - Bridge module providing simplified API
3. **storymem_wrapper/model_loader.py** - Creates StoryMemPipeline instances
4. **storymem_wrapper/shot_generator.py** - Uses pipeline for video generation
5. **nodes/** - ComfyUI node implementations

## Setup Instructions

### 1. Initialize Submodule

The StoryMem repository is added as a git submodule:

```bash
# Initialize and update submodule
git submodule update --init --recursive

# Or if cloning fresh
git clone --recursive <repo-url>
```

### 2. Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt

# Optional: Install performance optimizations
pip install flash-attn --no-build-isolation
pip install xformers
```

### 3. Download Models

Download the required models (~40-60GB):

```bash
python scripts/download_models.py
```

Or manually:

```bash
# Create models directory
mkdir -p ../../models/storymem

# Download T2V model
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ../../models/storymem/Wan2.2-T2V-A14B

# Download I2V model
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ../../models/storymem/Wan2.2-I2V-A14B

# Download M2V LoRA
huggingface-cli download Kevin-thu/StoryMem --local-dir ../../models/storymem/StoryMem
```

## Integration Details

### StoryMemPipeline

The bridge module (`storymem_bridge.py`) provides a unified `StoryMemPipeline` class:

```python
class StoryMemPipeline:
    def __init__(
        self,
        t2v_checkpoint_dir: str,
        i2v_checkpoint_dir: str,
        m2v_lora_path: str,
        device: str = "cuda",
        enable_offload: bool = True,
    ):
        ...

    def generate_t2v(...) -> torch.Tensor:
        """Generate video from text using T2V model."""
        ...

    def generate_m2v(...) -> torch.Tensor:
        """Generate video from text + memory using M2V model."""
        ...
```

### Tensor Format Conversion

Videos are automatically converted between formats:

- **StoryMem format**: `[B, C, F, H, W]` (channels first)
- **ComfyUI format**: `[B, F, H, W, C]` (channels last)

Conversion handled by `tensor_utils.py`:
- `storymem_to_comfyui()` - Convert StoryMem → ComfyUI
- `comfyui_to_storymem()` - Convert ComfyUI → StoryMem

### Model Loading Flow

```
1. User clicks "StoryMem Model Loader" node
   ↓
2. StoryMemModelManager.load_models()
   - Validates model paths
   - Checks VRAM availability
   - Creates StoryMemPipeline
   ↓
3. Returns models dict with pipeline
   {
       'pipeline': StoryMemPipeline instance,
       't2v_path': path,
       'i2v_path': path,
       'm2v_path': path,
       ...
   }
   ↓
4. Pipeline passed to generation nodes
```

### Video Generation Flow

```
1. User connects nodes and sets prompts
   ↓
2. StoryMemFirstShot.generate()
   - Creates StoryMemShotGenerator
   - Calls generate_first_shot()
   ↓
3. StoryMemShotGenerator.generate_first_shot()
   - Gets pipeline from models dict
   - Calls pipeline.generate_t2v()
   - Converts output format
   ↓
4. Returns video tensor [1, F, H, W, C]
   + memory buffer updated with keyframes
```

## Fallback Mode

If StoryMem modules are not available (e.g., submodule not initialized), the system operates in **fallback mode**:

- Model loader returns configuration dict without pipeline
- Shot generator returns placeholder videos (random tensors)
- User sees warnings in console

This allows the node structure to work even without full integration.

## Customization Points

### Adding New Generation Modes

To add support for additional modes (e.g., MM2V fully implemented):

1. **Add method to StoryMemPipeline** (`storymem_bridge.py`):
   ```python
   def generate_mm2v(self, prompt, memory_frames, motion_frames, ...):
       pipeline = self.load_mi2v_pipeline(config)
       video = pipeline.generate(...)
       return video
   ```

2. **Update ShotGenerator** (`shot_generator.py`):
   ```python
   elif generation_mode == 'MM2V':
       video_storymem = pipeline.generate_mm2v(
           prompt=prompt,
           memory_frames=memory_frames,
           motion_frames=motion_frames_list,
           ...
       )
   ```

3. **Update node INPUT_TYPES** if needed

### Customizing Model Paths

By default, models are loaded from `ComfyUI/models/storymem/`. To change:

1. **Using ComfyUI's folder_paths**:
   - Models automatically detected in registered paths
   - Add custom paths via ComfyUI's model path settings

2. **Manual override**:
   ```python
   # In model_loader.py
   def _get_model_dir(self):
       return "/custom/path/to/models"
   ```

### Adjusting VRAM Management

The system includes several VRAM optimization options:

- **enable_offload**: Move models to CPU when not in use
- **t5_cpu**: Keep T5 encoder on CPU
- **use_fsdp**: Fully Sharded Data Parallel (multi-GPU)

Configure in `StoryMemModelLoader` node or modify defaults:

```python
# In model_loader.py
pipeline = StoryMemPipeline(
    ...,
    enable_offload=True,  # Default: True
    t5_cpu=False,         # Default: False
)
```

## Debugging

### Check StoryMem Availability

```python
from storymem_wrapper.storymem_bridge import is_storymem_available

if is_storymem_available():
    print("✓ StoryMem modules loaded")
else:
    print("✗ StoryMem not available")
    print("Run: git submodule update --init --recursive")
```

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **"StoryMem modules not available"**
   - Solution: `git submodule update --init --recursive`

2. **"Model files not found"**
   - Solution: Run `python scripts/download_models.py`

3. **CUDA out of memory**
   - Solution: Enable offloading in ModelLoader node
   - Or reduce resolution/frames

4. **Import errors from wan module**
   - Solution: Install dependencies: `pip install -r requirements.txt`

## Testing Integration

### Manual Test

```python
# Test bridge module
from storymem_wrapper.storymem_bridge import StoryMemPipeline, is_storymem_available

assert is_storymem_available(), "StoryMem not available"

pipeline = StoryMemPipeline(
    t2v_checkpoint_dir="./models/Wan2.2-T2V-A14B",
    i2v_checkpoint_dir="./models/Wan2.2-I2V-A14B",
    m2v_lora_path="./models/StoryMem/Wan2.2-MI2V-A14B",
)

video = pipeline.generate_t2v(
    prompt="A cat playing piano",
    num_frames=25,
    height=480,
    width=832,
)

print(f"Generated video shape: {video.shape}")
```

### Full Workflow Test

1. Open ComfyUI
2. Add nodes:
   - StoryMemModelLoader
   - StoryMemMemoryBuffer
   - StoryMemFirstShot
   - VideoPreview
3. Connect nodes
4. Set prompt
5. Queue prompt
6. Check console for generation progress

## Performance Considerations

### Model Loading Time

- **First load**: 2-5 minutes (loading from disk)
- **Subsequent loads**: Faster with caching
- **With offloading**: Slower generation but less VRAM

### Generation Speed

- **T2V (25 frames, 832x480)**: ~30-120 seconds
  - Depends on: GPU, steps, offloading
- **M2V (with memory)**: Similar to T2V
- **Batch processing**: Not recommended (VRAM limits)

### Optimization Tips

1. **Use flash-attn**: 2-3x faster attention
2. **Enable xformers**: Better memory efficiency
3. **Lower steps**: 30-40 steps often sufficient
4. **Smaller resolution**: 640x360 for testing
5. **Sequential shots**: Generate one at a time

## Contributing

When modifying the integration:

1. **Keep storymem_src submodule unchanged**
   - Don't edit files in storymem_src/
   - All changes in storymem_wrapper/

2. **Update bridge module** for new features
   - Add methods to StoryMemPipeline
   - Maintain API compatibility

3. **Test fallback mode**
   - Ensure nodes work without pipeline
   - Provide useful error messages

4. **Document changes**
   - Update this file
   - Add examples to README.md

## References

- [StoryMem Paper](https://arxiv.org/abs/2512.19539)
- [StoryMem GitHub](https://github.com/Kevin-thu/StoryMem)
- [Wan2.2 Models](https://huggingface.co/Wan-AI)
- [ComfyUI Documentation](https://docs.comfy.org/)
