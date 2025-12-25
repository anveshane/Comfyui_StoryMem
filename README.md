# ComfyUI StoryMem Nodes

Custom ComfyUI nodes for [StoryMem](https://github.com/Kevin-thu/StoryMem) - Multi-shot long video storytelling with memory-conditioned video diffusion models.

## Overview

StoryMem is an AI system that generates extended narrative videos from text prompts. These ComfyUI nodes allow you to create minute-long, multi-shot videos with consistent characters and cinematic quality directly in ComfyUI workflows.

### Features

- **Shot-level control**: Generate videos shot-by-shot with full control over each segment
- **Memory-conditioned generation**: Maintain character and scene consistency across shots
- **Three generation modes**: M2V (memory-only), MI2V (memory+first-frame), MM2V (memory+motion)
- **Flexible workflows**: Chain nodes together to create complex narratives
- **ComfyUI integration**: Native tensor format support and model management

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 80GB VRAM (for full models) or 24GB+ VRAM (with optimizations)
- **Storage**: ~40-60GB for model files
- **RAM**: 32GB+ recommended

### Software
- ComfyUI (latest version)
- Python 3.10 or 3.11
- CUDA 11.8 or higher
- PyTorch 2.4.0 or higher

## Installation

### Method 1: ComfyUI Manager (Recommended - Coming Soon)

1. Open ComfyUI Manager
2. Search for "StoryMem"
3. Click Install
4. Restart ComfyUI
5. Download models (see below)

### Method 2: Manual Installation

```bash
# Navigate to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes

# Clone this repository
git clone https://github.com/[your-username]/comfyui_storymem.git
cd comfyui_storymem

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash attention for better performance
pip install flash-attn --no-build-isolation

# Optional: Install xformers for memory efficiency
pip install xformers
```

### Downloading Models

The StoryMem models are large (~40-60GB total). You can download them using the provided script:

```bash
python scripts/download_models.py
```

Or manually using HuggingFace CLI:

```bash
# Create models directory in ComfyUI
mkdir -p ../../models/storymem

# Download T2V model (~20-30GB)
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ../../models/storymem/Wan2.2-T2V-A14B

# Download I2V model (~20-30GB)
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ../../models/storymem/Wan2.2-I2V-A14B

# Download M2V LoRA weights
huggingface-cli download Kevin-thu/StoryMem --local-dir ../../models/storymem/StoryMem
```

**Model Storage Location**: Models should be placed in `ComfyUI/models/storymem/`

## Available Nodes

### StoryMem Model Loader
Loads and caches the three StoryMem models (T2V, I2V, M2V LoRA).

**Inputs**:
- `t2v_model`: Text-to-Video model selection
- `i2v_model`: Image-to-Video model selection
- `m2v_lora`: LoRA type (MI2V or MM2V)
- `device`: Device selection (auto/cuda/cpu)
- `enable_offload`: Enable CPU offloading for VRAM management

**Outputs**: `STORYMEM_MODELS`

### StoryMem Memory Buffer
Creates and configures a memory buffer for shot-to-shot continuity.

**Inputs**:
- `max_memory_size`: Maximum keyframes to retain (default: 10)
- `fixed_frames`: Number of earliest frames to always keep (default: 3)

**Outputs**: `STORYMEM_MEMORY`

### StoryMem First Shot
Generates the first shot of a story using T2V model.

**Inputs**:
- `models`: STORYMEM_MODELS from loader
- `prompt`: Text description (multiline)
- `num_frames`: Frame count (default: 25, must be 4n+1)
- `width`, `height`: Resolution (default: 832x480)
- `seed`, `steps`, `cfg_scale`: Generation parameters
- `use_t2v`: Use T2V vs M2V for first shot

**Outputs**: `VIDEO`, `STORYMEM_MEMORY`, `IMAGE` (last frame)

### StoryMem Continuation Shot
Generates subsequent shots with memory conditioning.

**Inputs**:
- `models`: STORYMEM_MODELS
- `memory`: STORYMEM_MEMORY from previous shot
- `prompt`: Text description
- `generation_mode`: M2V / MI2V / MM2V
- `num_frames`, `seed`, `steps`, `cfg_scale`: Generation parameters
- `is_scene_cut`: Reset memory for scene transitions

**Outputs**: `VIDEO`, `STORYMEM_MEMORY` (updated), `IMAGE` (last frame)

**Generation Modes**:
- **M2V**: Memory-only conditioning (most flexible)
- **MI2V**: Memory + first frame (better continuity)
- **MM2V**: Memory + 5 motion frames (smoothest transitions)

### StoryMem Memory Visualizer
Visualizes the current state of the memory buffer.

**Inputs**: `STORYMEM_MEMORY`
**Outputs**: `IMAGE` (grid of keyframes)

### StoryMem Video Combine
Concatenates multiple video shots into a single video.

**Inputs**: Multiple `VIDEO` inputs
**Outputs**: Single `VIDEO`

## Basic Workflow Example

```
[StoryMemModelLoader]
    ↓ models
[StoryMemMemoryBuffer] (max_size=10)
    ↓ memory
[StoryMemFirstShot]
    prompt: "A young woman walks through a bustling city street at sunset"
    ↓ video, memory, last_frame
[StoryMemContinuationShot]
    prompt: "She stops at a café window, looking at her reflection"
    mode: MI2V
    ↓ video, memory, last_frame
[StoryMemContinuationShot]
    prompt: "Inside the café, warm lighting illuminates the cozy interior"
    mode: M2V
    is_scene_cut: True
    ↓ video, memory, last_frame
[StoryMemVideoCombine]
    ↓ final_video
[VideoPreview]
```

## Memory Management

### Memory Buffer Strategy

StoryMem uses a sliding window approach:
- **max_size**: Total keyframes to retain (e.g., 10)
- **fixed_frames**: Earliest frames to always keep (e.g., 3)
- **Strategy**: Keep `fixed_frames` + most recent frames to fill remaining slots

This ensures:
- Long-term consistency (fixed frames preserve initial character appearance)
- Recent context (recent frames ensure smooth transitions)
- Bounded memory usage (never exceeds max_size)

### VRAM Management

The full StoryMem models require ~80GB VRAM. To manage this:

1. **Enable model offloading**: Set `enable_offload=True` in ModelLoader
2. **Use quantization**: Consider quantized model versions (lower quality but much less VRAM)
3. **Clear cache**: ComfyUI automatically manages cache, but you can force-clear if needed
4. **Sequential generation**: Generate one shot at a time rather than batching

## Tips for Best Results

1. **Prompt writing**: Be specific and descriptive. Include details about characters, setting, lighting, and camera angles.

2. **Generation modes**:
   - Use **MI2V** for smooth continuity within a scene
   - Use **M2V** for more variety or when scene cut occurs
   - Use **MM2V** for the smoothest transitions (requires more VRAM)

3. **Scene cuts**: Set `is_scene_cut=True` when transitioning to a completely new scene or location.

4. **Memory size**: Larger memory buffers (10-15) maintain consistency better but use more VRAM.

5. **Resolution**: Start with 832x480, increase to 720p if you have sufficient VRAM.

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory during model loading or generation.

**Solutions**:
- Enable `enable_offload=True` in ModelLoader
- Reduce `num_frames` (e.g., from 25 to 17 or 13)
- Lower resolution (e.g., 640x360)
- Close other VRAM-intensive applications
- Consider quantized model versions

### Slow Generation

**Problem**: Generation takes a very long time.

**Solutions**:
- Install `flash-attn` and `xformers` for acceleration
- Reduce `steps` (try 30-40 instead of 50)
- Use M2V mode instead of MM2V (less conditioning data)
- Ensure models are on GPU, not CPU

### Character Consistency Issues

**Problem**: Characters look different between shots.

**Solutions**:
- Increase `max_memory_size` (try 15-20)
- Use MI2V mode for better continuity
- Ensure `is_scene_cut=False` for same-scene transitions
- Use more detailed character descriptions in prompts

### Model Download Fails

**Problem**: HuggingFace download interrupted or fails.

**Solutions**:
- Downloads are resumable - just run the command again
- Check disk space (~60GB needed)
- Check internet connection
- Try manual download from HuggingFace website

## Technical Details

### Tensor Formats

- **ComfyUI format**: `[B, F, H, W, C]` (batch, frames, height, width, channels)
- **StoryMem format**: `[B, C, F, H, W]` (channels first)
- Conversion is handled automatically by the nodes

### Model Architecture

StoryMem uses:
- **Wan2.2-T2V-A14B**: 14B parameter text-to-video model with MoE architecture
- **Wan2.2-I2V-A14B**: 14B parameter image-to-video model
- **M2V LoRA**: Fine-tuned adaptation weights for memory conditioning

### Keyframe Extraction

Keyframes are extracted using:
- CLIP embeddings for similarity detection
- HPSv3 for quality filtering
- Configurable similarity threshold (default: 0.9)
- Maximum keyframes per shot (default: 3)

## Credits and License

This project wraps the original [StoryMem](https://github.com/Kevin-thu/StoryMem) repository.

**Original StoryMem**:
- Paper: [StoryMem: Multi-Shot Long Video Storytelling with Memory](https://arxiv.org/abs/2512.19539)
- Authors: Kevin Thu et al.
- License: Check the original repository for license details

**ComfyUI Nodes**:
- Repository: [Add your repo URL here]
- License: [Add your license here]

## Support and Contributing

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join discussions on ComfyUI forums
- **Contributing**: Pull requests are welcome!

## Changelog

### v0.1.0 (Initial Release)
- Basic node implementations
- Model loader with offloading support
- Shot-level generation (First Shot, Continuation Shot)
- Memory buffer management
- Three generation modes (M2V, MI2V, MM2V)
- Video combination utilities

## References

- [StoryMem Paper](https://arxiv.org/abs/2512.19539)
- [StoryMem GitHub](https://github.com/Kevin-thu/StoryMem)
- [Wan2.2 Models on HuggingFace](https://huggingface.co/Wan-AI)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
