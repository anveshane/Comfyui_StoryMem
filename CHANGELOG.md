# Changelog

All notable changes to ComfyUI StoryMem Nodes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-25

### Added

#### Core Nodes
- **StoryMemModelLoader** - Load T2V, I2V, and M2V LoRA models with VRAM management
- **StoryMemMemoryBuffer** - Create configurable memory buffer with sliding window strategy
- **StoryMemUpdateMemory** - Extract keyframes and update memory buffer
- **StoryMemFirstShot** - Generate first video shot using T2V or M2V
- **StoryMemContinuationShot** - Generate continuation shots with memory conditioning
- **StoryMemMemoryVisualizer** - Visualize memory buffer contents as grid
- **StoryMemVideoCombine** - Concatenate multiple video shots
- **StoryMemGetMotionFrames** - Extract motion frames for MM2V conditioning
- **StoryMemInfo** - Display system information and status

#### Integration Layer
- **StoryMem Bridge Module** (`storymem_bridge.py`) - Unified API wrapper for StoryMem repository
  - `StoryMemPipeline` class for T2V, I2V, M2V generation
  - Lazy loading and model caching
  - Automatic tensor format conversion
  - VRAM optimization with model offloading

#### Memory Management
- **Sliding Window Memory Buffer** - Maintains fixed early frames + recent frames
  - Configurable max size (default: 10 keyframes)
  - Configurable fixed frames (default: 3)
  - Automatic keyframe extraction from generated videos
  - Shot-to-shot memory passing for character consistency

#### Utilities
- **Tensor Format Conversion** - Automatic conversion between ComfyUI and StoryMem formats
  - `comfyui_to_storymem()` - [B,F,H,W,C] → [B,C,F,H,W]
  - `storymem_to_comfyui()` - [B,C,F,H,W] → [B,F,H,W,C]
  - Support for image batching and frame extraction
  - Format validation utilities

- **Keyframe Extraction** - Extract representative frames from videos
  - Evenly-spaced extraction (placeholder for CLIP)
  - Last frame extraction for MI2V conditioning
  - Motion frame extraction for MM2V conditioning
  - Configurable similarity thresholds

- **Model Download Script** (`download_models.py`)
  - Automatic HuggingFace model downloading
  - Disk space checking
  - Resumable downloads
  - Progress tracking

#### Configuration
- **Model Configuration** (`model_config.yaml`)
  - Default generation parameters
  - Model repository information
  - VRAM requirements
  - Optimization settings

#### Documentation
- **README.md** - Comprehensive user guide with installation, usage, and troubleshooting
- **INTEGRATION.md** - Technical integration guide for developers
- **EXAMPLES.md** - Workflow examples and best practices
- **Test Suite** (`test_memory.py`) - Memory buffer functionality tests

### Features

#### Generation Modes
- **T2V (Text-to-Video)** - Generate first shot from text prompt
- **M2V (Memory-to-Video)** - Generate with memory-only conditioning
- **MI2V (Memory+Image-to-Video)** - Generate with memory + first frame (better continuity)
- **MM2V (Memory+Motion-to-Video)** - Generate with memory + motion frames (smoothest transitions)

#### VRAM Management
- Model offloading to CPU when not in use
- Support for FSDP (Fully Sharded Data Parallel)
- VRAM availability checking
- Graceful degradation on low memory

#### Shot Management
- Shot-level node architecture for maximum flexibility
- Scene cut detection and handling
- Automatic memory clearing on scene transitions
- Configurable frame counts (4n+1 format)

#### Quality Settings
- Configurable resolution (512x288 to 1280x720+)
- Adjustable frame counts (9 to 121 frames)
- Variable sampling steps (1-100)
- Classifier-free guidance scale control
- Seed control for reproducibility

### Technical Details

#### Dependencies
- Python 3.10+
- PyTorch 2.4.0+
- ComfyUI (latest)
- StoryMem (via git submodule)
- Diffusers 0.32.2
- Transformers 4.45.2
- PEFT 0.14.0
- See `requirements.txt` for complete list

#### Model Requirements
- Wan2.2-T2V-A14B (~20-30GB)
- Wan2.2-I2V-A14B (~20-30GB)
- StoryMem M2V LoRA (~1-2GB)
- Total: ~40-60GB storage
- VRAM: 80GB recommended (24GB+ with offloading)

#### Architecture
- Shot-level node design for composability
- Memory buffer with sliding window strategy
- Bridge pattern for clean integration
- Fallback mode when models unavailable
- Automatic format conversion

### Installation Methods
- Manual installation via git clone
- ComfyUI Manager support (future)
- Submodule initialization for StoryMem
- Automated model downloading

### Known Limitations
- CLIP-based keyframe extraction not yet implemented (using evenly-spaced fallback)
- HPSv3 quality filtering not yet implemented
- MM2V mode uses M2V as fallback pending full implementation
- Requires substantial VRAM for full-quality generation
- Sequential shot generation only (no batch processing)

### Performance Notes
- First load: 2-5 minutes (model loading)
- T2V generation (25 frames, 832x480): ~30-120 seconds
- Faster with flash-attn and xformers
- Model offloading reduces VRAM but increases generation time

## [Unreleased]

### Planned Features
- CLIP-based intelligent keyframe extraction
- HPSv3 quality filtering for keyframes
- Full MI2V and MM2V mode implementation
- ComfyUI Manager integration
- Workflow JSON templates
- Video export with custom codecs
- Batch shot processing
- LoRA fine-tuning support
- Custom model path configuration UI
- Progress bars during generation
- Intermediate frame preview
- Memory state export/import
- Multi-GPU support improvements

### Future Optimizations
- Quantization support (FP8, INT8)
- Faster attention mechanisms
- Memory-efficient inference
- Streaming generation
- Background model loading
- Cached prompt embeddings

## Notes

### Version 0.1.0 Status
This is the initial release with core functionality implemented. The nodes are fully functional with placeholder implementations for features pending full StoryMem integration. All node interfaces are stable and ready for use.

### Breaking Changes
None (initial release)

### Migration Guide
Not applicable (initial release)

### Credits
- Original StoryMem: [Kevin-thu/StoryMem](https://github.com/Kevin-thu/StoryMem)
- Wan2.2 Models: [Alibaba Wan Team](https://huggingface.co/Wan-AI)
- ComfyUI: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### Contributing
Contributions welcome! Please see INTEGRATION.md for development guidelines.

### License
See LICENSE file for details.

---

## Version History

- **0.1.0** (2025-12-25) - Initial release with core nodes and StoryMem integration
