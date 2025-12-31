# ComfyUI StoryMem Test Workflows

This directory contains test workflows for verifying the StoryMem custom nodes work correctly after fixing issue #3 (import errors).

## Available Workflows

### 1. `test_workflow_simple.json` - Single Shot Test

**Purpose**: Basic functionality test
**Complexity**: Simple (1 shot)
**Runtime**: ~2-5 minutes

**What it tests**:
- Model loading
- Memory buffer creation
- First shot generation (T2V)
- Memory visualization

**Nodes used**:
- StoryMemModelLoader
- StoryMemMemoryBuffer
- StoryMemFirstShot
- StoryMemMemoryVisualizer
- PreviewImage (x2)

### 2. `test_workflow_3shot_story.json` - Multi-Shot Story

**Purpose**: Complete workflow test with memory continuity
**Complexity**: Advanced (3 shots)
**Runtime**: ~10-15 minutes

**What it tests**:
- Model loading
- Memory buffer creation
- First shot generation (T2V)
- Continuation shot with MI2V (memory + first frame)
- Continuation shot with M2V and scene cut
- Video combination
- Memory persistence across shots

**Story outline**:
1. Shot 1: "A young woman walks through a bustling city street at sunset"
2. Shot 2: "She stops at a café window, looking at her reflection" (MI2V mode)
3. Shot 3: "Inside the café, warm lighting illuminates the cozy interior" (M2V mode, scene cut)

**Nodes used**:
- StoryMemModelLoader
- StoryMemMemoryBuffer
- StoryMemFirstShot
- StoryMemContinuationShot (x2)
- StoryMemVideoCombine
- PreviewImage (x3)

## How to Use

### Prerequisites

1. **Install ComfyUI** (if not already installed)
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. **Install StoryMem Custom Nodes**
   ```bash
   cd custom_nodes
   git clone https://github.com/anveshane/Comfyui_StoryMem
   cd Comfyui_StoryMem
   git submodule update --init --recursive
   pip install -r requirements.txt
   ```

3. **Download Models**

   You need to download the StoryMem models:
   - Wan2.2-T2V-A14B (Text-to-Video) - ~20GB
   - Wan2.2-I2V-A14B (Image-to-Video) - ~20GB
   - StoryMem LoRA weights (MI2V/MM2V) - ~2GB

   Place models in: `ComfyUI/models/checkpoints/storymem/`

### Loading a Workflow

1. Start ComfyUI:
   ```bash
   python main.py
   ```

2. Open your browser to `http://127.0.0.1:8188`

3. Load a workflow:
   - Click "Load" button
   - Navigate to `custom_nodes/Comfyui_StoryMem/workflows/`
   - Select `test_workflow_simple.json` or `test_workflow_3shot_story.json`

4. Click "Queue Prompt" to run

### Expected Results

#### Simple Workflow
- **Output**: Single 25-frame video (832x480)
- **Memory**: 3 keyframes extracted
- **Preview**: Video frames + memory visualization grid

#### 3-Shot Story Workflow
- **Output**: Combined video with 75 frames total (3 shots × 25 frames)
- **Memory**: Up to 10 keyframes maintained across shots
- **Preview**: Individual shot previews + final combined video

## Troubleshooting

### Import Errors

If you see errors like:
```
NameError: name 'Dict' is not defined
```
or
```
cannot import name 'WanMI2V' from 'wan.textimage2video'
```

**Solution**: Make sure you're using the fixed version from PR that addresses issue #3.

### Model Not Found

```
FileNotFoundError: Model checkpoint not found
```

**Solution**:
- Verify models are downloaded and placed in correct directory
- Check model paths in StoryMemModelLoader node
- Use `StoryMemInfo` node to see expected model locations

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
- Enable "offload_model" in StoryMemModelLoader
- Reduce frame count from 25 to 17 or 13
- Reduce resolution from 832x480 to 704x384
- Close other GPU applications

### Slow Generation

**Solutions**:
- Reduce sampling steps (from 50 to 30)
- Enable model offloading (trades VRAM for speed)
- Use smaller frame counts
- Consider using a machine with more powerful GPU

## Testing the Fix for Issue #3

These workflows specifically test the fixes made in issue #3:

### Test 1: Import Verification
The workflows will fail to load if the imports are broken. Successfully loading the workflow confirms:
- ✅ `Dict` and `Any` are properly imported in `keyframe_extractor.py`
- ✅ `WanTI2V` (not `WanMI2V`) is correctly imported in `storymem_bridge.py`

### Test 2: Type Annotations
The `analyze_frame_diversity` method in `keyframe_extractor.py` uses `Dict[str, Any]`. If this executes without `NameError`, the fix is working.

### Test 3: MI2V Pipeline
The 3-shot workflow uses MI2V mode which requires the `WanTI2V` class. Successfully running shot 2 confirms the class name fix.

## Contributing

Found issues with these workflows? Please report them:
- GitHub: https://github.com/anveshane/Comfyui_StoryMem/issues
- Include: ComfyUI version, GPU model, error messages, workflow used

## Version History

- **v1.0.0** (2024-01-XX): Initial test workflows for issue #3 fix
  - Added simple single-shot test
  - Added 3-shot story test with memory continuity
  - Verified import fixes work correctly
