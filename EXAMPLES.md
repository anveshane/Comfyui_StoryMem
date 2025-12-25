# StoryMem ComfyUI Examples

Example workflows and usage patterns for StoryMem nodes.

## Basic Workflow: Single Shot

Generate a single video shot from a text prompt.

### Nodes Setup:

```
[StoryMemModelLoader]
    ↓ models
[StoryMemMemoryBuffer] (max_size=10)
    ↓ memory
[StoryMemFirstShot]
    - prompt: "A cinematic establishing shot of a futuristic city at sunset"
    - num_frames: 25
    - resolution: 832x480
    - seed: 42
    ↓ video
[VideoPreview]
```

### Result:
- 25-frame video (1 second at 25fps)
- Memory buffer initialized with 3 keyframes
- Ready for continuation shots

---

## Multi-Shot Story: Three Acts

Create a short story with three connected shots.

### Act 1: Establishing Shot

```
[StoryMemFirstShot]
    prompt: "Wide shot of a lone astronaut walking across a red Martian desert, dramatic lighting"
    use_t2v: True
    ↓ video1, memory, last_frame
```

### Act 2: Character Close-up

```
[StoryMemContinuationShot]
    (memory from Act 1)
    prompt: "Close-up of astronaut's helmet visor, Earth reflected in the glass"
    mode: MI2V
    first_frame: (from Act 1 last_frame)
    is_scene_cut: False
    ↓ video2, memory, last_frame
```

### Act 3: Discovery

```
[StoryMemContinuationShot]
    (memory from Act 2)
    prompt: "Astronaut discovers ancient alien structure half-buried in sand"
    mode: M2V
    is_scene_cut: False
    ↓ video3, memory, last_frame
```

### Combine

```
[StoryMemVideoCombine]
    video1, video2, video3
    ↓ final_video (75 frames total)
[VideoPreview]
```

---

## Advanced: Memory Visualization

Debug and understand memory state between shots.

### Workflow:

```
[StoryMemModelLoader]
    ↓ models
[StoryMemMemoryBuffer] (max_size=10, fixed_frames=3)
    ↓ memory

[StoryMemFirstShot]
    prompt: "..."
    ↓ video1, memory1, last_frame1

[StoryMemMemoryVisualizer]
    (memory1)
    ↓ visualization1
[Preview Image] ← Shows keyframes from first shot

[StoryMemContinuationShot]
    (memory1)
    prompt: "..."
    ↓ video2, memory2, last_frame2

[StoryMemMemoryVisualizer]
    (memory2)
    ↓ visualization2
[Preview Image] ← Shows accumulated keyframes
```

### What You See:
- Grid of keyframes in memory buffer
- First 3 frames (fixed) always present
- Recent frames from both shots

---

## Scene Transitions

Handle scene cuts vs. smooth transitions.

### Same Scene (Smooth Transition):

```
[StoryMemContinuationShot]
    prompt: "Camera pans to reveal hidden doorway"
    mode: MI2V
    is_scene_cut: False  ← Maintains memory
    first_frame: (previous last frame)
```

### New Scene (Cut):

```
[StoryMemContinuationShot]
    prompt: "Inside a dimly lit alien temple"
    mode: M2V
    is_scene_cut: True  ← Memory cleared/reset
```

**When to use scene cuts:**
- Location change
- Time jump
- Different characters
- New narrative beat

**When to maintain memory:**
- Same location
- Continuous action
- Following subject
- Smooth camera movement

---

## Generation Modes Comparison

### M2V (Memory-Only)

**Use when:**
- Scene cuts
- Want maximum variety
- Memory provides enough context

```python
mode: M2V
is_scene_cut: True or False
# No additional conditioning
```

**Pros**: Most flexible, works in all situations
**Cons**: Least continuity control

### MI2V (Memory + First Frame)

**Use when:**
- Smooth transitions
- Need visual continuity
- Following subject

```python
mode: MI2V
first_frame: (from previous shot last_frame)
is_scene_cut: False
```

**Pros**: Better continuity, smooth transitions
**Cons**: Requires first frame input

### MM2V (Memory + Motion)

**Use when:**
- Camera movements
- Action sequences
- Maximum smoothness needed

```python
mode: MM2V
motion_frames: (last 5 frames from previous shot)
is_scene_cut: False
```

**Pros**: Smoothest transitions, motion preserved
**Cons**: Requires MM2V LoRA, more VRAM

---

## Practical Tips

### Prompting Best Practices

**Good Prompts:**
```
"Wide establishing shot of a medieval castle on a hilltop, golden hour lighting, cinematic"

"Medium shot of a detective examining clues in a dimly lit office, film noir style"

"Close-up of hands typing on a vintage typewriter, shallow depth of field"
```

**Include:**
- Shot type (wide/medium/close-up)
- Subject/action
- Lighting
- Style/mood

**Avoid:**
- Multiple unrelated elements
- Contradictory descriptions
- Overly complex scenes

### Memory Management

**Small Stories** (2-3 shots):
```
max_memory_size: 6-8
fixed_frames: 2-3
```

**Medium Stories** (4-6 shots):
```
max_memory_size: 10-12
fixed_frames: 3-4
```

**Long Stories** (7+ shots):
```
max_memory_size: 15-20
fixed_frames: 4-5
```

### Performance Optimization

**For Testing:**
```
resolution: 640x360 or 512x288
num_frames: 13 or 17
steps: 30
```

**For Quality:**
```
resolution: 832x480 or 1280x720
num_frames: 25 or 29
steps: 50
```

**For Speed:**
- Enable offloading: Yes
- Use flash-attn: Yes
- Lower steps: 30-40
- Smaller resolution

---

## Example Prompts by Genre

### Sci-Fi

```
Shot 1: "Vast space station orbiting a gas giant, ships docking, sci-fi"
Shot 2: "Inside sterile corridor, fluorescent lights, lone figure walking"
Shot 3: "Control room with holographic displays, crew monitoring systems"
```

### Fantasy

```
Shot 1: "Enchanted forest with glowing mushrooms, misty atmosphere"
Shot 2: "Mystical creature emerging from shadows, ethereal lighting"
Shot 3: "Ancient rune-covered stone altar, magical energy swirling"
```

### Drama

```
Shot 1: "Empty cafe at dawn, single person sitting alone, melancholic"
Shot 2: "Close-up of hands holding old photograph, nostalgic mood"
Shot 3: "Person walking away down rainy street, cinematic framing"
```

### Action

```
Shot 1: "Urban rooftop chase scene, dynamic camera angle"
Shot 2: "Character leaping between buildings, dramatic slow motion"
Shot 3: "Landing and rolling, debris flying, intense action"
```

---

## Workflow Templates

### Template 1: Quick Test

Minimal setup for testing:

```
ModelLoader → MemoryBuffer → FirstShot → Preview
```

### Template 2: Three-Act Story

Standard narrative structure:

```
ModelLoader → MemoryBuffer
    ↓
FirstShot (Setup) → ContinuationShot (Conflict) → ContinuationShot (Resolution)
    ↓                    ↓                              ↓
    ├────────────────────┴──────────────────────────────┘
                         ↓
                  VideoCombine → Preview
```

### Template 3: Experimental

Testing different modes:

```
ModelLoader → MemoryBuffer
    ↓
FirstShot
    ├─→ ContinuationShot (M2V mode)  → Preview 1
    ├─→ ContinuationShot (MI2V mode) → Preview 2
    └─→ ContinuationShot (MM2V mode) → Preview 3
```

---

## Troubleshooting Workflows

### Memory Not Carrying Forward

**Problem**: Each shot looks completely different

**Solution**:
- Check memory buffer is connected
- Verify `is_scene_cut=False`
- Increase `max_memory_size`
- Use MI2V mode with first_frame

### Low Quality Output

**Problem**: Video looks blurry or artifacts

**Solution**:
- Increase `steps` (try 50-70)
- Increase resolution
- Check model paths are correct
- Try different seed values

### Out of Memory

**Problem**: CUDA out of memory error

**Solution**:
- Enable offloading in ModelLoader
- Reduce resolution (640x360)
- Reduce num_frames (17 instead of 25)
- Close other applications
- Generate shots one at a time

### Slow Generation

**Problem**: Each shot takes 5+ minutes

**Solution**:
- Install flash-attn
- Install xformers
- Reduce steps (30-40)
- Enable model offloading
- Check GPU utilization

---

## Next Steps

After mastering these workflows:

1. **Experiment with seeds**: Find consistent styles
2. **Try different resolutions**: Balance quality/speed
3. **Build shot libraries**: Save successful prompts
4. **Create longer stories**: 10+ shots with memory
5. **Mix modes strategically**: Use right mode for each shot

Happy storytelling! 🎬
