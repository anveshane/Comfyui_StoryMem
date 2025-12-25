"""
Shot generation nodes for StoryMem.

Core nodes for generating video shots:
- StoryMemFirstShot: Generate first shot with T2V
- StoryMemContinuationShot: Generate subsequent shots with memory conditioning
"""

from ..storymem_wrapper.shot_generator import create_shot_generator
from ..storymem_wrapper.keyframe_extractor import extract_keyframes_simple
from ..storymem_wrapper.tensor_utils import get_last_frame, get_motion_frames


class StoryMemFirstShot:
    """
    Generate the first shot of a story using T2V model.

    Creates the initial video and memory state for subsequent shots.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Valid frame counts (must be 4n+1)
        valid_frames = [9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121]

        return {
            "required": {
                "models": ("STORYMEM_MODELS",),
                "memory": ("STORYMEM_MEMORY",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of...",
                }),
                "num_frames": (valid_frames, {
                    "default": 25,
                }),
                "width": ("INT", {
                    "default": 832,
                    "min": 512,
                    "max": 1280,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 480,
                    "min": 320,
                    "max": 768,
                    "step": 64,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
                "use_t2v": ("BOOLEAN", {
                    "default": True,
                    "label_on": "T2V",
                    "label_off": "M2V (no memory)",
                }),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "keyframes_to_extract": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STORYMEM_MEMORY", "IMAGE")
    RETURN_NAMES = ("video", "memory", "last_frame")
    FUNCTION = "generate"
    CATEGORY = "StoryMem/generation"
    DESCRIPTION = "Generate the first shot of a story using T2V model. Creates initial memory state for subsequent shots."

    def generate(
        self,
        models,
        memory,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg_scale: float,
        use_t2v: bool,
        init_image=None,
        keyframes_to_extract: int = 3,
    ):
        """
        Generate first shot.

        Args:
            models: Loaded StoryMem models
            memory: Memory buffer (will be updated with keyframes)
            prompt: Text description of the shot
            num_frames: Number of frames (must be 4n+1)
            width: Video width
            height: Video height
            seed: Random seed
            steps: Diffusion steps
            cfg_scale: CFG scale
            use_t2v: Use T2V vs M2V
            init_image: Optional initial image
            keyframes_to_extract: Number of keyframes to extract

        Returns:
            (video, updated_memory, last_frame)
        """
        print(f"\n{'='*60}")
        print(f"StoryMem First Shot Generation")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Mode: {'T2V' if use_t2v else 'M2V (no memory)'}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}")

        # Create shot generator
        generator = create_shot_generator(models)

        # Generate video
        video = generator.generate_first_shot(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            use_t2v=use_t2v,
            init_image=init_image,
        )

        # Extract keyframes and update memory
        keyframes, indices = extract_keyframes_simple(video, keyframes_to_extract)
        memory.add_keyframes(keyframes, shot_id=1)

        print(f"✓ Generated {num_frames} frames")
        print(f"✓ Extracted {len(keyframes)} keyframes at indices {indices}")
        print(f"✓ Memory updated: {len(memory)} total frames")

        # Get last frame for next shot conditioning
        last_frame = get_last_frame(video)

        # Return video as IMAGE (batched frames)
        return (video, memory, last_frame.unsqueeze(0))


class StoryMemContinuationShot:
    """
    Generate a continuation shot with memory conditioning.

    Uses memory from previous shots to maintain consistency.
    Supports three modes: M2V (memory-only), MI2V (memory+image), MM2V (memory+motion).
    """

    @classmethod
    def INPUT_TYPES(cls):
        valid_frames = [9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121]

        return {
            "required": {
                "models": ("STORYMEM_MODELS",),
                "memory": ("STORYMEM_MEMORY",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The scene continues with...",
                }),
                "generation_mode": (["M2V", "MI2V", "MM2V"], {
                    "default": "MI2V",
                }),
                "num_frames": (valid_frames, {
                    "default": 25,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
                "is_scene_cut": ("BOOLEAN", {
                    "default": False,
                    "label_on": "yes (new scene)",
                    "label_off": "no (same scene)",
                }),
            },
            "optional": {
                "first_frame": ("IMAGE",),  # For MI2V mode
                "motion_frames": ("IMAGE",),  # For MM2V mode (5 frames)
                "width": ("INT", {"default": 832}),
                "height": ("INT", {"default": 480}),
                "keyframes_to_extract": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STORYMEM_MEMORY", "IMAGE")
    RETURN_NAMES = ("video", "memory", "last_frame")
    FUNCTION = "generate"
    CATEGORY = "StoryMem/generation"
    DESCRIPTION = "Generate continuation shot with memory conditioning. Supports M2V (memory), MI2V (memory+first frame), MM2V (memory+motion) modes."

    def generate(
        self,
        models,
        memory,
        prompt: str,
        generation_mode: str,
        num_frames: int,
        seed: int,
        steps: int,
        cfg_scale: float,
        is_scene_cut: bool,
        first_frame=None,
        motion_frames=None,
        width: int = 832,
        height: int = 480,
        keyframes_to_extract: int = 3,
    ):
        """
        Generate continuation shot.

        Args:
            models: Loaded StoryMem models
            memory: Memory buffer from previous shots
            prompt: Text description
            generation_mode: M2V / MI2V / MM2V
            num_frames: Number of frames
            seed: Random seed
            steps: Diffusion steps
            cfg_scale: CFG scale
            is_scene_cut: Whether this is a scene cut
            first_frame: First frame for MI2V
            motion_frames: Motion frames for MM2V
            width: Video width
            height: Video height
            keyframes_to_extract: Keyframes to extract

        Returns:
            (video, updated_memory, last_frame)
        """
        print(f"\n{'='*60}")
        print(f"StoryMem Continuation Shot")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Mode: {generation_mode}, Scene cut: {is_scene_cut}")
        print(f"Memory: {len(memory)} keyframes")
        print(f"Frames: {num_frames}, Seed: {seed}")

        # Create shot generator
        generator = create_shot_generator(models)

        # Extract conditioning frames if needed
        first_frame_tensor = None
        motion_frames_list = None

        if generation_mode == "MI2V" and first_frame is not None:
            # first_frame is [1, H, W, C], extract to [H, W, C]
            first_frame_tensor = first_frame[0] if first_frame.shape[0] == 1 else first_frame
            print(f"  Using first frame conditioning")

        if generation_mode == "MM2V" and motion_frames is not None:
            # motion_frames should be [5, H, W, C] or [1, 5, H, W, C]
            if motion_frames.ndim == 5:
                motion_frames = motion_frames[0]  # Remove batch dim
            # Convert to list of [H, W, C]
            motion_frames_list = [motion_frames[i] for i in range(min(5, motion_frames.shape[0]))]
            print(f"  Using {len(motion_frames_list)} motion frames")

        # Generate video
        video = generator.generate_continuation(
            prompt=prompt,
            memory=memory,
            generation_mode=generation_mode,
            num_frames=num_frames,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            is_scene_cut=is_scene_cut,
            first_frame=first_frame_tensor,
            motion_frames=motion_frames_list,
            width=width,
            height=height,
        )

        # Extract keyframes and update memory
        keyframes, indices = extract_keyframes_simple(video, keyframes_to_extract)

        # Determine shot ID (increment from current max)
        shot_ids = memory.metadata.get('shot_ids', [])
        shot_id = max(shot_ids) + 1 if shot_ids else 2  # First shot is 1, this is 2+

        memory.add_keyframes(keyframes, shot_id=shot_id)

        print(f"✓ Generated {num_frames} frames")
        print(f"✓ Extracted {len(keyframes)} keyframes at indices {indices}")
        print(f"✓ Memory updated: {len(memory)} total frames")

        # Get last frame for next shot
        last_frame = get_last_frame(video)

        return (video, memory, last_frame.unsqueeze(0))
