"""
Tensor format conversion utilities for ComfyUI and StoryMem.

ComfyUI uses [B, F, H, W, C] format (batch, frames, height, width, channels)
StoryMem uses [B, C, F, H, W] format (channels first)
"""

import torch
import numpy as np
from typing import Union, Tuple


def comfyui_to_storymem(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI video format to StoryMem format.

    Args:
        video_tensor: Tensor in ComfyUI format [B, F, H, W, C], values in [0, 1]

    Returns:
        storymem_tensor: Tensor in StoryMem format [B, C, F, H, W], normalized for models

    Example:
        >>> comfyui_video = torch.rand(1, 25, 480, 832, 3)  # [B, F, H, W, C]
        >>> storymem_video = comfyui_to_storymem(comfyui_video)
        >>> storymem_video.shape
        torch.Size([1, 3, 25, 480, 832])  # [B, C, F, H, W]
    """
    # Validate input shape
    if video_tensor.ndim != 5:
        raise ValueError(
            f"Expected 5D tensor [B, F, H, W, C], got {video_tensor.ndim}D tensor with shape {video_tensor.shape}"
        )

    # Permute dimensions: [B, F, H, W, C] -> [B, C, F, H, W]
    tensor = video_tensor.permute(0, 4, 1, 2, 3)

    # Normalize from [0, 1] to [-1, 1] range (standard for diffusion models)
    tensor = tensor * 2.0 - 1.0
    tensor = torch.clamp(tensor, -1.0, 1.0)

    return tensor


def storymem_to_comfyui(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert StoryMem format to ComfyUI format.

    Args:
        video_tensor: Tensor in StoryMem format [B, C, F, H, W], typically in [-1, 1]

    Returns:
        comfyui_tensor: Tensor in ComfyUI format [B, F, H, W, C], values in [0, 1]

    Example:
        >>> storymem_video = torch.randn(1, 3, 25, 480, 832)  # [B, C, F, H, W]
        >>> comfyui_video = storymem_to_comfyui(storymem_video)
        >>> comfyui_video.shape
        torch.Size([1, 25, 480, 832, 3])  # [B, F, H, W, C]
    """
    # Validate input shape
    if video_tensor.ndim != 5:
        raise ValueError(
            f"Expected 5D tensor [B, C, F, H, W], got {video_tensor.ndim}D tensor with shape {video_tensor.shape}"
        )

    # Denormalize from [-1, 1] to [0, 1]
    tensor = (video_tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Permute dimensions: [B, C, F, H, W] -> [B, F, H, W, C]
    tensor = tensor.permute(0, 2, 3, 4, 1)

    return tensor


def numpy_to_tensor(
    numpy_video: np.ndarray,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.

    Args:
        numpy_video: Numpy array, can be in various formats
        device: Target device for the tensor

    Returns:
        tensor: Torch tensor on specified device

    Note:
        Automatically detects if values are in [0, 255] or [0, 1] range
        and normalizes accordingly.
    """
    # Convert to float32
    if numpy_video.dtype != np.float32:
        numpy_video = numpy_video.astype(np.float32)

    # Normalize if needed (detect [0, 255] range)
    if numpy_video.max() > 1.0:
        numpy_video = numpy_video / 255.0

    # Convert to tensor
    tensor = torch.from_numpy(numpy_video).float().to(device)

    return tensor


def tensor_to_numpy(video_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    Args:
        video_tensor: Torch tensor

    Returns:
        numpy_video: Numpy array, values scaled to [0, 255] uint8
    """
    # Move to CPU if needed
    if video_tensor.is_cuda:
        video_tensor = video_tensor.cpu()

    # Convert to numpy
    numpy_video = video_tensor.numpy()

    # Scale to [0, 255] if needed
    if numpy_video.max() <= 1.0:
        numpy_video = (numpy_video * 255).astype(np.uint8)
    else:
        numpy_video = numpy_video.astype(np.uint8)

    return numpy_video


def image_to_tensor(
    image: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Convert image (numpy or tensor) to standardized torch tensor.

    Args:
        image: Image as numpy array or torch tensor [H, W, C] or [C, H, W]
        device: Target device

    Returns:
        tensor: Standardized tensor [H, W, C] in [0, 1] range
    """
    if isinstance(image, np.ndarray):
        image = numpy_to_tensor(image, device)

    # Ensure tensor is on correct device
    if isinstance(image, torch.Tensor):
        image = image.to(device)

    # Handle channel dimension
    if image.ndim == 3:
        # If channels first [C, H, W], convert to [H, W, C]
        if image.shape[0] == 3 or image.shape[0] == 1:
            if image.shape[0] < min(image.shape[1], image.shape[2]):
                image = image.permute(1, 2, 0)

    # Ensure [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0

    return image


def batch_images_to_video(
    images: list[torch.Tensor],
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Convert list of images to video tensor in ComfyUI format.

    Args:
        images: List of image tensors [H, W, C]
        device: Target device

    Returns:
        video_tensor: Video tensor [1, F, H, W, C] where F = len(images)
    """
    # Standardize all images
    standardized = [image_to_tensor(img, device) for img in images]

    # Stack into video [F, H, W, C]
    video = torch.stack(standardized, dim=0)

    # Add batch dimension [1, F, H, W, C]
    video = video.unsqueeze(0)

    return video


def extract_frames_from_video(video_tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Extract individual frames from video tensor.

    Args:
        video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]

    Returns:
        frames: List of frame tensors [H, W, C]
    """
    # Remove batch dimension if present
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]  # Take first batch

    # video_tensor is now [F, H, W, C]
    frames = [video_tensor[i] for i in range(video_tensor.shape[0])]

    return frames


def get_last_frame(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the last frame from a video tensor.

    Args:
        video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]

    Returns:
        last_frame: Last frame tensor [H, W, C]
    """
    # Remove batch dimension if present
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]  # Take first batch

    # Return last frame [H, W, C]
    return video_tensor[-1]


def get_motion_frames(
    video_tensor: torch.Tensor,
    num_frames: int = 5
) -> torch.Tensor:
    """
    Extract last N frames for motion conditioning (MM2V mode).

    Args:
        video_tensor: Video tensor [B, F, H, W, C] or [F, H, W, C]
        num_frames: Number of frames to extract from the end

    Returns:
        motion_frames: Tensor [num_frames, H, W, C] or [1, num_frames, H, W, C] if batch
    """
    # Handle batch dimension
    has_batch = video_tensor.ndim == 5

    if has_batch:
        # [B, F, H, W, C] -> [B, num_frames, H, W, C]
        motion_frames = video_tensor[:, -num_frames:]
    else:
        # [F, H, W, C] -> [num_frames, H, W, C]
        motion_frames = video_tensor[-num_frames:]

    return motion_frames


def concatenate_videos(
    video_list: list[torch.Tensor]
) -> torch.Tensor:
    """
    Concatenate multiple video tensors along the frame dimension.

    Args:
        video_list: List of video tensors [B, F, H, W, C]

    Returns:
        combined_video: Concatenated video tensor [B, F_total, H, W, C]

    Note:
        All videos must have the same B, H, W, C dimensions.
    """
    if not video_list:
        raise ValueError("video_list cannot be empty")

    # Validate dimensions
    first_shape = video_list[0].shape
    for i, video in enumerate(video_list[1:], 1):
        if video.shape[0] != first_shape[0]:  # batch
            raise ValueError(f"Video {i} has different batch size")
        if video.shape[2:] != first_shape[2:]:  # H, W, C
            raise ValueError(f"Video {i} has different spatial/channel dimensions")

    # Concatenate along frame dimension (dim=1)
    combined = torch.cat(video_list, dim=1)

    return combined


def validate_comfyui_video(video_tensor: torch.Tensor) -> Tuple[bool, str]:
    """
    Validate that a tensor is in correct ComfyUI video format.

    Args:
        video_tensor: Tensor to validate

    Returns:
        is_valid: True if valid, False otherwise
        message: Description of validation result
    """
    if not isinstance(video_tensor, torch.Tensor):
        return False, "Input is not a torch.Tensor"

    if video_tensor.ndim != 5:
        return False, f"Expected 5D tensor [B,F,H,W,C], got {video_tensor.ndim}D"

    b, f, h, w, c = video_tensor.shape

    if c not in [1, 3, 4]:
        return False, f"Expected 1, 3, or 4 channels, got {c}"

    if f < 1:
        return False, f"Expected at least 1 frame, got {f}"

    if video_tensor.min() < 0 or video_tensor.max() > 1:
        return False, f"Values should be in [0,1], got [{video_tensor.min():.3f}, {video_tensor.max():.3f}]"

    return True, f"Valid ComfyUI video: [B={b}, F={f}, H={h}, W={w}, C={c}]"


def validate_storymem_video(video_tensor: torch.Tensor) -> Tuple[bool, str]:
    """
    Validate that a tensor is in correct StoryMem video format.

    Args:
        video_tensor: Tensor to validate

    Returns:
        is_valid: True if valid, False otherwise
        message: Description of validation result
    """
    if not isinstance(video_tensor, torch.Tensor):
        return False, "Input is not a torch.Tensor"

    if video_tensor.ndim != 5:
        return False, f"Expected 5D tensor [B,C,F,H,W], got {video_tensor.ndim}D"

    b, c, f, h, w = video_tensor.shape

    if c not in [1, 3, 4]:
        return False, f"Expected 1, 3, or 4 channels, got {c}"

    if f < 1:
        return False, f"Expected at least 1 frame, got {f}"

    return True, f"Valid StoryMem video: [B={b}, C={c}, F={f}, H={h}, W={w}]"
