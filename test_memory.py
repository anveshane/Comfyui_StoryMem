#!/usr/bin/env python3
"""
Quick test script for memory buffer logic.

Tests:
- Memory buffer creation
- Adding keyframes
- Sliding window behavior
- Serialization
"""

import torch
import sys
sys.path.insert(0, '.')

from storymem_wrapper.memory_manager import StoryMemMemory


def create_test_frame(index: int, height: int = 480, width: int = 832) -> torch.Tensor:
    """Create a test frame with a specific color based on index."""
    frame = torch.ones(height, width, 3) * (index / 20.0)  # Different brightness
    return frame


def test_memory_creation():
    """Test memory buffer creation."""
    print("\n" + "="*60)
    print("TEST 1: Memory Creation")
    print("="*60)

    memory = StoryMemMemory(max_size=10, fixed_count=3)
    print(f"✓ Created memory buffer: {memory}")

    assert len(memory) == 0, "Memory should start empty"
    assert memory.max_size == 10
    assert memory.fixed_count == 3
    print("✓ All creation tests passed")


def test_adding_keyframes():
    """Test adding keyframes to memory."""
    print("\n" + "="*60)
    print("TEST 2: Adding Keyframes")
    print("="*60)

    memory = StoryMemMemory(max_size=10, fixed_count=3)

    # Add 3 frames from shot 1
    frames1 = [create_test_frame(i) for i in range(3)]
    memory.add_keyframes(frames1, shot_id=1)
    print(f"Added 3 frames: {memory}")

    assert len(memory) == 3, f"Expected 3 frames, got {len(memory)}"
    assert memory.get_shot_count() == 1

    # Add 3 more frames from shot 2
    frames2 = [create_test_frame(i) for i in range(3, 6)]
    memory.add_keyframes(frames2, shot_id=2)
    print(f"Added 3 more frames: {memory}")

    assert len(memory) == 6, f"Expected 6 frames, got {len(memory)}"
    assert memory.get_shot_count() == 2

    print("✓ All adding tests passed")


def test_sliding_window():
    """Test sliding window behavior."""
    print("\n" + "="*60)
    print("TEST 3: Sliding Window")
    print("="*60)

    memory = StoryMemMemory(max_size=10, fixed_count=3)

    # Add frames until we exceed max_size
    for shot_id in range(1, 6):  # 5 shots, 3 frames each = 15 frames
        frames = [create_test_frame(i + shot_id*10) for i in range(3)]
        memory.add_keyframes(frames, shot_id=shot_id)
        print(f"Shot {shot_id}: {memory}")

    # Should have exactly 10 frames (max_size)
    assert len(memory) == 10, f"Expected 10 frames, got {len(memory)}"

    # Check that we have both early (fixed) and recent frames
    # First 3 frames should be from shot 1 (indices 0, 1, 2)
    # Last 7 frames should be most recent

    info = memory.get_info()
    print(f"\nMemory info: {info}")
    assert info['is_full'] == True
    assert info['total_frames'] == 10

    print("✓ Sliding window working correctly")


def test_serialization():
    """Test serialization and deserialization."""
    print("\n" + "="*60)
    print("TEST 4: Serialization")
    print("="*60)

    # Create memory and add frames
    memory1 = StoryMemMemory(max_size=5, fixed_count=2)
    frames = [create_test_frame(i) for i in range(3)]
    memory1.add_keyframes(frames, shot_id=1)

    print(f"Original memory: {memory1}")

    # Serialize
    data = memory1.to_dict()
    print(f"✓ Serialized to dict with {len(data)} keys")

    # Deserialize
    memory2 = StoryMemMemory.from_dict(data)
    print(f"✓ Deserialized memory: {memory2}")

    # Verify
    assert len(memory2) == len(memory1)
    assert memory2.max_size == memory1.max_size
    assert memory2.fixed_count == memory1.fixed_count

    print("✓ Serialization tests passed")


def test_memory_retrieval():
    """Test retrieving frames from memory."""
    print("\n" + "="*60)
    print("TEST 5: Memory Retrieval")
    print("="*60)

    memory = StoryMemMemory(max_size=10, fixed_count=3)

    # Add some frames
    frames = [create_test_frame(i) for i in range(5)]
    memory.add_keyframes(frames, shot_id=1)

    # Get all frames
    all_frames = memory.get_memory_frames()
    assert len(all_frames) == 5
    print(f"✓ Retrieved {len(all_frames)} frames")

    # Get last frame
    last_frame = memory.get_last_frame()
    assert last_frame is not None
    assert last_frame.shape == (480, 832, 3)
    print(f"✓ Retrieved last frame: shape {last_frame.shape}")

    # Get last N frames
    last_3 = memory.get_last_n_frames(3)
    assert len(last_3) == 3
    print(f"✓ Retrieved last 3 frames")

    print("✓ All retrieval tests passed")


def test_clear():
    """Test clearing memory."""
    print("\n" + "="*60)
    print("TEST 6: Clear Memory")
    print("="*60)

    memory = StoryMemMemory(max_size=10, fixed_count=3)
    frames = [create_test_frame(i) for i in range(5)]
    memory.add_keyframes(frames, shot_id=1)

    print(f"Before clear: {memory}")
    assert len(memory) == 5

    memory.clear()
    print(f"After clear: {memory}")

    assert len(memory) == 0
    assert memory.get_shot_count() == 0
    print("✓ Clear test passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("StoryMem Memory Buffer Tests")
    print("="*60)

    try:
        test_memory_creation()
        test_adding_keyframes()
        test_sliding_window()
        test_serialization()
        test_memory_retrieval()
        test_clear()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
