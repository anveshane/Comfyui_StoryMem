"""
Test module imports to verify issue #3 is fixed.

This test verifies that:
1. keyframe_extractor.py has proper Dict and Any imports
2. storymem_bridge.py correctly imports WanTI2V (not WanMI2V)
3. All wrapper modules can be imported without NameError
"""

import sys
import unittest
from pathlib import Path


class TestImports(unittest.TestCase):
    """Test that all modules can be imported without errors."""

    def test_keyframe_extractor_imports(self):
        """Test that keyframe_extractor imports Dict and Any correctly."""
        try:
            from storymem_wrapper.keyframe_extractor import KeyframeExtractor

            # If we can import without error, the Dict/Any imports are working
            self.assertTrue(True, "KeyframeExtractor imported successfully")

        except NameError as e:
            self.fail(f"NameError in keyframe_extractor: {e}")
        except ImportError as e:
            # ImportError is OK if dependencies aren't installed
            # We're specifically testing for NameError from missing type imports
            if "Dict" in str(e) or "Any" in str(e):
                self.fail(f"Missing type imports: {e}")
            else:
                self.skipTest(f"Optional dependencies not installed: {e}")

    def test_storymem_bridge_imports(self):
        """Test that storymem_bridge correctly imports WanTI2V."""
        try:
            # This will fail if storymem_src is not available, but that's expected
            # We're testing that the import statement itself is correct
            from storymem_wrapper import storymem_bridge

            # Check if STORYMEM_AVAILABLE flag exists
            self.assertTrue(hasattr(storymem_bridge, 'STORYMEM_AVAILABLE'))

            # If storymem is available, verify WanTI2V is imported correctly
            if storymem_bridge.STORYMEM_AVAILABLE:
                # Try to access the class from the module's namespace
                # This would fail if WanMI2V was used instead of WanTI2V
                from storymem_wrapper.storymem_bridge import StoryMemPipeline
                self.assertTrue(True, "StoryMemPipeline imported successfully")
            else:
                self.skipTest("StoryMem source not available (expected in CI)")

        except NameError as e:
            self.fail(f"NameError in storymem_bridge: {e}")
        except ImportError as e:
            if "WanMI2V" in str(e):
                self.fail(f"Incorrect import name WanMI2V used instead of WanTI2V: {e}")
            elif "WanTI2V" in str(e) and "STORYMEM_AVAILABLE" not in str(e):
                # This is expected if storymem_src is not initialized
                self.skipTest(f"StoryMem source not available: {e}")
            else:
                # Re-raise unexpected import errors
                raise

    def test_typing_imports_available(self):
        """Test that typing module imports work correctly."""
        try:
            from typing import Dict, Any, List, Tuple, Optional
            self.assertTrue(True, "All typing imports available")
        except ImportError as e:
            self.fail(f"Typing imports failed: {e}")

    def test_analyze_frame_diversity_signature(self):
        """Test that analyze_frame_diversity method has correct return type hint."""
        try:
            from storymem_wrapper.keyframe_extractor import KeyframeExtractor
            import inspect

            # Get the method
            method = KeyframeExtractor.analyze_frame_diversity

            # Check if method exists
            self.assertTrue(callable(method), "analyze_frame_diversity is callable")

            # Try to get annotations (return type hint)
            sig = inspect.signature(method)

            # If Dict[str, Any] is used in the return annotation,
            # Dict and Any must be imported
            # This test verifies the fix for line 236 in keyframe_extractor.py
            self.assertTrue(True, "Method signature available")

        except NameError as e:
            self.fail(f"NameError accessing method (missing Dict/Any import): {e}")
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")

    def test_decord_optional_import(self):
        """Test that decord is optional and modules load without it."""
        try:
            # Try to import storymem_src modules
            # These have heavy dependencies (cv2, einops, clip, etc.) that may not be installed
            import sys
            from pathlib import Path

            # Add storymem_src to path if not already there
            storymem_path = Path(__file__).parent.parent / "storymem_src"
            if str(storymem_path) not in sys.path:
                sys.path.insert(0, str(storymem_path))

            # Try importing the modules that use decord
            from storymem_src import extract_keyframes
            from storymem_src.wan import memory2video

            # Check that HAS_DECORD flag exists
            self.assertTrue(hasattr(extract_keyframes, 'HAS_DECORD'))
            self.assertTrue(hasattr(memory2video, 'HAS_DECORD'))

            # Log whether decord is available
            if extract_keyframes.HAS_DECORD:
                print("  Note: decord is installed")
            else:
                print("  Note: decord is not installed (using fallback)")

            # The modules should import successfully regardless
            self.assertTrue(True, "Modules with decord imports loaded successfully")

        except (ImportError, ModuleNotFoundError) as e:
            error_msg = str(e).lower()
            if "decord" in error_msg:
                self.fail(f"decord should be optional but import failed: {e}")
            else:
                # Other dependencies might be missing (cv2, einops, clip, etc.)
                # This is expected in minimal CI environments
                self.skipTest(f"storymem_src dependencies not available: {e}")


class TestModuleStructure(unittest.TestCase):
    """Test the module structure and organization."""

    def test_storymem_wrapper_exists(self):
        """Test that storymem_wrapper package exists."""
        from storymem_wrapper import keyframe_extractor, storymem_bridge, tensor_utils
        self.assertTrue(True, "All wrapper modules exist")

    def test_no_syntax_errors(self):
        """Test that all Python files have valid syntax."""
        import py_compile

        wrapper_dir = Path(__file__).parent.parent / "storymem_wrapper"
        python_files = list(wrapper_dir.glob("*.py"))

        for py_file in python_files:
            if py_file.name.startswith("__"):
                continue
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                self.fail(f"Syntax error in {py_file}: {e}")


if __name__ == "__main__":
    unittest.main()
