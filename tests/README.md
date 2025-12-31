# Tests for Comfyui_StoryMem

This directory contains tests to verify the custom nodes work correctly.

## Running Tests Locally

### Install test dependencies

```bash
pip install -r tests/requirements-test.txt
```

### Install core dependencies

```bash
pip install torch torchvision numpy pillow easydict typing_extensions
```

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_imports.py -v
```

### Run with coverage

```bash
pytest tests/ --cov=storymem_wrapper --cov-report=html
```

## Test Structure

### `test_imports.py`

Tests for issue #3 - Import errors preventing module loading:

- **TestImports**: Verifies all imports work correctly
  - `test_keyframe_extractor_imports`: Tests Dict and Any imports in keyframe_extractor.py
  - `test_storymem_bridge_imports`: Tests WanTI2V import (not WanMI2V) in storymem_bridge.py
  - `test_typing_imports_available`: Verifies typing module is available
  - `test_analyze_frame_diversity_signature`: Tests method with Dict[str, Any] return type

- **TestModuleStructure**: Tests module organization
  - `test_storymem_wrapper_exists`: Verifies all wrapper modules exist
  - `test_no_syntax_errors`: Checks Python syntax in all files

## CI/CD Integration

The tests run automatically on:
- Push to main/master branches
- Push to fix/** branches
- Pull requests to main/master

See `.github/workflows/test.yml` for the full CI configuration.

## Test Philosophy

These tests focus on:
1. **Import correctness**: Ensuring no NameError or ImportError from missing types
2. **Syntax validation**: All Python files compile without errors
3. **Module structure**: Core wrapper modules are present and loadable

The tests are designed to pass even when optional dependencies (like the full StoryMem source) are not available, making them suitable for CI environments.
