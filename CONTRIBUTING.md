# Contributing to MediaRef

Thank you for your interest in contributing to MediaRef!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/open-world-agents/mediaref.git
cd mediaref
```

2. Install in development mode with all dependencies:
```bash
# With uv (recommended):
uv sync --all-extras --all-groups

# Or with pip:
pip install -e ".[pyav,torchcodec,hf]"
pip install ipython pytest pytest-cov ruff
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mediaref --cov-report=html

# Run specific test
pytest tests/test_loading.py::TestToNdarrayImage::test_to_ndarray_from_file -xvs
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and small

## Design & Implementation Notes

Understanding these implementation details can help when contributing to MediaRef:

- **Codec caching**: TensorCodec and TorchCodec use reference-counted leases with LRU eviction (default: 10 decoders).
- **PyAV cleanup**: The optional PyAV backend collects Python reference cycles every 10 playback calls; its remote container cache uses `AV_CACHE_SIZE`.
- **Lazy loading**: Video dependencies only imported when needed (not at module import time)

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Reporting Issues

Please include:
- Python version
- MediaRef version
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Open an issue or discussion on GitHub.
