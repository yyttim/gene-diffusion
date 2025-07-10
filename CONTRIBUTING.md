# Contributing to GeneDiffusion

We welcome contributions to GeneDiffusion! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in our [issue tracker](https://github.com/yourusername/gene-diffusion/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, PyTorch version)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add tests for new functionality
5. Run tests to ensure nothing is broken
6. Commit with descriptive messages
7. Push to your fork
8. Create a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep lines under 100 characters

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_masked_diffusion.py
```

### Documentation

- Update relevant documentation for any API changes
- Add examples for new features
- Keep README.md up to date

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/gene-diffusion.git
cd gene-diffusion

# Create development environment
conda create -n gene-diffusion-dev python=3.8
conda activate gene-diffusion-dev

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

## Code of Conduct

Please be respectful and inclusive in all interactions. We strive to maintain a welcoming environment for all contributors.

## Questions?

Feel free to open an issue for any questions about contributing! 