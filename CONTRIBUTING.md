# Contributing to MecaPy

Thank you for your interest in contributing to MecaPy! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists
2. Create a new issue with a clear title and description
3. Include:
   - Python version
   - MecaPy version
   - Steps to reproduce (for bugs)
   - Expected behavior
   - Actual behavior

### Submitting Changes

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Update documentation as needed
6. Commit with clear messages:
   ```bash
   git commit -m "Description of changes"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a Pull Request with a clear description

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings:
  ```python
  def my_function(param1, param2):
      """Brief description.
      
      Longer description if needed.
      
      Args:
          param1: Description
          param2: Description
          
      Returns:
          Description of return value
      """
  ```

### Testing

- Write unit tests for all new features
- Tests should be in the `tests/` directory
- Use pytest for testing
- Aim for >80% code coverage

### Code Quality Tools

```bash
# Format code with black
black .

# Check style with flake8
flake8 .

# Type checking with mypy
mypy mecapy/
```

## Pull Request Process

1. Update the README.md if needed
2. Update documentation in the `docs/` directory
3. Ensure all tests pass: `pytest`
4. Ensure code coverage is maintained or improved
5. Request review from maintainers
6. Address any feedback or requested changes

## Development Setup

All commands below are run from the `mecapy/` subdirectory of the repository,
which is where the Python project lives (`cd mecapy` after cloning).

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests:
   ```bash
   pytest
   ```

## Questions or Need Help?

- Create a discussion in the GitHub repository
- Email: pedrito00.taboada@gmail.com

## License

By contributing to MecaPy, you agree that your contributions will be licensed under its MIT License.
