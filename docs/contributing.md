# Contributing to Bangkong LLM Training System

Thank you for your interest in contributing to the Bangkong LLM Training System! We welcome contributions from the community and are excited to work with you.

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions with the project.

## How to Contribute

### Reporting Bugs

Before reporting a bug, please check if it has already been reported in the [issues](https://github.com/shadowofsorrow/bangkong/issues) section.

When reporting a bug, please include:

1. **Clear Description**: A clear and concise description of the bug
2. **Steps to Reproduce**: Steps to reproduce the behavior
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: OS, Python version, Bangkong version, etc.
6. **Additional Context**: Any other relevant information

### Suggesting Enhancements

We welcome suggestions for new features and enhancements. Before creating a suggestion, please check if it has already been suggested.

When suggesting an enhancement, please include:

1. **Clear Description**: A clear and concise description of the enhancement
2. **Problem Statement**: What problem does this solve?
3. **Proposed Solution**: How would you implement this?
4. **Alternative Solutions**: Any alternative approaches you've considered
5. **Additional Context**: Any other relevant information

### Code Contributions

#### Setting Up Your Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/shadowofsorrow/bangkong.git
   cd bangkong
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

#### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for your changes
4. Update documentation as needed
5. Run quality checks:
   ```bash
   make check
   ```
6. Commit your changes:
   ```bash
   git commit -m "Add feature: brief description of your changes"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Pull Request Process

1. Ensure your code follows the [coding standards](development.md)
2. Include tests for new functionality
3. Update documentation as needed
4. Run all tests to ensure nothing is broken
5. Create a pull request against the `main` branch
6. Address any feedback from reviewers

#### Code Review Process

All pull requests are reviewed by maintainers. The review process typically includes:

1. **Code Quality**: Is the code well-structured and readable?
2. **Tests**: Are there adequate tests for the changes?
3. **Documentation**: Is the documentation updated?
4. **Performance**: Are there any performance considerations?
5. **Security**: Are there any security implications?

## Development Guidelines

### Coding Standards

Please follow the [development guidelines](development.md) for coding standards, testing, and documentation.

### Testing

All code contributions should include tests. We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/unit/test_your_module.py

# Run tests with coverage
pytest --cov=bangkong tests/
```

### Documentation

All public APIs should be documented. We use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is invalid.
    """
    pass
```

### Commit Messages

We follow the conventional commit format:

```
type(scope): brief description

Detailed description of the changes.

Refs: #123
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance changes

### Versioning

We follow Semantic Versioning (SemVer):

- MAJOR version for incompatible API changes
- MINOR version for new functionality
- PATCH version for bug fixes

## Community

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general discussion and questions
- **Community Chat**: [Link to chat if available]

### Recognition

Contributors are recognized in:

- Release notes
- Contributor list
- Community highlights

## Getting Help

If you need help with contributing:

1. Check the documentation
2. Search existing issues
3. Ask in GitHub Discussions
4. Contact maintainers directly

Thank you for contributing to Bangkong!