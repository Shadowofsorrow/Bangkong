# Bangkong Development Guide

## Overview

This guide provides instructions for developers who want to contribute to the Bangkong LLM Training System. It covers setting up the development environment, project structure, coding standards, and contribution process.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/bangkong.git
   cd bangkong
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks** (Optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Project Structure

```
bangkong/
├── bangkong/           # Main Python package
│   ├── config/         # Configuration management
│   ├── hardware/       # Hardware detection and allocation
│   ├── data/           # Data processing pipeline
│   ├── models/         # Model training and packaging
│   ├── deployment/     # Deployment management
│   ├── monitoring/     # Monitoring and tracking
│   ├── utils/          # Utility functions
│   └── exceptions/     # Custom exceptions
├── configs/            # Configuration files
├── data/               # Data directories
├── docs/               # Documentation
├── notebooks/          # Jupyter notebooks
├── scripts/            # Command-line scripts
├── tests/              # Test suite
├── .github/            # GitHub workflows
└── docker/             # Docker files
```

## Coding Standards

### Python Style Guide

The project follows PEP 8 with some additional conventions:

1. **Line Length**: Maximum 88 characters (Black standard)
2. **Naming Conventions**:
   - Classes: PascalCase
   - Functions and variables: snake_case
   - Constants: UPPER_SNAKE_CASE
3. **Type Hints**: Required for all functions and class methods
4. **Docstrings**: Use Google style docstrings

### Code Formatting

The project uses Black for code formatting and isort for import sorting:

```bash
black .
isort .
```

### Linting

The project uses flake8 for linting and mypy for type checking:

```bash
flake8 .
mypy .
```

### Pre-commit Hooks

To ensure code quality, the project uses pre-commit hooks:

```bash
pre-commit run --all-files
```

## Development Workflow

### Creating a Feature Branch

1. **Update Main Branch**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create Feature Branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

### Making Changes

1. **Write Code**: Follow the coding standards above
2. **Add Tests**: Write unit tests for new functionality
3. **Update Documentation**: Update relevant documentation files
4. **Run Quality Checks**:
   ```bash
   make check
   ```

### Testing

The project uses pytest for testing:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/unit/test_config.py

# Run tests with coverage
pytest --cov=bangkong tests/
```

### Test Structure

```
tests/
├── unit/           # Unit tests
│   ├── test_config.py
│   ├── test_hardware.py
│   └── test_utils.py
├── integration/    # Integration tests
│   ├── test_data_pipeline.py
│   └── test_model_trainer.py
├── e2e/            # End-to-end tests
│   └── test_training_workflow.py
└── fixtures/       # Test data and fixtures
```

## Adding New Features

### Module Structure

When adding new features, follow this structure:

1. **Create Module Directory**: In `bangkong/` directory
2. **Add `__init__.py`**: With module documentation
3. **Implement Functionality**: In separate files
4. **Add Tests**: In corresponding test directory
5. **Update Documentation**: In `docs/` directory

### Example: Adding a New Data Processor

1. **Create Processor File**:
   ```python
   # bangkong/data/processors/csv_processor.py
   from typing import List, Dict
   from ...config.schemas import BangkongConfig
   
   class CsvProcessor:
       def __init__(self, config: BangkongConfig):
           self.config = config
       
       def process(self, data_path: str) -> List[Dict]:
           # Implementation here
           pass
   ```

2. **Add to Data Pipeline**:
   ```python
   # bangkong/data/pipeline.py
   from .processors.csv_processor import CsvProcessor
   
   class DataPipeline:
       def _initialize_processors(self):
           self.processors['csv'] = CsvProcessor(self.config)
   ```

3. **Add Tests**:
   ```python
   # tests/unit/test_csv_processor.py
   def test_csv_processor_initialization():
       # Test implementation
       pass
   ```

## Documentation

### Writing Documentation

Documentation should be clear, concise, and follow the existing style. Use Markdown for documentation files.

### API Documentation

API documentation is maintained in OpenAPI format in `docs/api/openapi.yaml`.

### Code Documentation

All public functions and classes should have docstrings explaining their purpose, parameters, and return values.

## Versioning

The project follows Semantic Versioning (SemVer):

- **Major**: Breaking changes
- **Minor**: New features
- **Patch**: Bug fixes

## Release Process

1. **Update Version**: In `setup.py` and `bangkong/__init__.py`
2. **Update Changelog**: In `CHANGELOG.md`
3. **Create Release Tag**:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

## Continuous Integration

The project uses GitHub Actions for CI:

- **Code Quality**: Formatting, linting, type checking
- **Testing**: Unit, integration, and E2E tests
- **Documentation**: Documentation build checks

## Contributing

### Pull Request Process

1. **Fork the Repository**
2. **Create Feature Branch**
3. **Implement Changes**
4. **Add Tests**
5. **Update Documentation**
6. **Run Quality Checks**
7. **Submit Pull Request**

### Code Review Process

All pull requests require review from maintainers. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation updates
- Performance considerations
- Security implications

### Reporting Issues

When reporting issues, include:

1. **Clear Description**: What the issue is
2. **Steps to Reproduce**: How to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment Information**: OS, Python version, etc.

## Best Practices

### Performance

1. **Profile Code**: Use profiling tools to identify bottlenecks
2. **Optimize Algorithms**: Choose efficient algorithms and data structures
3. **Memory Management**: Be mindful of memory usage
4. **Parallelization**: Use multiprocessing/multithreading where appropriate

### Security

1. **Input Validation**: Validate all inputs
2. **Dependency Management**: Keep dependencies up to date
3. **Secure Coding**: Follow secure coding practices
4. **Privacy**: Handle user data responsibly

### Maintainability

1. **Modular Design**: Keep modules focused and cohesive
2. **Clear Interfaces**: Define clear interfaces between components
3. **Documentation**: Keep documentation up to date
4. **Testing**: Maintain comprehensive test coverage

## Getting Help

If you need help with development:

1. **Check Documentation**: Review existing documentation
2. **Search Issues**: Look for existing issues or discussions
3. **Ask Questions**: Open an issue with your question
4. **Join Community**: Participate in community discussions