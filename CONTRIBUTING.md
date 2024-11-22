# Contributing to SSALib

Thank you for your interest in contributing to **SSALib** This document provides
guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.
Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:ADSCIAN/ssalib.git
   cd ssalib
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -e ".[dev]"
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature-or-fix-name
   ```

## Development Process

1. Make your changes in your branch
2. Add or update tests as needed
3. Run the test suite:
   ```bash
   pytest
   ```
4. Run the linter:
   ```bash
   flake8
   black .
   isort .
   ```
5. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if you're adding or modifying features
3. Run the full test suite and make sure all tests pass
4. Push to your fork and submit a pull request:
   ```bash
   git push origin feature-or-fix-name
   ```
5. Wait for maintainers to review your PR

### Pull Request Guidelines

* Keep your changes focused and atomic
* Write clear commit messages
* Include tests for new features
* Update documentation as needed
* Follow the project's coding style

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage report
pytest --cov=project_name
```

## Style Guidelines

This project uses:

* [Black](https://github.com/psf/black) for code formatting
* [isort](https://github.com/PyCQA/isort) for import sorting
* [Flake8](https://flake8.pycqa.org/) for code linting

Before submitting a PR, please ensure your code follows these guidelines:

```bash
black .
isort .
flake8
```

## Documentation

* Update docstrings for any modified functions (we follow numpydoc
  docstrings)
* Update the README.md if you're adding or modifying features
* If necessary, add a tutorial in the notebooks folder,
* Add notes to the CHANGELOG.md under the "Unreleased" section

## Questions or Need Help?

* Open an issue with the question tag
* Email the maintainers at [damien.delforge@adscian.be]

## License

By contributing, you agree that your contributions will be licensed under the
same license as the main project (see [LICENSE.md](LICENSE.md)).

Thank you for contributing to **SSALib**!