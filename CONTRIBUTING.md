# Contributing to SSALib

Thank you for your interest in contributing to **SSALib** This document provides
guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.
Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Development Priorities and Philosophy

### Core Priorities

SSALib's primary focus is consolidating the codebase to optimize performance, 
maintainability, and scalability. This foundation ensures new features can be 
integrated conveniently and reliably.

### Project Scope

SSALib was developed for teaching and research purposes, focusing on classical
Singular Spectrum Analysis theory for univariate time series decomposition. 
Future development should maintain this focus by integrating mature, essential 
features within the scope of core SSA theory.

### Future Development Areas

Potential areas for expansion include:

- Alternative embedding and SVD matrix approaches (including multichannel SSA)
- New decomposition algorithms and techniques beyond SVD
- Alternative significance testing approaches

Automated grouping methodologies should rather be documented than implemented
since interfacing SSALib with clustering algorithm is relatively 
straightforward.

### Development Philosophy

Given the numerous possible extensions, SSALib emphasizes composition over
inheritance. Expanding the API design should allow both developers and users
to register custom methods easily while maintaining a simple, object-oriented
core codebase.

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
2. Update the documentation if you are adding or modifying features
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

SSALib aligns with the 
[Python Enhancement Proposals](https://peps.python.org/pep-0000/).

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage report
pytest --cov=project_name
```

If you do not have R configured and installed R package Rssa, you can ignore
test_rssa.py.

```bash
pytest --ignore=tests\test_rssa.py
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

* Update docstrings for any modified functions (follow numpydoc
  docstrings),
* Update the README.md if you are adding or modifying features,
* If necessary, add a tutorial in the notebook folder,
* Add notes to the CHANGELOG.md under the "Unreleased" section

## Questions or Need Help?

* Open an issue with the question tag
* Email the maintainers at [damien.delforge@adscian.be]

## License

By contributing, you agree that your contributions will be licensed under the
same license as the main project (see [LICENSE](LICENSE)).

Thank you for contributing to **SSALib**!