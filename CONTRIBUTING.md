# Contributing to RAG Transformer

Thank you for your interest in improving **RAG Transformer**!
We appreciate your contributions and commitment to building a better AI assistant.

---

## Table of Contents

* [Code of Conduct](#code-of-conduct)
* [Getting Started](#getting-started)
* [Development Setup](#development-setup)
* [Code Style](#code-style)
* [Testing](#testing)
* [Pull Request Process](#pull-request-process)
* [Reporting Issues](#reporting-issues)

---

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project, you agree to uphold these standards.

---

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** your environment (see below)
4. **Create a branch** for your changes
5. **Implement and test** your changes
6. **Open a pull request** describing your updates

---

## Development Setup

### Prerequisites

* Python 3.8 or higher
* pip
* Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/bniladridas/rag.git
cd rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[test,dev]"
```

---

## Code Style

The following tools ensure consistent and high-quality code:

| Tool           | Purpose                  |
| -------------- | ------------------------ |
| **Black**      | Code formatting          |
| **Flake8**     | Linting                  |
| **Mypy**       | Type checking            |
| **isort**      | Import sorting           |
| **pre-commit** | Automates quality checks |

### Pre-commit Hooks

Enable automated checks before every commit:

```bash
pip install pre-commit
pre-commit install
```

(Optional) Run hooks on all files:

```bash
pre-commit run --all-files
```

> 💡 If a check fails, fix the issues and retry your commit.

### Manual Checks

```bash
black .
isort .
flake8
mypy .
```

Or run all checks together:

```bash
pre-commit run --all-files
```

---

## Testing

### Running Tests

```bash
pytest            # Run all tests
pytest tests/unit/         # Run unit tests
pytest tests/integration/  # Run integration tests
```

### Test Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

Ensure coverage thresholds are met before submitting a pull request.

---

## Pull Request Process

1. Ensure your code **passes all tests and linters**
2. Update **README.md** and relevant documentation if needed
3. Follow the **[Conventional Commits](https://www.conventionalcommits.org/)** style
4. Write **clear commit messages** and PR descriptions
5. Submit your pull request — reviewers will provide feedback promptly

---

## Reporting Issues

Use the [GitHub Issue Tracker](https://github.com/bniladridas/rag/issues) for bug reports or feature requests.
When reporting a bug, please include:

* **Description** of the issue
* **Steps to reproduce**
* **Expected vs actual behavior**
* **Relevant error messages or logs**
* **Environment details** (Python version, OS, etc.)

---

### Final Notes

Your contributions — whether through code, documentation, or feedback — help make RAG Transformer more robust and user-friendly.
We’re grateful for your involvement and collaboration.