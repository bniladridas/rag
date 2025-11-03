# RAG

A friendly **AI assistant** that understands **machine learning**, **science fiction**, and the **cosmos**.
It can **answer questions**, **perform calculations**, and **search Wikipedia** — all directly from your **terminal**.

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/bniladridas/rag.git
cd rag

# Setup (virtual environment)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[test,dev]"

# Run tests
pytest --cov=src --cov-report=term-missing
```

Or run with **Docker**:

```bash
docker build -f container/Dockerfile -t rag .
docker run -it rag
```

With API keys:

```bash
docker run -it -e TMDB_API_KEY=your_key -e NASA_API_KEY=your_key rag
```

---

## Capabilities

* **Knowledge Areas:** Machine learning, sci-fi, and space science
* **Built-in Tools:** Calculator, Wikipedia search, time/date
* **Interface:** Simple CLI and TUI modes
* **Design:** Modular, extensible, and fast (FAISS-powered search)

---

## Usage

### Interactive Mode

```bash
rag
```

### TUI Mode

```bash
rag-tui
```

Examples:

```
What is deep learning?
Tell me about Interstellar
WIKI: Quantum Computing
CALC: sqrt(144)
TIME:
```

### Data Collection

```bash
rag-collect
```

Fetches data from:

* Machine learning documentation
* TMDB (sci-fi movies)
* NASA (space/astronomy)

---

## Configuration

Create a `.env` file:

```env
TMDB_API_KEY=your_tmdb_api_key
NASA_API_KEY=your_nasa_api_key
```

Or pass via Docker:

```bash
docker run -it -e TMDB_API_KEY=your_key -e NASA_API_KEY=your_key rag
```

API keys:

* [TMDB API](https://www.themoviedb.org/settings/api)
* [NASA API](https://api.nasa.gov/)

---

## Development

```bash
# Lint and format
black . && isort . && flake8 && mypy .

# Run tests
pytest

# Local CI / lint scripts
./scripts/run_ci.sh
./scripts/fix_lint.sh

# Build package
python setup.py sdist bdist_wheel
```

Optional setup:

```bash
./scripts/setup_dev.sh
```

---

## Project Structure

```
src/rag/
├── __main__.py       → CLI entry
├── config.py         → Configuration and API keys
├── data_fetcher.py   → Data collection
├── rag_engine.py     → Core logic
├── tools.py          → Utilities (calc, wiki, etc.)
└── ui/tui.py         → Text-based UI
```

---

## Contribution

1. Fork and branch off `main`
2. Implement changes
3. Run tests and linters
4. Submit a pull request

Refer to:

* [Contributing Guidelines](CONTRIBUTING.md)
* [CLI Policy](CLI_POLICY.md)
* [Changelog](CHANGELOG.md)

---

## Git Hooks & Commit Style

```bash
cp scripts/commit-msg .git/hooks/
chmod +x .git/hooks/commit-msg
```

Commit format:

```
feat: add feature
fix: resolve issue
docs: update readme
```

---

## Automation

* **Dependabot**: Keeps dependencies updated
* **GitHub Actions**: Handles testing and releases
* **Release Webpage**: [release-webpage/index.html](release-webpage/index.html)

---

## License & Security

* Licensed under **Apache 2.0** — see [LICENSE](LICENSE)
* For vulnerabilities, see [Security Policy](SECURITY.md)

---

## Support

For help or questions, please open an issue on GitHub.
