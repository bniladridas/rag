# RAG

A friendly **AI assistant** that understands **machine learning**, **science fiction movies**, and the **cosmos**.
It can **answer questions**, **perform calculations**, and **search Wikipedia** — all from your **terminal**.

## Quick Start

```bash
# Run with Docker (basic knowledge without API keys)
docker build -f container/Dockerfile -t rag .
docker run -it rag

# Run with API keys
docker run -it -e TMDB_API_KEY=your_key -e NASA_API_KEY=your_key rag

# Run data collection
docker run -it rag rag-collect
```

## What It Can Do

* **Knowledge Areas:** Machine learning, sci-fi movies, space science
* **Built-in Tools:** Calculator, Wikipedia search, time/date checker
* **Command-Line Interface:** Simple to use
* **Modular Design:** Easy to extend and customize
* **Smart Search:** Uses FAISS for fast document retrieval

## Before You Start

* Python 3.8 or higher
* (Optional) API keys for TMDB and NASA
* Review the [CLI Policy](CLI_POLICY.md) for command-line interface standards and best practices

## Installation

### Option 1: From Source

```bash
git clone <repository-url>
cd rag
pip install -r requirements.txt
pip install -e .
```

### Option 2: Docker

```bash
# Build the image
docker build -f container/Dockerfile -t rag .

# Run the assistant
docker run -it rag

# With API keys
docker run -it -e TMDB_API_KEY=your_key -e NASA_API_KEY=your_key rag

# Run data collection
docker run -it rag rag-collect
```

### Option 3: From pip (if published)

```bash
pip install rag
```

## Configuration

### Local Development

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
NASA_API_KEY=your_nasa_api_key_here
```

### Docker

Pass API keys as environment variables:

```bash
docker run -it -e TMDB_API_KEY=your_key -e NASA_API_KEY=your_key rag
```

### GitHub Actions

Set repository secrets:

* `TMDB_API_KEY`
* `NASA_API_KEY`

Keys available from:

* [TMDB API](https://www.themoviedb.org/settings/api)
* [NASA API](https://api.nasa.gov/)

## Data Collection

Populate the knowledge base:

```bash
rag-collect
```

Collects:

* Machine learning documentation
* Sci-fi movie data (TMDB)
* Space and astronomy data (NASA)

## Usage

### Start Interactive Mode

```bash
rag
```

### Start TUI Mode

```bash
rag-tui
```

You can then ask:

* “What is deep learning?”
* “Tell me about the movie Interstellar”
* “Calculate 2^10”
* “WIKI: Quantum Computing”
* “What time is it?”

### Use Tools Directly

* `CALC: sqrt(144)`
* `WIKI: Machine Learning`
* `TIME:`

## Inside the Project

```
src/rag/
├── __init__.py        → Package initialization
├── __main__.py        → CLI entry point
├── config.py          → Handles configuration and API keys
├── data_fetcher.py    → Collects data from APIs
├── rag_engine.py      → Core RAG logic
├── tools.py           → Tools like calculator and wiki search
└── ui/
    ├── __init__.py
    └── tui.py         → Text User Interface
```

## Release Process

The project uses automated releases via GitHub Actions. For details on the release workflow, see [release-webpage/index.html](release-webpage/index.html).

## Development

**Run tests**

```bash
python -m pytest
```

**Lint code**

```bash
python scripts/lint.py
```

**Fix linting issues**

```bash
./scripts/fix_lint.sh
```

**Run CI locally**

```bash
./scripts/run_ci.sh
```

**Run E2E tests locally**

```bash
# Set API keys if available
export TMDB_API_KEY=your_key
export NASA_API_KEY=your_key
./scripts/run_e2e.sh
```

**Run Docker locally**

```bash
./scripts/run_docker.sh
```

**Build package**

```bash
python setup.py sdist bdist_wheel
```

## Automated Updates

This project uses [Dependabot](https://github.com/dependabot) to keep dependencies updated. All updates are automatically tested.

## Git Hooks

Enable commit message validation:

```bash
cp scripts/commit-msg .git/hooks/
chmod +x .git/hooks/commit-msg
```

Rewrite commit messages:

```bash
# Rewrite specific range
./scripts/rewrite_msg.sh HEAD~5..HEAD

# Rewrite all history (use with caution)
./scripts/rewrite_msg.sh --all
```

Commit messages must be lowercase, ≤40 characters, starting with `feat:`, `fix:`, `docs:`, etc.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Open a pull request

## Security

See [Security Policy](SECURITY.md) for reporting vulnerabilities.


## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Support

For help or questions, open an issue on GitHub.
