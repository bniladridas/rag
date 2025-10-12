# RAG

A friendly **AI assistant** that understands **machine learning**, **science fiction movies**, and the **cosmos**.
It can **answer questions**, **do calculations**, and **search Wikipedia** — all from your **terminal**.

## What It Can Do

* **Knowledge Areas:** Machine learning, sci-fi movies, space science
* **Built-in Tools:** Calculator, Wikipedia search, time/date checker
* **Command-Line Interface:** Simple to use
* **Modular Design:** Easy to extend and customize
* **Smart Search:** Uses FAISS for fast document retrieval

## Before You Start

You’ll need:

* Python 3.8 or higher
* (Optional) API keys for TMDB and NASA

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

# Run with API keys
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

Create a `.env` file in the project root and add your API keys:

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

For CI/CD, set repository secrets in GitHub:
- `TMDB_API_KEY`
- `NASA_API_KEY`

You can get the keys from:

* [TMDB API](https://www.themoviedb.org/settings/api)
* [NASA API](https://api.nasa.gov/)

## Data Collection

Before starting the assistant, collect its knowledge base:

```bash
rag-collect
```

This command gathers:

* Machine learning documentation
* Sci-fi movie data (TMDB)
* Space and astronomy data (NASA)

## Usage

### Start Interactive Mode

```bash
rag
```

Then you can ask:

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
src/
├── 🧩 config.py          → Handles configuration and API keys
├── 🚀 data_fetcher.py    → Collects data from APIs
├── 💬 main.py            → CLI entry point
├── 🧠 rag_engine.py      → Core RAG logic
└── 🛠️ tools.py           → Tools like calculator and wiki search
```

## Development

**Run tests**

```bash
python -m pytest
```

**Lint code**

```bash
python scripts/lint.py
```

**Build package**

```bash
python setup.py sdist bdist_wheel
```

### Automated Updates

This project uses [Dependabot](https://github.com/dependabot) for automatic dependency updates. Dependabot creates pull requests weekly to keep dependencies up-to-date and secure. All updates are automatically tested and validated.

### Git Hooks

This project uses conventional commits. To enable commit message validation:

```bash
cp scripts/commit-msg .git/hooks/
chmod +x .git/hooks/commit-msg
```

For cleaning up commit messages in history:

```bash
# Rewrite specific range
./scripts/rewrite_msg.sh HEAD~5..HEAD

# Rewrite all history (use with caution)
./scripts/rewrite_msg.sh --all
```

Commit messages must be lowercase, ≤40 characters, and start with types like `feat:`, `fix:`, `docs:`, etc.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Open a pull request

## Security

Please see our [Security Policy](SECURITY.md) for information on reporting security vulnerabilities.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

For help or questions, open an issue on GitHub.
