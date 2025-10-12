# RAG Transformer

An agentic Retrieval-Augmented Generation (RAG) system for machine learning, science fiction movies, and cosmos knowledge. This AI assistant can answer questions, perform calculations, and search Wikipedia using integrated tools.

## Features

- **Knowledge Domains**: Machine learning concepts, sci-fi movies, and astronomical data
- **Tool Integration**: Built-in calculator, Wikipedia search, and time/date tools
- **Interactive CLI**: Easy-to-use command-line interface
- **Modular Architecture**: Clean, extensible codebase
- **FAISS Vector Search**: Efficient document retrieval using embeddings

## Installation

### Prerequisites
- Python 3.8+
- API keys for TMDB and NASA (optional, for data collection)

### Install from Source
```bash
git clone <repository-url>
cd rag-transformer
pip install -r requirements.txt
pip install -e .
```

### Install via pip (if published)
```bash
pip install rag-transformer
```

## Configuration

Create a `.env` file in the project root with your API keys:

```env
TMDB_API_KEY=your_tmdb_api_key_here
NASA_API_KEY=your_nasa_api_key_here
```

Get API keys from:
- [TMDB API](https://www.themoviedb.org/settings/api)
- [NASA API](https://api.nasa.gov/)

## Data Collection

Collect knowledge base data before running the main application:

```bash
rag-collect
```

This will fetch:
- Machine learning documentation
- Sci-fi movie data from TMDB
- Cosmos data from NASA APIs

## Usage

### Interactive Mode
```bash
rag-transformer
```

This starts an interactive session where you can ask questions about:
- Machine learning concepts
- Science fiction movies
- Astronomical phenomena
- General knowledge with tool assistance

### Example Queries
- "What is deep learning?"
- "Tell me about Inception movie"
- "Calculate 2^10"
- "WIKI: Quantum Computing"
- "What time is it?"

### Direct Tool Usage
You can also use tools directly:
- `CALC: sqrt(144)`
- `WIKI: Machine Learning`
- `TIME:`

## Project Structure

```
src/
├── __init__.py
├── config.py          # Configuration and API keys
├── data_fetcher.py    # Data collection from APIs
├── main.py            # CLI entry point
├── rag_engine.py      # Core RAG logic
└── tools.py           # Tool execution
```

## Development

### Running Tests
```bash
python -m pytest
```

### Linting
```bash
python scripts/lint.py
```

### Building
```bash
python setup.py sdist bdist_wheel
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Add license information here]

## Support

For issues and questions, please open an issue on GitHub.