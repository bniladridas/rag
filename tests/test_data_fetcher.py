"""
Unit tests for data_fetcher.py
"""

from unittest.mock import Mock, patch

import pytest

from src.rag.data_fetcher import DataFetcher


@pytest.fixture
def data_fetcher():
    return DataFetcher()


@patch("src.rag.data_fetcher.requests.Session.get")
@patch("src.rag.data_fetcher.concurrent.futures.ThreadPoolExecutor")
def test_fetch_all_data(mock_executor, mock_get, data_fetcher):
    # Mock the session get
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [{"title": "Movie1"}]}
    mock_get.return_value = mock_response

    # Mock the executor
    mock_future = Mock()
    mock_future.result.return_value = [{"title": "Movie1"}]
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
    mock_executor.return_value.__enter__.return_value.map.return_value = [mock_future]

    # Mock the config
    data_fetcher.config = Mock()
    data_fetcher.config.MOVIE_PAGES = 1
    data_fetcher.config.COSMOS_DAYS = 1
    data_fetcher.config.TMDB_API_KEY = "fake_key"
    data_fetcher.config.NASA_API_KEY = "fake_key"

    result = data_fetcher.fetch_all_data()
    assert isinstance(result, list)
    # Should have movies and cosmos data
    assert len(result) > 0
