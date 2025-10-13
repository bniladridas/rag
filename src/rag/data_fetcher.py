"""
Data fetching module for collecting knowledge from various APIs
"""

import concurrent.futures
import json
import os
import time
from typing import List

import requests

from .config import Config


class DataFetcher:
    """Handles parallel data collection from multiple sources"""

    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.DATASET_DIR, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RAG-Transformer/1.0"})

    def fetch_ml_knowledge(self) -> List[str]:
        """Fetch machine learning knowledge from Wikipedia"""
        ml_topics = [
            "Machine_learning",
            "Deep_learning",
            "Neural_network",
            "Supervised_learning",
            "Unsupervised_learning",
            "Reinforcement_learning",
            "Feature_extraction",
            "Overfitting_(machine_learning)",
        ]

        documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_topic = {
                executor.submit(self._fetch_wiki_summary, topic): topic for topic in ml_topics
            }
            for future in concurrent.futures.as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    summary = future.result()
                    if summary:
                        documents.append(f"Machine Learning - {topic.replace('_', ' ')}: {summary}")
                except Exception as e:
                    print(f"Error fetching {topic}: {e}")

        return documents

    def _fetch_wiki_summary(self, topic: str) -> str:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("extract", "")
            return ""
        except Exception:
            return ""

    def fetch_sci_fi_movies(self) -> List[str]:
        """Fetch science fiction movies from TMDB"""
        if not self.config.TMDB_API_KEY:
            print("TMDB_API_KEY missing, skipping movie fetch.")
            return []

        movie_documents = []

        def fetch_page(page):
            try:
                url = (
                    f"https://api.themoviedb.org/3/discover/movie?"
                    f"api_key={self.config.TMDB_API_KEY}&with_genres=878&"
                    f"page={page}&language=en-US&sort_by=popularity.desc"
                )
                response = self.session.get(url, timeout=10)
                if response.status_code != 200:
                    return []
                page_docs = []
                for movie in response.json().get("results", [])[:10]:
                    doc = (
                        f"Sci-Fi Movie: {movie.get('title','Unknown')}. "
                        f"Release Date: {movie.get('release_date', 'Unknown')}. "
                        f"Overview: {movie.get('overview','No overview')}. "
                        f"Popularity: {movie.get('popularity','N/A')}"
                    )
                    page_docs.append(doc)
                return page_docs
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(fetch_page, page) for page in range(1, self.config.MOVIE_PAGES + 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                movie_documents.extend(future.result())
        return movie_documents

    def fetch_cosmos_content(self) -> List[str]:
        """Fetch cosmos content from NASA API"""
        if not self.config.NASA_API_KEY:
            print("NASA_API_KEY missing, skipping cosmos fetch.")
            return []

        cosmos_documents = []

        def fetch_day(days_ago):
            try:
                base_url = "https://api.nasa.gov/planetary/apod"
                params = {
                    "api_key": self.config.NASA_API_KEY,
                    "date": time.strftime(
                        "%Y-%m-%d", time.localtime(time.time() - days_ago * 86400)
                    ),
                }
                response = self.session.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    content = response.json()
                    return (
                        f"Cosmos: {content.get('title', 'No Title')}. "
                        f"Date: {content.get('date', 'Unknown')}. "
                        f"Explanation: {content.get('explanation', 'No details')}"
                    )
                return None
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [
                executor.submit(fetch_day, days_ago) for days_ago in range(self.config.COSMOS_DAYS)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    cosmos_documents.append(result)
        return cosmos_documents

    def fetch_all_data(self) -> List[str]:
        """Fetch all data sources in parallel"""
        print("Fetching data from multiple sources in parallel...")
        all_documents = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            ml_future = executor.submit(self.fetch_ml_knowledge)
            movies_future = executor.submit(self.fetch_sci_fi_movies)
            cosmos_future = executor.submit(self.fetch_cosmos_content)

            all_documents.extend(ml_future.result())
            all_documents.extend(movies_future.result())
            all_documents.extend(cosmos_future.result())

        print(f"Fetched {len(all_documents)} documents total")
        return all_documents

    def save_documents(self, documents: List[str]):
        """Save documents to knowledge base file"""
        filepath = self.config.KNOWLEDGE_BASE_FILE
        with open(filepath, "w") as f:
            json.dump(documents, f, indent=2)
        print(f"Saved {len(documents)} documents to {filepath}")


def main():
    """Main function for data collection"""
    fetcher = DataFetcher()
    documents = fetcher.fetch_all_data()
    fetcher.save_documents(documents)


if __name__ == "__main__":
    main()
