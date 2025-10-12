"""
Data fetching module for collecting knowledge from various APIs
"""
import os
import json
import time
import requests
import concurrent.futures
from typing import List
from .config import Config


class DataFetcher:
    """Handles parallel data collection from multiple sources"""

    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.DATASET_DIR, exist_ok=True)

        # Create session for reuse
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'RAG-Transformer/1.0'})

    def fetch_ml_knowledge(self) -> List[str]:
        """Fetch machine learning knowledge from Wikipedia"""
        ml_topics = [
            'Machine_learning', 'Deep_learning', 'Neural_network',
            'Supervised_learning', 'Unsupervised_learning', 'Reinforcement_learning',
            'Feature_extraction', 'Overfitting_(machine_learning)'
        ]

        documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_topic = {
                executor.submit(self._fetch_wiki_summary, topic): topic
                for topic in ml_topics
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
        """Fetch summary from Wikipedia API"""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')
            return ""
        except Exception:
            return ""

    def fetch_sci_fi_movies(self) -> List[str]:
        """Fetch science fiction movies from TMDB"""
        movie_documents = []

        def fetch_page(page):
            try:
                discover_url = (
                    f"https://api.themoviedb.org/3/discover/movie?"
                    f"api_key={self.config.TMDB_API_KEY}&with_genres=878&"
                    f"page={page}&language=en-US&sort_by=popularity.desc"
                )
                response = self.session.get(discover_url, timeout=10)
                if response.status_code != 200:
                    return []
                data = response.json()

                page_docs = []
                for movie in data['results'][:10]:  # Limit to 10 per page
                    doc = (
                        f"Sci-Fi Movie: {movie['title']}. "
                        f"Release Date: {movie.get('release_date', 'Unknown')}. "
                        f"Overview: {movie['overview']}. "
                        f"Popularity: {movie.get('popularity', 'N/A')}"
                    )
                    page_docs.append(doc)
                return page_docs
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(fetch_page, page)
                for page in range(1, self.config.MOVIE_PAGES + 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                movie_documents.extend(future.result())
                time.sleep(0.5)  # Rate limiting

        return movie_documents

    def fetch_cosmos_content(self) -> List[str]:
        """Fetch cosmos content from NASA API"""
        cosmos_documents = []

        def fetch_day(days_ago):
            try:
                base_url = 'https://api.nasa.gov/planetary/apod'
                params = {
                    'api_key': self.config.NASA_API_KEY,
                    'date': time.strftime(
                        '%Y-%m-%d',
                        time.localtime(time.time() - days_ago * 86400)
                    )
                }

                response = self.session.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    content = response.json()
                    doc = (
                        f"Cosmos: {content.get('title', 'No Title')}. "
                        f"Date: {content.get('date', 'Unknown')}. "
                        f"Explanation: {content.get('explanation', 'No details')}"
                    )
                    return doc
                return None
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [
                executor.submit(fetch_day, days_ago)
                for days_ago in range(self.config.COSMOS_DAYS)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    cosmos_documents.append(result)
                time.sleep(0.2)  # Rate limiting

        return cosmos_documents

    def fetch_all_data(self) -> List[str]:
        """Fetch all data sources in parallel"""
        print("Fetching data from multiple sources in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all fetch tasks
            ml_future = executor.submit(self.fetch_ml_knowledge)
            movies_future = executor.submit(self.fetch_sci_fi_movies)
            cosmos_future = executor.submit(self.fetch_cosmos_content)

            # Collect results
            all_documents = []
            all_documents.extend(ml_future.result())
            all_documents.extend(movies_future.result())
            all_documents.extend(cosmos_future.result())

        print(f"Fetched {len(all_documents)} documents total")
        return all_documents

    def save_documents(self, documents: List[str]):
        """Save documents to knowledge base file"""
        filepath = os.path.join(self.config.DATASET_DIR, self.config.KNOWLEDGE_BASE_FILE)
        with open(filepath, 'w') as f:
            json.dump(documents, f, indent=2)

        print(f"Saved {len(documents)} documents to {filepath}")


def main():
    """Main function for data collection"""
    fetcher = DataFetcher()
    documents = fetcher.fetch_all_data()
    fetcher.save_documents(documents)


if __name__ == "__main__":
    main()
