import os
import json
import time
import requests
import concurrent.futures
from typing import List, Dict, Any

# Load API key from environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class DataFetcher:
    def __init__(self):
        """
        Initialize the data fetcher for parallel data collection
        """
        # Set API keys
        self.tmdb_api_key = os.getenv('TMDB_API_KEY', '')
        self.nasa_api_key = os.getenv('NASA_API_KEY', '')

        # Directories for data storage
        self.base_dir = os.path.join(os.path.dirname(__file__), 'datasets')
        os.makedirs(self.base_dir, exist_ok=True)

        # Create session for reuse
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'RAG-Transformer/1.0'})
    
    def make_request(self, url: str, max_retries: int = 3) -> requests.Response:
        """
        Make a request with retry logic
        
        Args:
            url (str): URL to request
            max_retries (int): Maximum number of retries
        
        Returns:
            requests.Response: The response object
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                return response
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise e

    def fetch_ml_knowledge(self) -> List[str]:
        """
        Fetch machine learning knowledge from Wikipedia
        """
        ml_topics = [
            'Machine_learning', 'Deep_learning', 'Neural_network',
            'Supervised_learning', 'Unsupervised_learning', 'Reinforcement_learning',
            'Feature_extraction', 'Overfitting_(machine_learning)'
        ]

        documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_topic = {executor.submit(self._fetch_wiki_summary, topic): topic for topic in ml_topics}
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
        """
        Fetch summary from Wikipedia API
        """
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')
            return ""
        except Exception:
            return ""

    def fetch_sci_fi_movies(self, pages: int = 5) -> List[str]:
        """
        Fetch science fiction movies from TMDB

        Args:
            pages (int): Number of pages to collect

        Returns:
            List of movie document strings
        """
        movie_documents = []

        def fetch_page(page):
            try:
                discover_url = f"https://api.themoviedb.org/3/discover/movie?api_key={self.tmdb_api_key}&with_genres=878&page={page}&language=en-US&sort_by=popularity.desc"
                response = self.session.get(discover_url, timeout=10)
                if response.status_code != 200:
                    return []
                data = response.json()

                page_docs = []
                for movie in data['results'][:10]:  # Limit to 10 per page
                    doc = f"Sci-Fi Movie: {movie['title']}. Release Date: {movie.get('release_date', 'Unknown')}. Overview: {movie['overview']}. Popularity: {movie.get('popularity', 'N/A')}"
                    page_docs.append(doc)
                return page_docs
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(fetch_page, page) for page in range(1, pages + 1)]
            for future in concurrent.futures.as_completed(futures):
                movie_documents.extend(future.result())
                time.sleep(0.5)  # Rate limiting

        return movie_documents
    
    def fetch_cosmos_content(self) -> List[str]:
        """
        Fetch cosmos and astronomy content from NASA API

        Returns:
            List of cosmos document strings
        """
        cosmos_documents = []

        def fetch_day(days_ago):
            try:
                base_url = 'https://api.nasa.gov/planetary/apod'
                params = {
                    'api_key': self.nasa_api_key,
                    'date': time.strftime('%Y-%m-%d', time.localtime(time.time() - days_ago * 86400))
                }

                response = self.session.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    content = response.json()
                    doc = f"Cosmos: {content.get('title', 'No Title')}. Date: {content.get('date', 'Unknown')}. Explanation: {content.get('explanation', 'No details')}"
                    return doc
                return None
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_day, days_ago) for days_ago in range(30)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    cosmos_documents.append(result)
                time.sleep(0.2)  # Rate limiting

        return cosmos_documents
    
    def fetch_all_data(self) -> List[str]:
        """
        Fetch all data sources in parallel

        Returns:
            List of all document strings
        """
        print("Fetching data from multiple sources in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all fetch tasks
            ml_future = executor.submit(self.fetch_ml_knowledge)
            movies_future = executor.submit(self.fetch_sci_fi_movies, 5)
            cosmos_future = executor.submit(self.fetch_cosmos_content)

            # Collect results
            all_documents = []
            all_documents.extend(ml_future.result())
            all_documents.extend(movies_future.result())
            all_documents.extend(cosmos_future.result())

        print(f"Fetched {len(all_documents)} documents total")
        return all_documents

    def save_documents(self, documents: List[str], filename: str):
        """
        Save documents to JSON file

        Args:
            documents (List[str]): Documents to save
            filename (str): Output filename
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(documents, f, indent=2)

        print(f"Saved {len(documents)} documents to {filepath}")

def main():
    # Initialize fetcher
    fetcher = DataFetcher()

    # Fetch all data in parallel
    documents = fetcher.fetch_all_data()

    # Save combined dataset
    fetcher.save_documents(documents, 'knowledge_base.json')

if __name__ == "__main__":
    main()
