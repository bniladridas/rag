import os
import json
import time
import requests
import psutil

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from memory_profiler import profile
from line_profiler import LineProfiler

# Load API key from environment
from dotenv import load_dotenv
import tmdbsimple as tmdb
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class SciFiDatasetCollector:
    def __init__(self):
        """
        Initialize the dataset collector with TMDB API
        """
        # Set TMDB API key from environment variable
        self.tmdb_api_key = os.getenv('TMDB_API_KEY', '')
        tmdb.API_KEY = self.tmdb_api_key
        
        # Directories for data storage
        self.base_dir = os.path.join(os.path.dirname(__file__), 'datasets')
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_log = []
    
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

    @profile
    def collect_sci_fi_movies(self, pages: int = 10) -> List[Dict[str, Any]]:
        """
        Collect science fiction movies from TMDB

        Args:
            pages (int): Number of pages to collect

        Returns:
            List of movie dictionaries
        """
        sci_fi_movies = []

        # Create a session for connection reuse
        self.session = requests.Session()

        # Start performance tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        try:
            for page in range(1, pages + 1):
                # Fetch sci-fi movies using requests
                discover_url = f"https://api.themoviedb.org/3/discover/movie?api_key={self.tmdb_api_key}&with_genres=878&page={page}&language=en-US&sort_by=popularity.desc"
                try:
                    discover_response = self.make_request(discover_url)
                    if discover_response.status_code != 200:
                        print(f"Failed to fetch page {page}: {discover_response.status_code}")
                        continue
                    data = discover_response.json()
                except Exception as e:
                    print(f"Failed to fetch page {page} after retries: {e}")
                    continue

                # Process each movie
                for movie in data['results']:
                    try:
                        # Fetch detailed movie information
                        detail_url = f"https://api.themoviedb.org/3/movie/{movie['id']}?api_key={self.tmdb_api_key}"
                        detail_response = self.make_request(detail_url)
                        if detail_response.status_code != 200:
                            print(f"Failed to fetch details for movie {movie['id']}: {detail_response.status_code}")
                            continue
                        info = detail_response.json()

                        # Fetch keywords
                        keyword_url = f"https://api.themoviedb.org/3/movie/{movie['id']}/keywords?api_key={self.tmdb_api_key}"
                        keyword_response = self.make_request(keyword_url)
                        keywords = []
                        if keyword_response.status_code == 200:
                            keywords = [k['name'] for k in keyword_response.json()['keywords']]

                        # Extract relevant information
                        sci_fi_movie = {
                            'id': movie['id'],
                            'title': movie['title'],
                            'overview': movie['overview'],
                            'release_date': movie.get('release_date', 'Unknown'),
                            'popularity': movie['popularity'],
                            'vote_average': movie['vote_average'],
                            'genres': [genre['name'] for genre in info.get('genres', [])],
                            'production_companies': [
                                company['name'] for company in info.get('production_companies', [])
                            ],
                            'keywords': keywords
                        }

                        sci_fi_movies.append(sci_fi_movie)
                    except Exception as e:
                        print(f"Failed to process movie {movie['id']}: {e}")
                        continue

                # Rate limiting
                time.sleep(1.0)  # Increased delay
        
        except Exception as e:
            print(f"Error collecting movies: {e}")
        
        # End performance tracking
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Log performance metrics
        performance_metrics = {
            'total_movies': len(sci_fi_movies),
            'collection_time': end_time - start_time,
            'memory_usage': end_memory - start_memory
        }
        self.performance_log.append(performance_metrics)
        
        return sci_fi_movies
    
    def collect_cosmos_content(self) -> List[Dict[str, Any]]:
        """
        Collect cosmos and astronomy-related content from NASA API
        
        Returns:
            List of cosmos-related content
        """
        cosmos_content = []
        
        try:
            # NASA APOD (Astronomy Picture of the Day) API
            nasa_api_key = os.getenv('NASA_API_KEY', '')
            base_url = 'https://api.nasa.gov/planetary/apod'
            
            # Collect multiple days of content
            for days_ago in range(30):
                params = {
                    'api_key': nasa_api_key,
                    'date': time.strftime('%Y-%m-%d', time.localtime(time.time() - days_ago * 86400))
                }
                
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    content = response.json()
                    cosmos_content.append({
                        'date': content.get('date'),
                        'title': content.get('title'),
                        'explanation': content.get('explanation'),
                        'media_type': content.get('media_type'),
                        'url': content.get('url')
                    })
                
                time.sleep(0.5)  # Rate limiting
        
        except Exception as e:
            print(f"Error collecting cosmos content: {e}")
        
        return cosmos_content
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """
        Save dataset to JSON file
        
        Args:
            data (List[Dict]): Dataset to save
            filename (str): Output filename
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    def analyze_performance(self):
        """
        Analyze and print performance metrics
        """
        if not self.performance_log:
            print("No performance data available")
            return
        
        # Convert performance log to DataFrame
        df = pd.DataFrame(self.performance_log)
        
        print("\n--- Performance Analysis ---")
        print(f"Total Collections: {len(df)}")
        print(f"Average Collection Time: {df['collection_time'].mean():.2f} seconds")
        print(f"Average Memory Usage: {df['memory_usage'].mean():.2f} MB")
        print(f"Total Movies Collected: {df['total_movies'].sum()}")

def main():
    print("Starting data collection...")
    # Initialize collector
    collector = SciFiDatasetCollector()
    print(f"TMDB API Key loaded: {bool(collector.tmdb_api_key)}")
    print(f"NASA API Key loaded: {bool(os.getenv('NASA_API_KEY'))}")

    # Collect sci-fi movies
    print("Collecting sci-fi movies...")
    sci_fi_movies = collector.collect_sci_fi_movies(pages=5)
    print(f"Collected {len(sci_fi_movies)} movies")
    collector.save_dataset(sci_fi_movies, 'sci_fi_movies.json')

    # Collect cosmos content
    print("Collecting cosmos content...")
    cosmos_content = collector.collect_cosmos_content()
    print(f"Collected {len(cosmos_content)} cosmos items")
    collector.save_dataset(cosmos_content, 'cosmos_content.json')

    # Analyze performance
    collector.analyze_performance()

if __name__ == "__main__":
    main()
