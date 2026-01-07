"""REST API client for tongue server."""

import requests
from typing import Optional


class TongueAPIClient:
    """Client for communicating with the tongue REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", user_id: str = "default"):
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request."""
        if params is None:
            params = {}
        params['user_id'] = self.user_id
        response = self.session.get(f"{self.base_url}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: dict) -> dict:
        """Make a POST request."""
        data['user_id'] = self.user_id
        response = self.session.post(f"{self.base_url}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        """Check if the server is running."""
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

    def get_status(self) -> dict:
        """Get user status and progress."""
        return self._get("/api/status")

    def get_story(self, force_new: bool = False) -> dict:
        """Get current story or generate a new one."""
        params = {'force_new': str(force_new).lower()}
        return self._get("/api/story", params)

    def get_next_sentence(self) -> dict:
        """Get the next sentence for translation."""
        return self._get("/api/next")

    def submit_translation(self, sentence: str, translation: str) -> dict:
        """Submit a translation for evaluation."""
        return self._post("/api/translate", {
            'sentence': sentence,
            'translation': translation
        })

    def get_hint(self, sentence: str) -> dict:
        """Get a hint for the current sentence."""
        return self._post("/api/hint", {
            'sentence': sentence
        })

    def get_missed_words(self, limit: int = 20) -> dict:
        """Get words that need more practice."""
        return self._get("/api/missed-words", {'limit': limit})

    def get_mastered_words(self, limit: int = 50) -> dict:
        """Get mastered words."""
        return self._get("/api/mastered-words", {'limit': limit})
