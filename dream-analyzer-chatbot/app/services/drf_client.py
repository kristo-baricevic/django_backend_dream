import httpx
from typing import Dict, List, Optional, Any
from app.config import settings

class APIClient:
    """Client for communicating with Django DRF and Dream Analyzer APIs"""
    
    def __init__(self):
        self.django_base_url = settings.DJANGO_API_URL
        self.analyzer_base_url = settings.DREAM_ANALYZER_URL
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    # ==================== DJANGO DRF ENDPOINTS ====================
    
    async def get_journal_entries(
        self, 
        limit: int = 10,
        page: int = 1,
        search: str = None
    ) -> Dict:
        """Get journal entries from Django backend"""
        params = {"page_size": limit, "page": page}
        if search:
            params["entries"] = search
            
        response = await self.client.get(
            f"{self.django_base_url}/entries/",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_journal_entry(self, entry_id: str) -> Dict:
        """Get a single journal entry by ID"""
        response = await self.client.get(
            f"{self.django_base_url}/entries/{entry_id}/"
        )
        response.raise_for_status()
        return response.json()
    
    async def create_journal_entry(self, content: str) -> Dict:
        """Create a new journal entry"""
        response = await self.client.post(
            f"{self.django_base_url}/entries/create/",
            json={"content": content}
        )
        response.raise_for_status()
        return response.json()
    
    async def update_journal_entry(self, entry_id: str, content: str) -> Dict:
        """Update an existing journal entry"""
        response = await self.client.put(
            f"{self.django_base_url}/entries/{entry_id}/update/",
            json={"content": content}
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_journal_entry(self, entry_id: str) -> Dict:
        """Delete a journal entry"""
        response = await self.client.delete(
            f"{self.django_base_url}/entries/{entry_id}/delete/"
        )
        response.raise_for_status()
        return {"success": True}
    
    async def get_cumulative_analyses(
        self,
        page: int = 1,
        page_size: int = 10,
        search: str = None
    ) -> Dict:
        """Get cumulative analyses"""
        params = {"page": page, "page_size": page_size}
        if search:
            params["search"] = search
            
        response = await self.client.get(
            f"{self.django_base_url}/cumulative-analyses/",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_custom_questions(
        self,
        page: int = 1,
        page_size: int = 10,
        search: str = None
    ) -> Dict:
        """Get custom questions and answers"""
        params = {"page": page, "page_size": page_size}
        if search:
            params["search"] = search
            
        response = await self.client.get(
            f"{self.django_base_url}/custom-questions/",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_moods(self) -> Dict:
        """Get available moods"""
        response = await self.client.get(
            f"{self.django_base_url}/entries/moods/"
        )
        response.raise_for_status()
        return response.json()
    
    async def get_settings(self) -> Dict:
        """Get user settings"""
        response = await self.client.get(
            f"{self.django_base_url}/settings/"
        )
        response.raise_for_status()
        results = response.json()
        return results[0] if results else {}
    
    async def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get workflow execution status"""
        response = await self.client.get(
            f"{self.django_base_url}/workflows/{workflow_id}/"
        )
        response.raise_for_status()
        return response.json()
    
    # ==================== DREAM ANALYZER ENDPOINTS ====================
    
    async def analyze_dream(self, content: str, settings: Dict = None) -> Dict:
        """Analyze a dream entry using the dream analyzer"""
        payload = {
            "content": content,
            "settings": settings or {}
        }
        response = await self.client.post(
            f"{self.analyzer_base_url}/analyze",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_cumulative_analysis_workflow(
        self,
        entries: List[Dict],
        settings: Dict = None
    ) -> Dict:
        """Start a cumulative analysis workflow"""
        payload = {
            "entries": entries,
            "settings": settings or {}
        }
        print(f"get_cumulative_analysis_workflow step {analyzer_base_url}")
        response = await self.client.post(
            f"{self.analyzer_base_url}/qa-with-workflow",
            json=payload
        )
        print("get_cumulative_analysis_workflow response", response)

        response.raise_for_status()
        return response.json()
    
    async def ask_custom_question_workflow(
        self,
        question: str,
        entries: List[Dict],
        settings: Dict = None
    ) -> Dict:
        """Ask a custom question with workflow tracking"""
        payload = {
            "question": question,
            "entries": entries,
            "settings": settings or {}
        }
        response = await self.client.post(
            f"{self.analyzer_base_url}/custom-question-with-workflow",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def generate_sample_dream(self, theme: str = "flying") -> Dict:
        """Generate a sample dream based on a theme"""
        response = await self.client.post(
            f"{self.analyzer_base_url}/generate-dream",
            json={"theme": theme}
        )
        response.raise_for_status()
        return response.json()

api_client = APIClient()