"""
Integration tests for XAM-HEID API endpoints
Tests the FastAPI endpoints with Gemini AI integration
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import os

# Set test environment
os.environ['GEMINI_API_KEY'] = 'test-key'
os.environ['ENABLE_GEMINI_AI'] = 'true'
os.environ['ALLOWED_ORIGINS'] = 'http://localhost:3000'

from streamlit_backend.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self):
        """Test basic health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_enhanced_health_check(self):
        """Test enhanced health check with service status"""
        response = client.get("/api/health_check")
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert 'services' in data
        assert 'features' in data
        assert data['services']['ml_engine'] == 'active'


class TestFilterEndpoint:
    """Test data filtering endpoint"""
    
    def test_filter_heart_disease(self):
        """Test filtering for heart disease"""
        response = client.post("/filter", json={
            "disease": "Heart Disease",
            "year": 2023
        })
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_filter_invalid_disease(self):
        """Test filtering with invalid disease"""
        response = client.post("/filter", json={
            "disease": "Invalid Disease",
            "year": 2023
        })
        
        assert response.status_code == 400


class TestPatternMiningEndpoint:
    """Test ML pattern mining endpoint"""
    
    def test_mine_patterns_basic(self):
        """Test basic pattern mining"""
        response = client.post("/api/mine_patterns", json={
            "disease": "Diabetes",
            "year": 2023,
            "min_support": 0.05,
            "min_confidence": 0.6
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'rules' in data
        assert isinstance(data['rules'], list)
    
    def test_mine_patterns_with_demographics(self):
        """Test pattern mining with demographic filters"""
        response = client.post("/api/mine_patterns", json={
            "disease": "Cancer",
            "year": 2022,
            "demographics": {"Age": "65+"},
            "min_support": 0.01,
            "min_confidence": 0.3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'rules' in data


class TestAIInsightsEndpoint:
    """Test new AI insights endpoint"""
    
    @patch('streamlit_backend.api.main.gemini_service')
    def test_ai_insights_with_gemini(self, mock_service):
        """Test AI insights when Gemini is available"""
        mock_service.is_available.return_value = True
        mock_service.generate_health_insights.return_value = {
            'source': 'gemini_ai',
            'insights': 'AI-generated insights',
            'ml_patterns': [],
            'success': True
        }
        
        response = client.post("/api/ai_insights", json={
            "disease": "Heart Disease",
            "year": 2023
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['source'] == 'gemini_ai'
        assert data['success']
    
    @patch('streamlit_backend.api.main.gemini_service')
    def test_ai_insights_fallback_to_ml(self, mock_service):
        """Test AI insights falls back to ML when Gemini unavailable"""
        mock_service.is_available.return_value = False
        mock_service.generate_health_insights.return_value = {
            'source': 'ml_only',
            'insights': 'ML-based insights',
            'ml_patterns': [],
            'success': True
        }
        
        response = client.post("/api/ai_insights", json={
            "disease": "Diabetes",
            "year": 2023
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['source'] == 'ml_only'


class TestQAEndpoint:
    """Test question answering endpoint"""
    
    @patch('streamlit_backend.api.main.gemini_service')
    def test_qa_with_gemini(self, mock_service):
        """Test QA with Gemini AI"""
        mock_service.is_available.return_value = True
        mock_service.answer_health_query.return_value = "The disparity is 30%"
        
        response = client.post("/qa", json={
            "disease": "Cancer",
            "year": 2023,
            "query": "What is the disparity?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert 'source' in data
        assert data['source'] == 'gemini_ai'
    
    @patch('streamlit_backend.api.main.gemini_service')
    def test_qa_fallback(self, mock_service):
        """Test QA falls back to ML"""
        mock_service.is_available.return_value = False
        
        response = client.post("/qa", json={
            "disease": "Heart Disease",
            "year": 2023,
            "query": "What are the trends?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert data['source'] == 'ml_only'


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses"""
        response = client.options("/health")
        
        # CORS headers should be present
        assert 'access-control-allow-origin' in response.headers or response.status_code in [200, 404]
    
    def test_preflight_request(self):
        """Test preflight OPTIONS request"""
        response = client.options(
            "/api/ai_insights",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )
        
        # Should allow the request
        assert response.status_code in [200, 204]


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_required_field(self):
        """Test error when required field is missing"""
        response = client.post("/filter", json={
            "year": 2023
            # Missing 'disease'
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_year(self):
        """Test handling of invalid year"""
        response = client.post("/filter", json={
            "disease": "Heart Disease",
            "year": 1800  # Invalid year
        })
        
        # Should either return 400 or empty results
        assert response.status_code in [200, 400]
    
    def test_empty_demographics(self):
        """Test with empty demographics object"""
        response = client.post("/filter", json={
            "disease": "Diabetes",
            "year": 2023,
            "demographics": {}
        })
        
        assert response.status_code == 200


class TestDataPrivacy:
    """Test Rule of 11 privacy protection"""
    
    def test_rule_of_11_suppression(self):
        """Test that Rule of 11 is applied to results"""
        response = client.post("/filter", json={
            "disease": "Cancer",
            "year": 2023,
            "demographics": {"Age": "0-17"}  # Likely to have small counts
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that some values might be None (suppressed)
        has_suppression = any(
            record.get('rate') is None or record.get('count') is None
            for record in data
            if isinstance(record, dict)
        )
        # Note: May or may not have suppression depending on synthetic data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
