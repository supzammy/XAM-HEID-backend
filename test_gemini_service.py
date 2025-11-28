"""
Unit tests for Gemini AI Service
Tests the AI integration, fallback logic, and error handling
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Set test environment variables before imports
os.environ['GEMINI_API_KEY'] = 'test-key'
os.environ['ENABLE_GEMINI_AI'] = 'true'
os.environ['FALLBACK_TO_ML'] = 'true'

from streamlit_backend.api.gemini_service import GeminiAIService, get_gemini_service


class TestGeminiAIService:
    """Test cases for Gemini AI Service"""
    
    def setup_method(self):
        """Reset singleton before each test"""
        import streamlit_backend.api.gemini_service as gs
        gs._gemini_service = None
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_initialization_success(self, mock_genai):
        """Test successful initialization with API key"""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        service = GeminiAIService()
        
        assert service.is_available()
        assert service.api_key == 'test-key'
        assert service.enabled
        mock_genai.configure.assert_called_once_with(api_key='test-key')
    
    def test_initialization_no_api_key(self):
        """Test initialization without API key falls back to ML"""
        os.environ['GEMINI_API_KEY'] = ''
        service = GeminiAIService()
        
        assert not service.is_available()
        os.environ['GEMINI_API_KEY'] = 'test-key'  # Restore
    
    def test_initialization_disabled(self):
        """Test initialization when AI is disabled"""
        os.environ['ENABLE_GEMINI_AI'] = 'false'
        service = GeminiAIService()
        
        assert not service.is_available()
        os.environ['ENABLE_GEMINI_AI'] = 'true'  # Restore
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_generate_health_insights_success(self, mock_genai):
        """Test successful AI insight generation"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "AI-generated health equity insights"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        service = GeminiAIService()
        data_summary = {
            'total_cases': 1000,
            'disparity_index': 45.5,
            'disease': 'Heart Disease'
        }
        ml_patterns = [
            {'antecedent': ['high_income'], 'consequent': ['low_rate'], 'confidence': 0.8, 'support': 0.3}
        ]
        
        result = service.generate_health_insights(
            data_summary=data_summary,
            disease='Heart Disease',
            year=2023,
            ml_patterns=ml_patterns
        )
        
        assert result['source'] == 'gemini_ai'
        assert result['success']
        assert 'AI-generated' in result['insights']
        assert result['ml_patterns'] == ml_patterns
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_generate_health_insights_api_error(self, mock_genai):
        """Test fallback when Gemini API fails"""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model
        
        service = GeminiAIService()
        data_summary = {'total_cases': 1000, 'disparity_index': 45.5}
        
        result = service.generate_health_insights(
            data_summary=data_summary,
            disease='Diabetes',
            year=2023,
            ml_patterns=[]
        )
        
        assert result['source'] == 'ml_only'
        assert result['success']
        assert 'note' in result
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_answer_health_query_success(self, mock_genai):
        """Test successful QA"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Based on the data, there is a 30% disparity."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        service = GeminiAIService()
        df = pd.DataFrame({
            'state': ['CA', 'TX', 'NY'],
            'rate': [100, 150, 120]
        })
        
        answer = service.answer_health_query(
            query="What is the disparity?",
            context_data=df,
            disease='Cancer',
            year=2023
        )
        
        assert 'disparity' in answer.lower()
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_fallback_to_ml_when_disabled(self, mock_genai):
        """Test that fallback works when AI is explicitly disabled"""
        os.environ['ENABLE_GEMINI_AI'] = 'false'
        service = GeminiAIService()
        
        data_summary = {'total_cases': 500, 'disparity_index': 20.0}
        result = service.generate_health_insights(
            data_summary=data_summary,
            disease='Heart Disease',
            ml_patterns=[]
        )
        
        assert result['source'] == 'ml_only'
        os.environ['ENABLE_GEMINI_AI'] = 'true'  # Restore
    
    def test_singleton_pattern(self):
        """Test that get_gemini_service returns the same instance"""
        service1 = get_gemini_service()
        service2 = get_gemini_service()
        
        assert service1 is service2
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_empty_response_fallback(self, mock_genai):
        """Test fallback when Gemini returns empty response"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = ""  # Empty response
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        service = GeminiAIService()
        result = service.generate_health_insights(
            data_summary={'total_cases': 100},
            disease='Diabetes',
            ml_patterns=[]
        )
        
        assert result['source'] == 'ml_only'
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_ml_only_analysis_high_disparity(self, mock_genai):
        """Test ML-only analysis correctly identifies high disparity"""
        service = GeminiAIService()
        service.model = None  # Force ML-only mode
        
        data_summary = {
            'total_cases': 1000,
            'disparity_index': 65.0  # High disparity
        }
        
        result = service._ml_only_analysis(
            data_summary=data_summary,
            disease='Cancer',
            ml_patterns=[]
        )
        
        assert 'High disparity' in result['insights']
        assert '65.0' in result['insights']
    
    @patch('streamlit_backend.api.gemini_service.genai')
    def test_prompt_building(self, mock_genai):
        """Test that prompts are built correctly"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        service = GeminiAIService()
        
        data_summary = {'total_cases': 500}
        ml_patterns = [{'description': 'Pattern 1'}]
        
        service.generate_health_insights(
            data_summary=data_summary,
            disease='Heart Disease',
            year=2023,
            ml_patterns=ml_patterns
        )
        
        # Verify generate_content was called with a string prompt
        call_args = mock_model.generate_content.call_args[0][0]
        assert 'Heart Disease' in call_args
        assert '2023' in call_args
        assert 'Pattern 1' in call_args


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
