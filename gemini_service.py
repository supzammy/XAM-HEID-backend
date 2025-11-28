"""
Google Gemini AI Service for XAM HEID
Provides AI-driven healthcare insights with fallback to ML-only mode.
"""
import os
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai package not installed. Gemini AI features will be disabled.")


class GeminiAIService:
    """
    Service class for Google Gemini AI integration.
    Implements fallback logic to ML-only mode if API is unavailable.
    """
    
    def __init__(self):
        self.enabled = os.getenv('ENABLE_GEMINI_AI', 'true').lower() == 'true'
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-pro')
        self.fallback_enabled = os.getenv('FALLBACK_TO_ML', 'true').lower() == 'true'
        self.model = None
        
        if self.enabled and GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini AI initialized successfully with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini AI: {str(e)}")
                self.model = None
        else:
            if not self.enabled:
                logger.info("Gemini AI is disabled via configuration")
            elif not GEMINI_AVAILABLE:
                logger.warning("Gemini AI package not available")
            elif not self.api_key:
                logger.warning("GEMINI_API_KEY not set. AI features disabled.")
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available and configured."""
        return self.model is not None
    
    def generate_health_insights(
        self, 
        data_summary: Dict[str, Any],
        disease: str,
        year: Optional[int] = None,
        ml_patterns: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-driven health equity insights.
        
        Args:
            data_summary: Summary statistics of the filtered dataset
            disease: Disease being analyzed
            year: Year of analysis
            ml_patterns: Patterns discovered by ML association rule mining
        
        Returns:
            Dictionary containing AI insights or fallback to ML-only analysis
        """
        if not self.is_available():
            logger.info("Gemini AI not available, using ML-only analysis")
            return self._ml_only_analysis(data_summary, disease, year, ml_patterns)
        
        try:
            # Construct prompt for Gemini
            prompt = self._build_insight_prompt(data_summary, disease, year, ml_patterns)
            
            # Generate response with timeout and error handling
            response = self.model.generate_content(prompt)
            
            if response.text:
                return {
                    "source": "gemini_ai",
                    "insights": response.text,
                    "ml_patterns": ml_patterns,
                    "data_summary": data_summary,
                    "success": True
                }
            else:
                logger.warning("Empty response from Gemini AI, falling back to ML")
                return self._ml_only_analysis(data_summary, disease, year, ml_patterns)
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            if self.fallback_enabled:
                logger.info("Falling back to ML-only analysis")
                return self._ml_only_analysis(data_summary, disease, year, ml_patterns)
            else:
                return {
                    "source": "error",
                    "error": str(e),
                    "success": False
                }
    
    def answer_health_query(
        self,
        query: str,
        context_data: pd.DataFrame,
        disease: str,
        year: Optional[int] = None
    ) -> str:
        """
        Answer specific questions about health equity data using AI.
        
        Args:
            query: User's question
            context_data: Relevant dataset context
            disease: Disease being queried
            year: Year of interest
        
        Returns:
            AI-generated answer or fallback response
        """
        if not self.is_available():
            return self._ml_only_qa(query, context_data, disease, year)
        
        try:
            # Build context-aware prompt
            prompt = self._build_qa_prompt(query, context_data, disease, year)
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                return self._ml_only_qa(query, context_data, disease, year)
                
        except Exception as e:
            logger.error(f"Gemini QA error: {str(e)}")
            if self.fallback_enabled:
                return self._ml_only_qa(query, context_data, disease, year)
            else:
                return f"Error processing query: {str(e)}"
    
    def _build_insight_prompt(
        self,
        data_summary: Dict[str, Any],
        disease: str,
        year: Optional[int] = None,
        ml_patterns: Optional[List[Dict]] = None
    ) -> str:
        """Build a comprehensive prompt for health equity insights."""
        year_str = f" in {year}" if year else ""
        
        prompt = f"""You are a public health data analyst specializing in health equity. 
Analyze the following healthcare disparity data for {disease}{year_str} and provide actionable insights.

DATA SUMMARY:
{self._format_data_summary(data_summary)}

"""
        if ml_patterns:
            prompt += f"""
MACHINE LEARNING PATTERNS DISCOVERED:
{self._format_ml_patterns(ml_patterns)}

"""
        
        prompt += """
Please provide:
1. Key health equity insights and disparities identified
2. Geographic or demographic patterns of concern
3. Potential root causes or contributing factors
4. Evidence-based policy recommendations to address inequities
5. Priority areas for intervention

Important: Maintain strict data privacy. All data represents aggregated, anonymized information.
Focus on actionable, evidence-based insights for policymakers and public health officials.
"""
        return prompt
    
    def _build_qa_prompt(
        self,
        query: str,
        context_data: pd.DataFrame,
        disease: str,
        year: Optional[int] = None
    ) -> str:
        """Build a prompt for answering specific health equity questions."""
        year_str = f" in {year}" if year else ""
        
        # Create a concise data summary
        data_preview = context_data.head(10).to_string() if not context_data.empty else "No data available"
        stats = context_data.describe().to_string() if not context_data.empty else "No statistics available"
        
        prompt = f"""You are a public health data analyst. Answer the following question about {disease}{year_str} based on the provided data.

QUESTION: {query}

DATA CONTEXT:
{data_preview}

STATISTICAL SUMMARY:
{stats}

Provide a clear, concise, evidence-based answer. If the data doesn't support a definitive answer, explain the limitations.
All data is aggregated and anonymized per healthcare privacy standards (Rule of 11).
"""
        return prompt
    
    def _format_data_summary(self, summary: Dict[str, Any]) -> str:
        """Format data summary for prompt."""
        formatted = []
        for key, value in summary.items():
            formatted.append(f"  - {key}: {value}")
        return "\n".join(formatted)
    
    def _format_ml_patterns(self, patterns: List[Dict]) -> str:
        """Format ML patterns for prompt."""
        if not patterns:
            return "No significant patterns discovered"
        
        formatted = []
        for i, pattern in enumerate(patterns[:5], 1):  # Top 5 patterns
            formatted.append(f"  {i}. {pattern.get('description', str(pattern))}")
        return "\n".join(formatted)
    
    def _ml_only_analysis(
        self,
        data_summary: Dict[str, Any],
        disease: str,
        year: Optional[int] = None,
        ml_patterns: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Fallback to ML-only analysis when Gemini is unavailable.
        Uses existing association rule mining and statistical analysis.
        """
        insights = []
        
        # Statistical insights
        if 'total_cases' in data_summary:
            insights.append(f"Total cases analyzed: {data_summary['total_cases']}")
        
        if 'disparity_index' in data_summary:
            disparity = data_summary['disparity_index']
            if disparity > 50:
                insights.append(f"High disparity detected (Index: {disparity:.1f}%). Significant inequity exists.")
            elif disparity > 25:
                insights.append(f"Moderate disparity (Index: {disparity:.1f}%). Intervention recommended.")
            else:
                insights.append(f"Low disparity (Index: {disparity:.1f}%). Relatively equitable distribution.")
        
        # ML pattern insights
        if ml_patterns:
            insights.append(f"\nKey patterns from association rule mining ({len(ml_patterns)} rules):")
            for pattern in ml_patterns[:3]:
                insights.append(f"  - {pattern.get('description', str(pattern))}")
        
        return {
            "source": "ml_only",
            "insights": "\n".join(insights) if insights else "No significant patterns detected",
            "ml_patterns": ml_patterns,
            "data_summary": data_summary,
            "success": True,
            "note": "Generated using ML-only analysis (Gemini AI unavailable)"
        }
    
    def _ml_only_qa(
        self,
        query: str,
        context_data: pd.DataFrame,
        disease: str,
        year: Optional[int] = None
    ) -> str:
        """
        Fallback QA using basic statistical analysis.
        """
        try:
            num_records = len(context_data)
            
            if num_records == 0:
                return f"No data available for {disease} in {year or 'the selected period'}."
            
            # Generate basic insights from the data
            response = f"Based on the available data for {disease}:\n\n"
            response += f"• Found {num_records} states with reportable data\n"
            
            if 'Value' in context_data.columns:
                avg_value = context_data['Value'].mean()
                max_value = context_data['Value'].max()
                min_value = context_data['Value'].min()
                response += f"• Average value: {avg_value:.2f}\n"
                response += f"• Range: {min_value:.2f} to {max_value:.2f}\n"
            
            response += f"\nRegarding your question: {query}\n"
            response += "Note: Enhanced AI analysis is temporarily unavailable. Using basic statistical analysis."
            
            return response
        except Exception as e:
            logger.error(f"Fallback QA error: {str(e)}")
            return f"Unable to process the question. Please try again."


# Singleton instance
_gemini_service = None

def get_gemini_service() -> GeminiAIService:
    """Get or create the Gemini AI service singleton."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiAIService()
    return _gemini_service
