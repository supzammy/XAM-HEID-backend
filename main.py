"""
FastAPI backend for XAM HEID ML services.
Exposes endpoints for filtering, mining, and QA using the backend modules.
Enhanced with Google Gemini AI integration for advanced insights.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import numpy as np
from typing import Optional, Dict, Any
import pandas as pd

from data_loader import load_data, filter_dataset, aggregate_by_state, apply_rule_of_11
from pattern_mining import make_transactions, run_apriori, summarize_rules
from gemini_service import get_gemini_service

app = FastAPI(
    title="XAM HEID ML & AI Backend",
    description="Backend API for Health Equity Intelligence Dashboard with Google Gemini AI integration",
    version="2.0.0"
)

# Enhanced CORS configuration for Vercel + Cloud Run deployment
allowed_origins_env = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5173')
origins = [origin.strip() for origin in allowed_origins_env.split(',')]

# Add Vercel domain pattern support (allows *.vercel.app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / 'data' / 'synthetic_health.csv'

# Initialize Gemini AI service
gemini_service = get_gemini_service()

def get_data():
    return load_data(str(DATA_PATH))

def normalize_disease_name(disease_name: str) -> str:
    """Converts 'Heart Disease' to 'heart_disease'."""
    return disease_name.lower().replace(' ', '_')

class FilterRequest(BaseModel):
    disease: str
    year: Optional[int] = None
    demographics: Optional[Dict[str, Any]] = None

class MiningRequest(BaseModel):
    disease: str
    year: Optional[int] = None
    demographics: Optional[Dict[str, Any]] = None
    min_support: float = 0.05
    min_confidence: float = 0.6

class QARequest(BaseModel):
    disease: str
    year: Optional[int] = None
    demographics: Optional[Dict[str, Any]] = None
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/filter")
def filter_endpoint(req: FilterRequest):
    try:
        df = get_data()
        # normalize the disease name to match dataframe column names
        disease_normalized = normalize_disease_name(req.disease)
        filtered = filter_dataset(df, disease=disease_normalized, year=req.year, demographics=req.demographics)

        agg = aggregate_by_state(filtered, disease=disease_normalized)
        agg = apply_rule_of_11(agg)
        # Convert NaN (numpy) to JSON-friendly None
        agg_clean = agg.where(pd.notnull(agg), None)

        # Prepare records and sanitize values (replace NaN/inf with None, convert numpy scalars)
        records = agg_clean.to_dict(orient='records')

        def sanitize_value(v):
            # convert numpy scalar to native
            if isinstance(v, (np.integer, np.int64, np.int32)):
                return int(v)
            if isinstance(v, (np.floating, np.float64, np.float32)):
                fv = float(v)
                return None if not math.isfinite(fv) else fv
            if isinstance(v, (np.bool_,)):
                return bool(v)
            # native floats
            if isinstance(v, float):
                return None if not math.isfinite(v) else v
            return v

        sanitized = []
        for r in records:
            newr = {k: sanitize_value(v) for k, v in r.items()}
            sanitized.append(newr)

        encoded = jsonable_encoder(sanitized)
        return JSONResponse(content=encoded)
    except ValueError as e:
        # Expected validation error from filter_dataset mapping/validation
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected error: include the error text in the response for debugging
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mine_patterns")
def mine_patterns_endpoint(req: MiningRequest):
    df = get_data()
    disease_normalized = normalize_disease_name(req.disease)
    try:
        filtered = filter_dataset(df, disease=disease_normalized, year=req.year, demographics=req.demographics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # The make_transactions function now handles the Rule of 11 internally
    tx = make_transactions(filtered, disease=disease_normalized)
    
    if tx.empty:
        return {"rules": []}

    fi, rules = run_apriori(tx, min_support=req.min_support, min_threshold=req.min_confidence)
    summarized = summarize_rules(rules, top_n=10)
    return {"rules": summarized}

@app.post("/qa")
def qa_endpoint(req: QARequest):
    df = get_data()
    disease_normalized = normalize_disease_name(req.disease)
    try:
        filtered = filter_dataset(df, disease=disease_normalized, year=req.year, demographics=req.demographics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Aggregate and apply Rule of 11 before answering
    agg = aggregate_by_state(filtered, disease=disease_normalized)
    agg_secure = apply_rule_of_11(agg)

    # Enhanced: Use Gemini AI if available, otherwise provide basic response
    if gemini_service.is_available():
        result = gemini_service.answer_health_query(
            query=req.query,
            context_data=agg_secure,
            disease=req.disease,
            year=req.year
        )
        answer = result["answer"]
        source = result["source"]
    else:
        # Fallback: Basic response when Gemini not available
        answer = f"Based on the filtered data for {req.disease} in {req.year}, I found {len(agg_secure)} states with data. {req.query}\n\nNote: Enhanced AI analysis requires Gemini API configuration."
        source = "ml_only"
                                                                                                          
    return {"answer": answer, "source": source}


@app.post("/api/ai_insights")
def ai_insights_endpoint(req: MiningRequest):
    """
    New endpoint: Generate AI-driven insights using Gemini API.
    Combines ML pattern mining with Gemini's natural language understanding.
    Falls back to ML-only analysis if Gemini is unavailable.
    """
    df = get_data()
    disease_normalized = normalize_disease_name(req.disease)
    
    try:
        filtered = filter_dataset(df, disease=disease_normalized, year=req.year, demographics=req.demographics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Run ML pattern mining
    tx = make_transactions(filtered, disease=disease_normalized)
    ml_patterns = []
    
    if not tx.empty:
        fi, rules = run_apriori(tx, min_support=req.min_support, min_threshold=req.min_confidence)
        ml_patterns = summarize_rules(rules, top_n=10)
    
    # Generate data summary
    agg = aggregate_by_state(filtered, disease=disease_normalized)
    agg_secure = apply_rule_of_11(agg)
    
    data_summary = {
        "total_states": len(agg_secure),
        "total_cases": int(filtered.shape[0]) if not filtered.empty else 0,
        "disease": req.disease,
        "year": req.year,
    }
    
    # Calculate disparity index if we have valid data
    if not agg_secure.empty and 'rate' in agg_secure.columns:
        valid_rates = agg_secure['rate'].dropna()
        if len(valid_rates) > 0:
            disparity_index = ((valid_rates.max() - valid_rates.min()) / valid_rates.max() * 100)
            data_summary["disparity_index"] = float(disparity_index)
            data_summary["max_rate"] = float(valid_rates.max())
            data_summary["min_rate"] = float(valid_rates.min())
            data_summary["avg_rate"] = float(valid_rates.mean())
    
    # Generate AI insights
    insights_result = gemini_service.generate_health_insights(
        data_summary=data_summary,
        disease=req.disease,
        year=req.year,
        ml_patterns=ml_patterns
    )
    
    return insights_result


@app.get("/api/health_check")
def enhanced_health_check():
    """
    Enhanced health check endpoint with service status information.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "ml_engine": "active",
            "gemini_ai": "active" if gemini_service.is_available() else "inactive",
            "data_loader": "active"
        },
        "features": {
            "pattern_mining": True,
            "ai_insights": gemini_service.is_available(),
            "fallback_mode": gemini_service.fallback_enabled
        }
    }
