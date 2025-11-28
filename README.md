# XAM HEID Backend API

## Overview

This is the dedicated backend service for the **XAM HEID (Health Equity Intelligence Dashboard)**. It provides advanced machine learning capabilities and AI-driven insights to analyze health disparities across different demographics and regions.

Built with **FastAPI**, this service handles heavy data processing, association rule mining (Apriori algorithm), and integrates with **Google Gemini AI** to generate natural language insights from health data.

## Features

- **Pattern Mining Engine**: Uses the Apriori algorithm to discover hidden correlations in health data (e.g., "High poverty rates in Region X correlate with increased diabetes prevalence").
- **AI Insights**: Integrates Google Gemini Pro to provide narrative explanations and policy recommendations based on statistical findings.
- **Rule of 11 Compliance**: Automatically suppresses data for small population groups to ensure privacy and HIPAA compliance.
- **High Performance**: Optimized with Pandas and NumPy for efficient data handling.
- **RESTful API**: Fully documented endpoints for easy integration with any frontend (Streamlit, Next.js, React).

## Tech Stack

- **Framework**: FastAPI (Python 3.11)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: mlxtend (Association Rule Mining)
- **AI/LLM**: Google Gemini Pro (via `google-generativeai`)
- **Deployment**: Docker, optimized for Cloud Run / Railway

## Project Structure

```
backend-api/
├── main.py                 # FastAPI application entry point
├── gemini_service.py       # Google Gemini AI integration service
├── pattern_mining.py       # ML logic for association rule mining
├── data_loader.py          # Data ingestion and preprocessing
├── data/                   # Synthetic health datasets
├── Dockerfile              # Container configuration
└── requirements.txt        # Python dependencies
```

## Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/supzammy/XAM-HEID-backend.git
   cd XAM-HEID-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_api_key_here
   ENABLE_GEMINI_AI=true
   ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.
   Interactive docs: `http://localhost:8000/docs`

## Deployment on Railway

This project is optimized for Railway deployment.

1. Fork or push this repo to your GitHub.
2. Log in to [Railway](https://railway.app/).
3. Click **"New Project"** -> **"Deploy from GitHub repo"**.
4. Select this repository.
5. Railway will automatically detect the `Dockerfile` and build the service.
6. **Important**: Go to the "Variables" tab in your Railway project and add your `GEMINI_API_KEY`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health check |
| `POST` | `/filter` | Filter dataset by disease, year, and demographics |
| `POST` | `/api/mine_patterns` | Run ML pattern mining on filtered data |
| `POST` | `/api/ai_insights` | Generate AI-driven insights using Gemini |
| `POST` | `/qa` | Ask natural language questions about the data |

## License

[MIT](LICENSE)
