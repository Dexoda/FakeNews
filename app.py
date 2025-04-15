import os
import sys
import logging
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    try:
        with open("config.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.info("Configuration loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

# Initialize modules
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration on startup
    app.state.config = load_config()

    # Initialize database connection
    from database import init_db
    await init_db(app.state.config)

    logger.info("Server started successfully")
    yield
    logger.info("Server shutting down")

# Initialize FastAPI application
app = FastAPI(lifespan=lifespan)

# Define request models
class NewsAnalysisRequest(BaseModel):
    text: str
    url: str = None

# API endpoints
@app.get("/")
async def root():
    return {"status": "ok", "message": "FakeNewsDetector API is running"}

@app.post("/analyze")
async def analyze_news(request: NewsAnalysisRequest):
    try:
        logger.info(f"Received analysis request: {request.text[:100]}...")

        # Perform multi-level analysis
        from analyzer.pipeline import analyze_text
        analysis_results = await analyze_text(request.text, request.url)

        # Check facts through external APIs
        from factcheck.api_client import check_facts
        factcheck_results = await check_facts(analysis_results["key_claims"])

        # Search open sources
        from factcheck.sources import search_sources
        sources_results = await search_sources(analysis_results["key_claims"])

        # Generate report with visualizations
        from visualization.reports import generate_report
        report = await generate_report(
            request.text,
            analysis_results,
            factcheck_results,
            sources_results
        )

        return report
    except Exception as e:
        logger.error(f"Error analyzing news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))

    logger.info(f"Starting FakeNewsDetector API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
