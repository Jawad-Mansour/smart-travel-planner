import sys
from pathlib import Path

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ml

app = FastAPI(
    title="Smart Travel Planner API",
    description="ML-powered travel destination classifier and agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ml.router)


@app.get("/")
async def root():
    return {"message": "Smart Travel Planner API", "status": "running"}