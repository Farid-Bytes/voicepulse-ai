from fastapi import FastAPI
from app.routers import analysis, dashboard # Add dashboard

app = FastAPI(title="VoicePulse AI")

app.include_router(analysis.router)
app.include_router(dashboard.router) # Add this

@app.get("/")
def root():
    return {"message": "VoicePulse AI API is running. Go to /dashboard for UI."}