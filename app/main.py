from fastapi import FastAPI
from app.routers import analysis, dashboard, auth

app = FastAPI(title="VoicePulse AI")

# Order matters: Auth routes first, then dashboard
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(dashboard.router)

@app.get("/")
def root():
    # This redirects root to login if not authenticated (handled by dashboard logic usually, 
    # but here we just let the routers handle it).
    # The dashboard router has a "/" route that handles the UI.
    return {"message": "Go to /login or /signup"}