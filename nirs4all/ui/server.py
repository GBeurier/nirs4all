from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pathlib import Path
import logging

from nirs4all.ui.predictions_api import router as predictions_router
from nirs4all.ui.workspace_api import router as workspace_router
from nirs4all.ui.files_api import router as files_router


app = FastAPI(title="nirs4all UI (minimal)")

# Allow simple CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static UI from top-level `webapp/` when present
HERE = Path(__file__).resolve()
WEBAPP_DIR = HERE.parents[2] / "webapp"
WEBAPP_MOCKUP_DIR = HERE.parents[2] / "webapp_mockup"
UI_TEMPLATES_DIR = HERE.parents[2] / "ui_templates"
if UI_TEMPLATES_DIR.exists():
    # Use ui_templates/ as the canonical source for UI templates when present
    app.mount("/webapp", StaticFiles(directory=str(UI_TEMPLATES_DIR), html=True), name="ui_templates")

    @app.get("/")
    def _root_redirect_templates():
        return RedirectResponse('/webapp/workspace.html')
elif WEBAPP_DIR.exists():
    app.mount("/webapp", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="webapp")

    @app.get("/")
    def _root_redirect():
        return RedirectResponse('/webapp/workspace.html')
else:
    # Friendly fallback to archived mockup
    if WEBAPP_MOCKUP_DIR.exists():
        app.mount("/webapp", StaticFiles(directory=str(WEBAPP_MOCKUP_DIR), html=True), name="webapp_mockup")

        @app.get("/")
        def _root_redirect_mockup():
            logging.getLogger(__name__).info("Serving fallback mockup UI from %s", WEBAPP_MOCKUP_DIR)
            return RedirectResponse('/webapp/workspace.html')
    else:
        logging.getLogger(__name__).info("webapp directory not found at %s; static UI not mounted", WEBAPP_DIR)


# Include modular routers
app.include_router(predictions_router)
app.include_router(workspace_router)
app.include_router(files_router)


@app.get("/api/ping")
def ping():
    return {"status": "ok", "service": "nirs4all-ui-minimal"}
