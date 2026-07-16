"""App factory. Domain exceptions map to HTTP here and nowhere else."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.draft_service import DraftNotFound, InvalidPick
from api.routers import draft, players


def create_app() -> FastAPI:
    app = FastAPI(title="FantasyForecast API")

    @app.exception_handler(DraftNotFound)
    async def _not_found(request: Request, exc: DraftNotFound):
        return JSONResponse(status_code=404, content={"detail": f"session {exc} not found"})

    @app.exception_handler(InvalidPick)
    async def _invalid(request: Request, exc: InvalidPick):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(players.router)
    app.include_router(draft.router)
    return app


app = create_app()
