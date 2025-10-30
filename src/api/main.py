from __future__ import annotations

from fastapi import FastAPI

from db.init_db import init_db
from .routes.players import router as players_router
from .routes.games import router as games_router
from .routes.predict import router as predict_router
from .routes.leaderboard import router as leaderboard_router


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="Fantasy Football Predictors API")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(players_router)
    app.include_router(games_router)
    app.include_router(predict_router)
    app.include_router(leaderboard_router)
    return app


app = create_app()


