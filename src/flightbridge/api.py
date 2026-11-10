"""Stateless API. Incoming passenger messages are neither stored nor logged."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from flightbridge.domain import AnalyzeRequest, HandoffRequest, PredictRequest
from flightbridge.service import FlightBridge


def create_app(service: FlightBridge | None = None):
    @asynccontextmanager
    async def lifespan(app):
        if service is not None:
            app.state.service = service
        else:
            directory = Path(os.environ.get("FLIGHTBRIDGE_ARTIFACTS", "artifacts"))
            app.state.service = (
                FlightBridge.load(directory)
                if (directory / "entitlements.joblib").exists()
                else FlightBridge()
            )
        yield

    app = FastAPI(title="FlightBridge", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": app.state.service.bundle is not None}

    @app.get("/ready")
    def ready():
        if app.state.service.bundle is None:
            raise HTTPException(503, "Train a model before serving predictions.")
        return {"status": "ready"}

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        try:
            return app.state.service.analyze(request.message)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            return app.state.service.predict(request.facts, request.confirmed)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/handoff")
    def handoff(request: HandoffRequest):
        try:
            return app.state.service.handoff(request.facts, request.intent, request.confirmed)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    return app


app = create_app()
