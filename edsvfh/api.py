from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from .config import AppConfig
from .pipeline import EventDrivenVerifierPipeline
from .schemas import ObservationPayload, PipelineStatusResponse, VerificationResponse
from .train_public import train_on_fixture
from .models import VerifierBundle


def create_app(bundle_path: str | Path | None = None, config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    if bundle_path is None:
        bundle, metrics, fixture_path = train_on_fixture(output_path=None, config=config)
    else:
        bundle = VerifierBundle.load(bundle_path)
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=config)
    app = FastAPI(title='EDSV-FH Public Reference API', version='0.2.0')
    app.state.pipeline = pipeline

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/v1/bundle')
    def bundle_info() -> dict:
        return app.state.pipeline.bundle.describe()

    @app.post('/v1/reset')
    def reset() -> dict[str, str]:
        app.state.pipeline.reset()
        return {'status': 'reset'}

    @app.get('/v1/status', response_model=PipelineStatusResponse)
    def status() -> PipelineStatusResponse:
        return PipelineStatusResponse(**app.state.pipeline.status())

    @app.post('/v1/step', response_model=VerificationResponse)
    def step(payload: ObservationPayload) -> VerificationResponse:
        out = app.state.pipeline.step(payload.to_observation())
        return VerificationResponse.from_output(out)

    return app
