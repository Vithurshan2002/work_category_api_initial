from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.category import router
from app.services.predictor import CategoryPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup-இல் model load ஆகிறது.
    app.state.predictor = CategoryPredictor()

    yield

    # Application shutdown cleanup.
    del app.state.predictor


app = FastAPI(
    title="Work Category API",
    lifespan=lifespan,
)

app.include_router(router)