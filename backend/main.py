from contextlib import asynccontextmanager

import uvicorn
from api.routes import api_router
from bot.memory.document_registry import SQLModel
from core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from helpers.log import get_logger
from vector_database import registry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(registry.engine)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Note: A single Uvicorn worker is probably what you would want to use when using a distributed container
# management system like Kubernetes.

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=settings.HOST,
        port=settings.PORT,
        # log_config=None,
        # workers=max(1, os.cpu_count() - 1),
        workers=1,
    )
