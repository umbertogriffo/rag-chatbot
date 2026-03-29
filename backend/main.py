from contextlib import asynccontextmanager

import state
import uvicorn
from api.routes import api_router
from core.config import settings
from database import create_db_engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from helpers.log import get_logger
from llm_client import create_llm_client
from vector_database import init_index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize global state
    state.engine = create_db_engine()
    state.llm_client = create_llm_client(settings.MODEL_FOLDER)
    state.index = init_index(settings.VECTOR_STORE_PATH)

    yield

    # Cleanup
    if state.engine:
        state.engine.dispose()
        logger.info("Database engine disposed")
    if state.llm_client:
        state.llm_client.close()
        logger.info("LLM client closed")


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
