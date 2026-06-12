"""
Global application state.
Holds singleton instances that are initialized during app startup.
"""


from llm_providers.llamacpp_client import LlamaCppClient
from memory.vector_database.chroma import Chroma
from sqlalchemy import Engine

# Global singleton instances
db_engine: Engine | None = None
llm_client: LlamaCppClient | None = None
vector_database: Chroma | None = None
