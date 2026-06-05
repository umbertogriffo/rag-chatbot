"""
Global application state.
Holds singleton instances that are initialized during app startup.
"""


from bot.client.llamacpp_client import LlamaCppClient
from bot.memory.vector_database.chroma import Chroma
from sqlalchemy import Engine

# Global singleton instances
engine: Engine | None = None
llm_client: LlamaCppClient | None = None
index: Chroma | None = None
