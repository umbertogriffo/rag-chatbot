"""
Global application state.
Holds singleton instances that are initialized during app startup.
"""


from bot.client.lama_cpp_client import LamaCppClient
from bot.memory.vector_database.chroma import Chroma
from sqlalchemy import Engine

# Global singleton instances
engine: Engine | None = None
llm_client: LamaCppClient | None = None
index: Chroma | None = None
