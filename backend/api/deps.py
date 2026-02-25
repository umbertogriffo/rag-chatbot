"""
Defines dependencies used by the endpoints.
"""
from typing import Annotated, Generator

from bot.client.lama_cpp_client import LamaCppClient
from fastapi import Depends
from llm_client import llm_client


def get_llm_client() -> Generator[LamaCppClient, None, None]:
    """
    Dependency to get the LLM client instance.
    """

    yield llm_client


LamaCppClientDep = Annotated[LamaCppClient, Depends(get_llm_client)]
