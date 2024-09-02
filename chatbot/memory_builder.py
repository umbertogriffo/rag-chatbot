import argparse
import sys
from pathlib import Path
from typing import List

from bot.memory.embedder import EmbedderHuggingFace
from bot.memory.vector_memory import VectorMemory
from document_loader.loader import DirectoryLoader
from document_loader.text_splitter import Format, create_recursive_text_splitter
from entities.document import Document
from helpers.log import get_logger

logger = get_logger(__name__)


def load_documents(docs_path: Path) -> List[Document]:
    """
    Loads Markdown documents from the specified path.

    Args:
        docs_path (Path): The path to the documents.

    Returns:
        List[Document]: A list of loaded documents.
    """
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.md",
        show_progress=True,
    )
    return loader.load()


def split_chunks(sources: List, chunk_size: int = 512, chunk_overlap: int = 25) -> List:
    """
    Splits a list of sources into smaller chunks.

    Args:
        sources (List): The list of sources to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 512.
        chunk_overlap (int, optional): The amount of overlap between consecutive chunks. Defaults to 0.

    Returns:
        List: A list of smaller chunks obtained from the input sources.
    """
    chunks = []
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def build_memory_index(docs_path: Path, vector_store_path: str, chunk_size: int, chunk_overlap: int):
    logger.info(f"Loading documents from: {docs_path}")
    sources = load_documents(docs_path)
    logger.info(f"Number of loaded documents: {len(sources)}")

    logger.info("Chunking documents...")
    chunks = split_chunks(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"Number of generated chunks: {len(chunks)}")

    logger.info("Creating memory index...")
    embedding = EmbedderHuggingFace().get_embedding()
    VectorMemory.create_memory_index(embedding, chunks, vector_store_path)
    logger.info("Memory Index has been created successfully!")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Builder")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="The maximum size of each chunk. Defaults to 512.",
        required=False,
        default=512,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="The amount of overlap between consecutive chunks. Defaults to 25.",
        required=False,
        default=25,
    )

    return parser.parse_args()


def main(parameters):
    root_folder = Path(__file__).resolve().parent.parent
    doc_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    build_memory_index(
        doc_path,
        str(vector_store_path),
        parameters.chunk_size,
        parameters.chunk_overlap,
    )


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
