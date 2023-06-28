import argparse
import sys
from pathlib import Path
from typing import List

from langchain.document_loaders import (DirectoryLoader,
                                        UnstructuredMarkdownLoader)
from langchain.text_splitter import MarkdownTextSplitter
from helpers.log import get_logger
from memory.vector_memory import VectorMemory, initialize_embedding

logger = get_logger(__name__)


def load_documents(docs_path: str) -> List:
    """
    Loads Markdown documents from the specified path.

    Args:
        docs_path (str): The path to the documents.

    Returns:
        List: A list of loaded documents.

    Raises:
        None

    Example:
        >>> load_documents('/path/to/documents')
        [document1, document2, document3]
    """
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    return loader.load()


def split_chunks(sources: List, chunk_size: int = 512, chunk_overlap: int = 0) -> List:
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
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def build_memory_index(
    docs_path: str, vector_store_path: str, chunk_size: int, chunk_overlap: int
):
    sources = load_documents(str(docs_path))
    logger.info(f"Number of Documents: {len(sources)}")
    chunks = split_chunks(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"Number of Chunks: {len(chunks)}")
    embedding = initialize_embedding()
    memory = VectorMemory(embedding=embedding)
    memory.create_memory_index(chunks, vector_store_path)
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
        help="The amount of overlap between consecutive chunks. Defaults to 0.",
        required=False,
        default=0,
    )

    return parser.parse_args()


def main(parameters):
    root_folder = Path(__file__).resolve().parent.parent
    doc_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    build_memory_index(
        str(doc_path),
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
