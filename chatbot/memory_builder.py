import argparse
import sys
from pathlib import Path

from bot.memory.document_registry import DocumentRegistry
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.memory.vector_database.id_generator import compute_version_hash, generate_deterministic_id
from document_loader.format import Format
from document_loader.loader import DirectoryLoader
from document_loader.text_splitter import create_recursive_text_splitter
from entities.document import Document
from helpers.log import get_logger

logger = get_logger(__name__)


def load_documents(docs_path: Path) -> list[Document]:
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


def split_chunks(sources: list, chunk_size: int = 1000, chunk_overlap: int = 50) -> list:
    """
    Splits a list of sources into smaller chunks.

    Args:
        sources (List): The list of sources to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The amount of overlap between consecutive chunks. Defaults to 50.

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


def build_memory_index(
    docs_path: Path,
    vector_store_path: str,
    chunk_size: int,
    chunk_overlap: int,
    registry: DocumentRegistry | None = None,
    vector_database: Chroma | None = None,
    full_rebuild: bool = False,
) -> dict:
    """
    Build or incrementally update the vector memory index.

    When *full_rebuild* is ``True`` the Chroma collection and the registry are
    wiped before ingesting everything from scratch.  Otherwise only new and
    changed documents are processed, and deleted documents are removed.

    Returns a stats dict with counts of processed / deleted / skipped docs.
    """
    # ------------------------------------------------------------------
    # bootstrap vector DB + registry if not injected
    # ------------------------------------------------------------------
    if vector_database is None:
        embedding = Embedder()
        vector_database = Chroma(is_persistent=True, persist_directory=str(vector_store_path), embedding=embedding)

    if registry is None:
        registry_path = Path(vector_store_path).parent / "document_registry.db"
        registry = DocumentRegistry(registry_path)

    # ------------------------------------------------------------------
    # full-rebuild: wipe everything first
    # ------------------------------------------------------------------
    if full_rebuild:
        logger.info("Full rebuild requested – wiping collection and registry.")
        vector_database.delete_collection()
        # Re-create the collection after deletion so subsequent operations work
        vector_database = Chroma(
            is_persistent=True,
            persist_directory=str(vector_store_path),
            embedding=vector_database.embedding,
        )
        for rec in registry.get_all():
            registry.remove(rec.document_id)

    # ------------------------------------------------------------------
    # (a) load source docs, compute document_id + version_hash
    # ------------------------------------------------------------------
    logger.info(f"Loading documents from: {docs_path}")
    sources = load_documents(docs_path)
    logger.info(f"Number of loaded documents: {len(sources)}")

    current_docs: dict[str, str] = {}  # {document_id: version_hash}
    doc_map: dict[str, Document] = {}  # {document_id: Document}
    for doc in sources:
        source_path = doc.metadata.get("source", "")
        doc_id = generate_deterministic_id(source_path)
        ver_hash = compute_version_hash(doc.page_content)
        current_docs[doc_id] = ver_hash
        doc_map[doc_id] = doc

    # ------------------------------------------------------------------
    # (b) diff against registry
    # ------------------------------------------------------------------
    new_ids, changed_ids, deleted_ids = registry.get_stale_documents(current_docs)
    logger.info(
        "Diff result – new: %d, changed: %d, deleted: %d, unchanged: %d",
        len(new_ids),
        len(changed_ids),
        len(deleted_ids),
        len(current_docs) - len(new_ids) - len(changed_ids),
    )

    # ------------------------------------------------------------------
    # (c) remove changed / deleted docs from Chroma + registry
    # ------------------------------------------------------------------
    for doc_id in changed_ids | deleted_ids:
        rec = registry.get(doc_id)
        chunk_ids = rec.chunk_ids if rec else None
        vector_database.delete_chunks_by_document_id(doc_id, chunk_ids=chunk_ids)
        if doc_id in deleted_ids:
            registry.remove(doc_id)

    # ------------------------------------------------------------------
    # (d) chunk & ingest new + changed docs
    # ------------------------------------------------------------------
    to_ingest = new_ids | changed_ids
    for doc_id in to_ingest:
        doc = doc_map[doc_id]
        source_path = doc.metadata.get("source", "")
        ver_hash = current_docs[doc_id]

        chunks = split_chunks([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # inject document_id + version_hash into every chunk's metadata
        for chunk in chunks:
            chunk.metadata["document_id"] = doc_id
            chunk.metadata["version_hash"] = ver_hash

        chunk_ids = vector_database.from_chunks(chunks)

        # ------------------------------------------------------------------
        # (e) upsert into registry
        # ------------------------------------------------------------------
        registry.upsert(
            doc_id,
            source=source_path,
            filename=Path(source_path).name,
            size=len(doc.page_content),
            content_type="text/markdown",
            version_hash=ver_hash,
            chunk_ids=chunk_ids,
        )

    stats = {
        "processed": len(to_ingest),
        "deleted": len(deleted_ids),
        "skipped": len(current_docs) - len(to_ingest),
    }
    logger.info("Memory Index updated – %s", stats)
    return stats


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Builder")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="The maximum size of each chunk. Defaults to 1000.",
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="The amount of overlap between consecutive chunks. Defaults to 50.",
        required=False,
        default=50,
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        default=False,
        help="Wipe the vector store and registry and rebuild from scratch.",
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
        full_rebuild=parameters.full_rebuild,
    )


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
