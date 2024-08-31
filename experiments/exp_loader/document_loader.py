import concurrent.futures
from pathlib import Path
from typing import Any, List

from document import Document
from exp_loader.text_splitter import MarkdownTextSplitter
from helpers.log import get_logger
from tqdm import tqdm
from unstructured.partition.md import partition_md

logger = get_logger(__name__)


class DirectoryLoader:
    """Load from a directory."""

    def __init__(
        self,
        path: Path,
        glob: str = "**/[!.]*",
        recursive: bool = False,
        show_progress: bool = False,
        use_multithreading: bool = False,
        max_concurrency: int = 4,
    ):
        """Initialize with a path to directory and how to glob over it.

        Args:
            path: Path to directory.
            glob: Glob pattern to use to find files. Defaults to "**/[!.]*"
               (all files except hidden).
            recursive: Whether to recursively search for files. Defaults to False.
            show_progress: Whether to show a progress bar. Defaults to False.
            use_multithreading: Whether to use multithreading. Defaults to False.
            max_concurrency: The maximum number of threads to use. Defaults to 4.
        """
        self.path = path
        self.glob = glob
        self.recursive = recursive
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency

    def load(self) -> List[Document]:
        """Load documents."""
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not self.path.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        docs: List[Document] = []
        items = list(self.path.rglob(self.glob) if self.recursive else self.path.glob(self.glob))

        pbar = None
        if self.show_progress:
            pbar = tqdm(total=len(items))

        if self.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                executor.map(lambda i: self.load_file(i, docs, pbar), items)
        else:
            for i in items:
                self.load_file(i, docs, pbar)

        if pbar:
            pbar.close()

        return docs

    def load_file(self, item: Path, docs: List[Document], pbar: Any | None) -> None:
        """Load a file.

        Args:
            item (str): The path to the documents.
            docs: List of documents to append to.
            pbar: Progress bar. Defaults to None.

        """
        if item.is_file():
            try:
                logger.debug(f"Processing file: {str(item)}")
                # Loads Markdown document from the specified path
                elements = partition_md(filename=str(item))
                text = "\n\n".join([str(el) for el in elements])
                docs.extend([Document(page_content=text, metadata={"source": item})])
            finally:
                if pbar:
                    pbar.update(1)


def split_chunks(sources: List[Document], chunk_size: int = 512, chunk_overlap: int = 0) -> List[Document]:
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
    for source in sources:
        text = source.page_content
        metadata = source.metadata
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(Document(page_content=text[i : i + chunk_size], metadata=metadata))
    return chunks


def split_chunks_2(sources: List, chunk_size: int = 512, chunk_overlap: int = 0) -> List:
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


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent
    docs_path = root_folder / "docs"
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.md",
        recursive=True,
        use_multithreading=True,
        show_progress=True,
    )
    documents = loader.load()
    chunks = split_chunks_2(documents, chunk_size=512, chunk_overlap=0)
    for chunk in chunks:
        print(chunk)
