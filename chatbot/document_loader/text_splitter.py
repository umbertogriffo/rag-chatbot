"""
MIT License

Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

from entities.document import Document

from document_loader.format import get_separators

logger = logging.getLogger(__name__)


class TextSplitter(ABC):
    """
    Interface for splitting text into chunks.
    TextSplitter class has been extracted and refactored from LangChain's project.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks. A strategy employed to maintain context continuity
                           between adjacent chunks. A small overlap ensures that critical information is not lost at
                           the boundaries of chunks.
            length_function: Function that measures the length of given chunks.
            keep_separator: Whether to keep the separator in the chunks.
            add_start_index: If `True`, includes chunk's start index in metadata.
            strip_whitespace: If `True`, strips whitespace from the start and end of every document.
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size " f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components."""

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        """
        Joins a list of document strings using the specified separator.

        Args:
            docs (list[str]): The list of document strings to join.
            separator (str): The separator to use for joining the document strings.

        Returns:
            str | None: The joined document string, or None if the resulting string is empty.
        """
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)
        docs, current_doc = [], []
        total = 0

        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, " f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Splitting text by recursively divides text based on a hierarchy of separators (e.g., paragraphs, sentences,
    and words). This approach allows for more nuanced splitting, ensuring that chunks maintain semantic coherence.

    RecursiveCharacterTextSplitter class has been extracted and refactored from LangChain's project.
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        Given a large text it recursively tries to split it based on a specified chunk size.
        It does this by using a set of characters. The default characters provided to it are ["\n\n", "\n", " ", ""].
        It takes in the large text then tries to split it by the first character \n\n.
        If the first split by \n\n is still large then it moves to the next character which is \n and tries to split
        by it.
        If it is still larger than our specified chunk size it moves to the next character in the set until we get a
        split that is less than our specified chunk size.

        More details here https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
        """
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self._separators)

    @staticmethod
    def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
        """
        Splits the input text using the specified separator.

        Args:
            text (str): The text to be split.
            separator (str): The separator to use for splitting the text.
            keep_separator (bool): If True, the separator is included in the resulting splits.

        Returns:
            List[str]: A list of strings resulting from the split operation.

        """
        # Now that we have the separator, split the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]


def create_recursive_text_splitter(format: str, **kwargs: Any) -> RecursiveCharacterTextSplitter:
    """
    Factory function to create a RecursiveCharacterTextSplitter instance based on the specified format.

    Args:
        format (Format): The format of the text to be split.
        **kwargs (Any): Additional keyword arguments to be passed to the RecursiveCharacterTextSplitter constructor.

    Returns:
        An instance of RecursiveCharacterTextSplitter configured with the appropriate separators.
    """
    separators = get_separators(format)
    return RecursiveCharacterTextSplitter(separators=separators, **kwargs)
