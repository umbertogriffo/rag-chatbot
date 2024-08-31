import re
from typing import List


class CustomMarkdownTextSplitter:
    """Splitting markdown text using custom separators."""

    def __init__(self, separators: List[str], chunk_size: int = 512, chunk_overlap: int = 0) -> None:
        """Initialize the custom markdown text splitter."""
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # Create a regex pattern from the separators
        pattern = "|".join(f"({sep})" for sep in self.separators)
        # Split the text using the pattern
        splits = re.split(pattern, text)
        # Filter out empty strings from the splits
        chunks = [chunk for chunk in splits if chunk]

        # Further split the chunks based on chunk_size and chunk_overlap
        final_chunks = []
        for chunk in chunks:
            for i in range(0, len(chunk), self.chunk_size - self.chunk_overlap):
                final_chunks.append(chunk[i : i + self.chunk_size])

        return final_chunks


# Define the separators
separators = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# Example usage
if __name__ == "__main__":
    text = """# Heading 1
    Some content here.
    ## Heading 2
    More content here.
    ```
    Code block
    ```
    ***
    Horizontal line
    """
    splitter = CustomMarkdownTextSplitter(separators, chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        print(chunk)
