from enum import Enum


class Format(Enum):
    MARKDOWN = "markdown"


SUPPORTED_FORMATS = {
    Format.MARKDOWN.value: [
        # First, try to split along Markdown headings (starting with level 2)
        "\n#{1,6} ",
        # Note the alternative syntax for headings (below) is not handled here
        # Heading level 2
        # ---------------
        # End of code block
        "```\n",
        # Horizontal lines
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        # Note that this splitter doesn't handle horizontal lines defined
        # by *three or more* of ***, ---, or ___, but this is not handled
        "\n\n",
        "\n",
        " ",
        "",
    ]
}


def get_separators(format: str):
    """
    Retrieve the list of separators for a given format.

    Args:
        format (str): The format for which to retrieve separators.

    Returns:
        list[str]: A list of separators for the specified format.

    Raises:
        KeyError: If the format is not supported.
    """
    separators = SUPPORTED_FORMATS.get(format)

    # validate input
    if separators is None:
        raise KeyError(format + " is a not supported format")

    return separators
