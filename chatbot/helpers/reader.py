def read_input():
    """
    Reads input from the user until an empty line is entered.

    Returns:
        str: A string containing the concatenated non-empty lines entered by the user,
             separated by newline characters ('\n').
    """
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)
