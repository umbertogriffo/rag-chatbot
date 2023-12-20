def generate_prompt(template: str, system: str, question: str):
    """
    Generate a prompt by formatting a template with provided parameters.

    Args:
        template (str): The template string with placeholders for 'system', 'context', and 'question'.
        system (str): The system information to be inserted into the template.
        question (str): The question to be inserted into the template.

    Returns:
        str: The formatted prompt string.
    """

    prompt = template.format(system=system, question=question)
    return prompt


def generate_summarization_prompt(template: str, text: str):
    prompt = template.format(text=text)
    return prompt
