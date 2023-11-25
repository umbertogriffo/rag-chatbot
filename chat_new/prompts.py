def generate_prompt(template, system, question, context):
    """
    Generate a prompt by formatting a template with provided parameters.

    Args:
        template (str): The template string with placeholders for 'system', 'context', and 'question'.
        system (str): The system information to be inserted into the template.
        question (str): The question to be inserted into the template.
        context (str): The context information to be inserted into the template.

    Returns:
        str: The formatted prompt string.
    """

    prompt = template.format(system=system, context=context, question=question)
    return prompt
