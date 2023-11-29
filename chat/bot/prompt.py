def generate_qa_prompt(template: str, system: str, question: str):
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


def generate_ctx_prompt(template: str, system: str, question: str, context: str = ""):
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


def generate_refine_prompt(
    template: str, system: str, question: str, existing_answer: str, context: str = ""
):
    """
    Generate a prompt by formatting a template with provided parameters.

    Args:
        template (str): The template string with placeholders for 'system', 'context', and 'question'.
        system (str): The system information to be inserted into the template.
        question (str): The question to be inserted into the template.
        existing_answer (str): The existing answer to be inserted into the template.
        context (str): The context information to be inserted into the template.

    Returns:
        str: The formatted prompt string.
    """

    prompt = template.format(
        system=system,
        context=context,
        existing_answer=existing_answer,
        question=question,
    )
    return prompt
