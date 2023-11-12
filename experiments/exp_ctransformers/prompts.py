def generate_prompt(template, system, question, context):
    """Generates a prompt for the LLM from the given question and context.

    Returns:
      A string containing the prompt for the LLM.

    Parameters
    ----------
    """

    prompt = template.format(system=system, context=context, question=question)
    return prompt
