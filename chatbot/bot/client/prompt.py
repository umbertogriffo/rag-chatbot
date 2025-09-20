# A string template for the system message.
# This template is used to define the behavior and characteristics of the assistant.
SYSTEM_TEMPLATE = """You are a helpful, respectful and honest assistant.
"""

# A string template for the system message when the assistant can call functions.
# This template is used to define the behavior and characteristics of the assistant
# with the capability to call functions with appropriate input when necessary.
TOOL_SYSTEM_TEMPLATE = """You are a helpful, respectful and honest assistant.
You can call functions with appropriate input when necessary.
"""

# A string template with placeholders for question.
QA_PROMPT_TEMPLATE = """Answer the question below:
{question}
"""

# A string template with placeholders for question, and context.
CTX_PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}
"""

# A string template with placeholders for question, existing_answer, and context.
REFINED_CTX_PROMPT_TEMPLATE = """The original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:
"""

# A string template with placeholders for question, and chat_history to refine the question based on the chat history.
REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:
"""

# A string template with placeholders for question, and chat_history to answer the question based on the chat history.
REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """
You are engaging in a conversation with a human participant who is unaware that they might be
interacting with a machine.
Your goal is to respond in a way that convincingly simulates human-like intelligence and behavior.
The conversation should be natural, coherent, and contextually relevant.
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}\n
Given the context provided in the Chat History and the follow up question, please answer the follow up question above.
If the follow up question isn't correlated to the context provided in the Chat History, please just answer the follow up
question, ignoring the context provided in the Chat History.
Please also don't reformulate the follow up question, and write just a concise answer.
"""


def generate_qa_prompt(template: str, question: str) -> str:
    """
    Generates a prompt for a question-answer task.

    Args:
        template (str): A string template with placeholders for system, question.
        question (str): The question to be included in the prompt.

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(question=question)
    return prompt


def generate_ctx_prompt(template: str, question: str, context: str = "") -> str:
    """
    Generates a prompt for a context-aware question-answer task.

    Args:
        template (str): A string template with placeholders for question, and context.
        question (str): The question to be included in the prompt.
        context (str, optional): Additional context information. Defaults to "".

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(context=context, question=question)
    return prompt


def generate_refined_ctx_prompt(template: str, question: str, existing_answer: str, context: str = "") -> str:
    """
    Generates a prompt for a refined context-aware question-answer task.

    Args:
        template (str): A string template with placeholders for question, existing_answer, and context.
        question (str): The question to be included in the prompt.
        existing_answer (str): The existing answer associated with the question.
        context (str, optional): Additional context information. Defaults to "".

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(
        context=context,
        existing_answer=existing_answer,
        question=question,
    )
    return prompt


def generate_conversation_awareness_prompt(template: str, question: str, chat_history: str) -> str:
    """
    Generates a prompt for a conversation-awareness task.

    Args:
        template (str): A string template with placeholders for question, and chat_history.
        question (str): The question to be included in the prompt.
        chat_history (str): The chat history associated with the conversation.

    Returns:
        str: The generated prompt.
    """

    prompt = template.format(
        chat_history=chat_history,
        question=question,
    )
    return prompt
