from helpers.log import get_logger

logger = get_logger(__name__)


def generate_response_cr(
        retrieved_contents, question, llm
):
    """Generate a response using create and refine strategy.

    The first node uses the 'QA' prompt.
    All subsequent nodes use the 'refine' prompt.

    """
    cur_response = None
    fmt_prompts = []
    num_of_contents = len(retrieved_contents)
    for idx, node in enumerate(retrieved_contents):
        print(f"[Processing Content {idx} ... ]")
        context = node.page_content
        if idx == 0:
            fmt_prompt = llm.generate_contextual_prompt(question=question, context=context)
        else:
            fmt_prompt = llm.generate_cr_prompt(
                context=context,
                question=question,
                existing_answer=str(cur_response),
            )

        if idx == num_of_contents - 1:
            cur_response = llm.generate_answer_streaming(fmt_prompt, max_new_tokens=512)
        else:
            cur_response = llm.generate_answer(fmt_prompt, max_new_tokens=512)
        fmt_prompts.append(fmt_prompt)

    return str(cur_response), fmt_prompts


def combine_results(
        texts,
        question,
        llm,
        cur_prompt_list,
        num_children=10,
):
    new_texts = []
    for idx in range(0, len(texts), num_children):
        text_batch = texts[idx: idx + num_children]
        context = "\n\n".join([t for t in text_batch])
        fmt_qa_prompt = llm.generate_contextual_prompt(question=question, context=context)
        combined_response = llm.generate_answer(fmt_qa_prompt, max_new_tokens=512)
        new_texts.append(str(combined_response))
        cur_prompt_list.append(fmt_qa_prompt)

    if len(new_texts) == 1:
        return new_texts[0]
    else:
        return combine_results(
            new_texts, question, llm, num_children=num_children
        )


def generate_response_hs(
        retrieved_contents, question, llm, num_children=10
):
    """Generate a response using hierarchical summarization strategy.

    Combine num_children nodes hierarchically until we get one root node.

    """
    fmt_prompts = []
    node_responses = []
    for content in retrieved_contents:
        context = content.page_content
        fmt_qa_prompt = llm.generate_contextual_prompt(question=question, context=context)
        node_response = llm.generate_answer(fmt_qa_prompt, max_new_tokens=512)
        node_responses.append(node_response)
        fmt_prompts.append(fmt_qa_prompt)

    response_txt = combine_results(
        [str(r) for r in node_responses],
        question,
        llm,
        fmt_prompts,
        num_children=num_children,
    )

    return response_txt, fmt_prompts
