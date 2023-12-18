from helpers.log import get_logger
from bot.model import Model

logger = get_logger(__name__)


class ContextSynthesisStrategy:
    def __init__(self, llm: Model) -> None:
        self.llm = llm

    def generate_response_cr(self, retrieved_contents, question, return_generator=False):
        """
        Generate a response using create and refine strategy.

        To deal with context overflows, we synthesize a response sequentially through all retrieved contents.
        Start with the first content and generate an initial response.
        Then for subsequent contents, refine the answer using additional context.

        The first content uses the 'Contextual' prompt.
        All subsequent contents use the 'Refine' prompt.

        """
        cur_response = None
        fmt_prompts = []
        num_of_contents = len(retrieved_contents)
        for idx, node in enumerate(retrieved_contents):
            logger.info(f"--- Generating an answer for the content {idx} ... ---")
            context = node.page_content
            if idx == 0:
                fmt_prompt = self.llm.generate_ctx_prompt(
                    question=question, context=context
                )
            else:
                fmt_prompt = self.llm.generate_refined_ctx_prompt(
                    context=context,
                    question=question,
                    existing_answer=str(cur_response),
                )

            if idx == num_of_contents - 1:
                if return_generator:
                    cur_response = self.llm.start_answer_iterator_streamer(fmt_prompt, max_new_tokens=512)
                else:
                    cur_response = self.llm.stream_answer(fmt_prompt, max_new_tokens=512)

            else:
                cur_response = self.llm.generate_answer(fmt_prompt, max_new_tokens=512)
            fmt_prompts.append(fmt_prompt)

        return cur_response, fmt_prompts

    def generate_response_hs(self, retrieved_contents, question, num_children=10):
        """
        Generate a response using hierarchical summarization strategy.

        Combine `num_children` contents hierarchically until we get one root content.
        """
        fmt_prompts = []
        node_responses = []
        for content in retrieved_contents:
            context = content.page_content
            fmt_qa_prompt = self.llm.generate_ctx_prompt(
                question=question, context=context
            )
            node_response = self.llm.stream_answer(fmt_qa_prompt, max_new_tokens=512)
            node_responses.append(node_response)
            fmt_prompts.append(fmt_qa_prompt)

        response_txt = self.combine_results(
            [str(r) for r in node_responses],
            question,
            fmt_prompts,
            num_children=num_children,
        )

        return response_txt, fmt_prompts

    def combine_results(
            self,
            texts,
            question,
            cur_prompt_list,
            num_children=10,
    ):
        new_texts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx: idx + num_children]
            context = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.llm.generate_ctx_prompt(
                question=question, context=context
            )
            combined_response = self.llm.stream_answer(
                fmt_qa_prompt, max_new_tokens=512
            )
            new_texts.append(str(combined_response))
            cur_prompt_list.append(fmt_qa_prompt)

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return self.combine_results(new_texts, question, num_children=num_children)
