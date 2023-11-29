from typing import List, Tuple

from bot.model import Model
from helpers.log import get_logger

logger = get_logger(__name__)


class ContextSynthesisStrategy:
    def __init__(self, llm: Model) -> None:
        self.llm = llm

    def generate_response_cr(self, retrieved_contents, question):
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
            logger.info(f"[Generating an answer for the content {idx} ... ]")
            context = node.page_content
            if idx == 0:
                fmt_prompt = self.llm.generate_ctx_prompt(
                    question=question, context=context
                )
            else:
                fmt_prompt = self.llm.generate_refine_prompt(
                    context=context,
                    question=question,
                    existing_answer=str(cur_response),
                )

            if idx == num_of_contents - 1:
                cur_response = self.llm.stream_answer(fmt_prompt, max_new_tokens=512)
            else:
                cur_response = self.llm.generate_answer(fmt_prompt, max_new_tokens=512)
            fmt_prompts.append(fmt_prompt)

        return str(cur_response), fmt_prompts

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
            text_batch = texts[idx : idx + num_children]
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


class Conversation:
    """
    Question and Answer system using ContextSynthesisStrategy.
    """

    def __init__(self, llm: Model) -> None:
        self.llm = llm
        self.chat_history = []

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """
        Gets the chat history.

        Returns:
            List[Tuple[str, str]]: The chat history, a list of tuples where each tuple
                consists of the question and answer.
        """
        return self.chat_history

    def update_history(self, question: str, answer: str) -> List[Tuple[str, str]]:
        """
        Updates the chat history.

        Args:
            question: The question that was asked.
            answer: The answer that was given.

        Returns:
            List[Tuple[str, str]]: The updated chat history, a list of tuples where each tuple
                consists of the question and answer.
        """
        self.chat_history.append((question, answer))
        self.chat_history = self.keep_chat_history_size()

        return self.chat_history

    def keep_chat_history_size(self, max_size: int = 2) -> List[Tuple[str, str]]:
        """
        Keeps the list of chat history at the specified maximum size by popping out the oldest elements.

        Args:
            max_size: The maximum size of the list.

        Returns:
            The updated list of chat history.
        """

        if len(self.chat_history) > max_size:
            self.chat_history = self.chat_history[-max_size:]
        return self.chat_history

    def answer(self, question: str, retrieved_contents) -> str:
        """
        Generates an answer using the `ContextSynthesisStrategy` for the given question based on the chat history.

        Parameters:
        -----------
        question : str
            The question to generate an answer for.

        Returns:
        -------
        str
            The generated answer for the question.

        """
        strategy = ContextSynthesisStrategy(self.llm)
        # TODO: use the chat history
        answer, fmt_prompts = strategy.generate_response_cr(
            retrieved_contents, question
        )

        # Update the history
        # self.update_history(question, answer)

        return answer
