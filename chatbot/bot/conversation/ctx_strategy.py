import asyncio
from enum import Enum
from typing import Any

import nest_asyncio
from entities.document import Document
from helpers.log import get_logger

from bot.client.lama_cpp_client import LamaCppClient

logger = get_logger(__name__)
nest_asyncio.apply()


class SynthesisStrategyType(Enum):
    CREATE_AND_REFINE = "create-and-refine"
    TREE_SUMMARIZATION = "tree-summarization"
    ASYNC_TREE_SUMMARIZATION = "async-tree-summarization"


class BaseSynthesisStrategy:
    """
    Base class for synthesis strategies.

    Attributes:
        llm (LlmClient): The language model client used for generating responses.
    """

    def __init__(self, llm: LamaCppClient) -> None:
        """
        Initialize the synthesis strategy with the provided LlmClient.

        Args:
            llm (LlmClient): The language model client.
        """
        self.llm = llm

    def generate_response(self, retrieved_contents: list[Document], question: str, max_new_tokens: int = 512):
        """
        Generate a response using the synthesis strategy.

        This method should be implemented by subclasses.

        Args:
            retrieved_contents (List[Document]): List of retrieved contents.
            question (str): The question or input prompt.
            max_new_tokens (int, optional): Maximum number of tokens for the generated response. Default is 512.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement generate_response method")


class CreateAndRefineStrategy(BaseSynthesisStrategy):
    """
    Strategy for sequential refinement of responses using retrieved contents.
    """

    def __init__(self, llm: LamaCppClient):
        super().__init__(llm)

    def generate_response(
        self, retrieved_contents: list[Document], question: str, max_new_tokens: int = 512
    ) -> str | Any:
        """
        Generate a response using create and refine strategy.

        To deal with context overflows, we synthesize a response sequentially through all retrieved contents.
        Start with the first content and generate an initial response.
        Then for subsequent contents, refine the answer using additional context.

        The first content uses the 'Contextual' prompt.
        All subsequent contents use the 'Refine' prompt.

        Args:
            retrieved_contents (List[Document]): List of retrieved contents.
            question (str): The question or input prompt.
            max_new_tokens (int, optional): Maximum number of tokens for the generated response. Default is 512.

        Returns:
            Any: A response generator.

        """
        cur_response = None
        fmt_prompts = []

        num_of_contents = len(retrieved_contents)

        for idx, node in enumerate(retrieved_contents, start=1):
            logger.info(f"--- Generating an answer for the chunk {idx} ... ---")
            context = node.page_content
            logger.debug(f"--- Context: '{context}' ... ---")
            if idx == 0:
                fmt_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            else:
                fmt_prompt = self.llm.generate_refined_ctx_prompt(
                    context=context,
                    question=question,
                    existing_answer=str(cur_response),
                )

            if idx == num_of_contents:
                cur_response = self.llm.start_answer_iterator_streamer(fmt_prompt, max_new_tokens=max_new_tokens)

            else:
                cur_response = self.llm.generate_answer(fmt_prompt, max_new_tokens=max_new_tokens)
                logger.debug(f"--- Current response: '{cur_response}' ... ---")
            fmt_prompts.append(fmt_prompt)

        return cur_response, fmt_prompts


class TreeSummarizationStrategy(BaseSynthesisStrategy):
    """
    Strategy for hierarchical summarization of contents.
    """

    def __init__(self, llm: LamaCppClient):
        super().__init__(llm)

    def generate_response(
        self, retrieved_contents: list[Document], question: str, max_new_tokens: int = 512, num_children: int = 2
    ) -> Any:
        """
        Generate a response using hierarchical summarization strategy.

        Combine `num_children` contents hierarchically until we get one root content.
        Args:
            retrieved_contents (List[Document]): List of retrieved contents.
            question (str): The question or input prompt.
            max_new_tokens (int, optional): Maximum number of tokens for the generated response. Default is 512.
            num_children (int, optional): Number of child nodes to create for the response. Default is 2.

        Returns:
            Any: A response generator.
        """
        fmt_prompts = []
        node_responses = []

        for idx, content in enumerate(retrieved_contents, start=1):
            context = content.page_content
            logger.info(f"--- Generating a response for the chunk {idx} ... ---")
            fmt_qa_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            node_response = self.llm.generate_answer(fmt_qa_prompt, max_new_tokens=max_new_tokens)
            node_responses.append(node_response)
            fmt_prompts.append(fmt_qa_prompt)

        response = self.combine_results(
            [str(r) for r in node_responses],
            question,
            fmt_prompts,
            max_new_tokens=max_new_tokens,
            num_children=num_children,
        )

        return response, fmt_prompts

    def combine_results(
        self,
        texts: list[str],
        question: str,
        cur_prompt_list: list[str],
        max_new_tokens: int = 512,
        num_children: int = 2,
    ) -> Any:
        """
        Combine results of hierarchical summarization.

        Args:
            texts (List[str]): List of texts to combine.
            question (str): The question or input prompt.
            cur_prompt_list (List[str]): List of current prompts.
            max_new_tokens (int, optional): Maximum number of tokens for the generated response. Default is 512.
            num_children (int, optional): Number of child nodes to create for the response. Default is 2.

        Returns:
            Any: A response generator.
        """
        fmt_prompts = []
        new_texts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx : idx + num_children]
            context = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            fmt_prompts.append(fmt_qa_prompt)

        if len(fmt_prompts) == 1:
            logger.info("--- Generating final response ... ---")
            combined_response_stream = self.llm.start_answer_iterator_streamer(
                fmt_prompts[0], max_new_tokens=max_new_tokens
            )
            return combined_response_stream
        else:
            logger.info(f"--- Combining {len(fmt_prompts)} responses ... ---")
            for fmt_qa_prompt in fmt_prompts:
                combined_response = self.llm.generate_answer(fmt_qa_prompt, max_new_tokens=max_new_tokens)
                new_texts.append(str(combined_response))
                cur_prompt_list.append(fmt_qa_prompt)
            return self.combine_results(
                new_texts,
                question,
                cur_prompt_list,
                num_children=num_children,
            )


class AsyncTreeSummarizationStrategy(BaseSynthesisStrategy):
    """
    Asynchronous version of TreeSummarizationStrategy.
    """

    def __init__(self, llm: LamaCppClient):
        super().__init__(llm)

    async def generate_response(
        self,
        retrieved_contents: list[Document],
        question: str,
        max_new_tokens: int = 512,
        num_children: int = 2,
    ) -> Any:
        """
        Generate a response using hierarchical summarization strategy.

        Combine `num_children` contents hierarchically until we get one root content.

        Args:
            retrieved_contents (List[Document]): A list of text content for the AI to consider when generating a
                response.
            question (str): The question or input prompt that the AI will use as context for its response.
            max_new_tokens (int, optional): The maximum number of tokens for the generated response. Default is 512.
            num_children (int, optional): The number of child nodes to create for the response. Default is 2.

        Returns:
            Any: A response generator.
        """
        fmt_prompts = []

        for idx, content in enumerate(retrieved_contents, start=1):
            context = content.page_content
            logger.info(f"--- Generating a response for the chunk {idx} ... ---")
            fmt_qa_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            fmt_prompts.append(fmt_qa_prompt)

        tasks = [self.llm.async_generate_answer(p, max_new_tokens=max_new_tokens) for p in fmt_prompts]
        node_responses = await asyncio.gather(*tasks)

        response = await self.combine_results(
            [str(r) for r in node_responses],
            question,
            fmt_prompts,
            max_new_tokens=max_new_tokens,
            num_children=num_children,
        )

        return response, fmt_prompts

    async def combine_results(
        self,
        texts: list[str],
        question: str,
        cur_prompt_list: list[str],
        max_new_tokens: int = 512,
        num_children: int = 2,
    ):
        """
        Combine results of hierarchical summarization.

        Args:
            texts (List[str]): List of texts to combine.
            question (str): The question or input prompt.
            cur_prompt_list (List[str]): List of current prompts.
            max_new_tokens (int, optional): Maximum number of tokens for the generated response. Default is 512.
            num_children (int, optional): Number of child nodes to create for the response. Default is 2.

        Returns:
            Any: A response generator.
        """
        fmt_prompts = []
        for idx in range(0, len(texts), num_children):
            logger.info(f"--- Creating prompts in batches of size {num_children} ... ---")
            text_batch = texts[idx : idx + num_children]
            context = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            fmt_prompts.append(fmt_qa_prompt)

        if len(fmt_prompts) == 1:
            logger.info("--- Generating final response ... ---")
            combined_responses_stream = await asyncio.gather(
                self.llm.async_start_answer_iterator_streamer(fmt_prompts[0], max_new_tokens=max_new_tokens)
            )
            return combined_responses_stream[0]
        else:
            logger.info(f"--- Combining {len(fmt_prompts)} responses ... ---")
            tasks = [self.llm.async_generate_answer(p, max_new_tokens=max_new_tokens) for p in fmt_prompts]
            combined_responses = await asyncio.gather(*tasks)
            new_texts = [str(r) for r in combined_responses]
            return await self.combine_results(
                new_texts,
                question,
                cur_prompt_list,
                num_children=num_children,
            )


STRATEGIES = {
    SynthesisStrategyType.CREATE_AND_REFINE.value: CreateAndRefineStrategy,
    SynthesisStrategyType.TREE_SUMMARIZATION.value: TreeSummarizationStrategy,
    SynthesisStrategyType.ASYNC_TREE_SUMMARIZATION.value: AsyncTreeSummarizationStrategy,
}


def get_ctx_synthesis_strategies():
    return list(STRATEGIES.keys())


def get_ctx_synthesis_strategy(strategy_name: str, **kwargs):
    strategy = STRATEGIES.get(strategy_name)

    # validate input
    if strategy is None:
        raise KeyError(strategy_name + " is a not supported synthesis strategy")

    return strategy(**kwargs)
