import asyncio
from typing import Any, Union

import nest_asyncio
from bot.client.llm_client import LlmClient
from helpers.log import get_logger

logger = get_logger(__name__)
nest_asyncio.apply()


class BaseSynthesisStrategy:
    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    def generate_response(self, retrieved_contents, question, max_new_tokens=512, return_generator=False):
        raise NotImplementedError("Subclasses must implement generate_response method")


class CreateAndRefineStrategy(BaseSynthesisStrategy):
    def __init__(self, llm: LlmClient):
        super().__init__(llm)

    def generate_response(
        self, retrieved_contents, question, max_new_tokens=512, return_generator=False
    ) -> Union[str, Any]:
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
        if num_of_contents > 0:
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
                    if return_generator:
                        cur_response = self.llm.start_answer_iterator_streamer(
                            fmt_prompt, max_new_tokens=max_new_tokens
                        )
                    else:
                        cur_response = self.llm.stream_answer(fmt_prompt, max_new_tokens=max_new_tokens)

                else:
                    cur_response = self.llm.generate_answer(fmt_prompt, max_new_tokens=max_new_tokens)
                    logger.debug(f"--- Current response: '{cur_response}' ... ---")
                fmt_prompts.append(fmt_prompt)
        else:
            fmt_prompt = self.llm.generate_qa_prompt(question=question)
            cur_response = self.llm.start_answer_iterator_streamer(fmt_prompt, max_new_tokens=max_new_tokens)
            fmt_prompts.append(fmt_prompt)

        return cur_response, fmt_prompts


class TreeSummarizationStrategy(BaseSynthesisStrategy):
    def __init__(self, llm: LlmClient):
        super().__init__(llm)

    def generate_response(
        self,
        retrieved_contents,
        question,
        max_new_tokens=512,
        num_children=2,
        return_generator=False,
    ) -> Union[str, Any]:
        """
        Generate a response using hierarchical summarization strategy.

        Combine `num_children` contents hierarchically until we get one root content.
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
            [],
            question,
            fmt_prompts,
            max_new_tokens=max_new_tokens,
            return_generator=return_generator,
            num_children=num_children,
        )

        return response, fmt_prompts

    def combine_results(
        self,
        texts,
        streams,
        question,
        cur_prompt_list,
        max_new_tokens=512,
        return_generator=False,
        num_children=2,
    ):
        new_texts = []
        new_streams = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx : idx + num_children]
            context = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            logger.info(f"--- Combining {len(texts)} responses ... ---")
            combined_response = self.llm.generate_answer(fmt_qa_prompt, max_new_tokens=max_new_tokens)
            combined_response_stream = self.llm.start_answer_iterator_streamer(
                fmt_qa_prompt, max_new_tokens=max_new_tokens
            )
            new_texts.append(str(combined_response))
            new_streams.append(combined_response_stream)
            cur_prompt_list.append(fmt_qa_prompt)

        if len(new_texts) == 1:
            if return_generator:
                return new_streams[0]
            else:
                return new_texts[0]
        else:
            return self.combine_results(
                new_texts,
                new_streams,
                question,
                cur_prompt_list,
                return_generator=return_generator,
                num_children=num_children,
            )


class AsyncTreeSummarizationStrategy(BaseSynthesisStrategy):
    def __init__(self, llm: LlmClient):
        super().__init__(llm)

    async def generate_response(
        self,
        retrieved_contents,
        question,
        max_new_tokens=512,
        num_children=2,
        return_generator=False,
    ) -> Union[str, Any]:
        """
        Generate a response using hierarchical summarization strategy.

        Combine `num_children` contents hierarchically until we get one root content.
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
            [],
            question,
            fmt_prompts,
            max_new_tokens=max_new_tokens,
            return_generator=return_generator,
            num_children=num_children,
        )

        return response, fmt_prompts

    async def combine_results(
        self,
        texts,
        streams,
        question,
        cur_prompt_list,
        max_new_tokens=512,
        return_generator=False,
        num_children=2,
    ):
        fmt_prompts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx : idx + num_children]
            context = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            fmt_prompts.append(fmt_qa_prompt)
            logger.info(f"--- Combining {len(texts)} responses ... ---")

        tasks = [self.llm.async_generate_answer(p, max_new_tokens=max_new_tokens) for p in fmt_prompts]
        combined_responses = await asyncio.gather(*tasks)

        tasks = [self.llm.async_start_answer_iterator_streamer(p, max_new_tokens=max_new_tokens) for p in fmt_prompts]
        combined_responses_stream = await asyncio.gather(*tasks)

        new_texts = [str(r) for r in combined_responses]
        new_streams = [r for r in combined_responses_stream]

        if len(new_texts) == 1:
            if return_generator:
                return new_streams[0]
            else:
                return new_texts[0]
        else:
            return await self.combine_results(
                new_texts,
                new_streams,
                question,
                cur_prompt_list,
                return_generator=return_generator,
                num_children=num_children,
            )


STRATEGIES = {
    "create_and_refine": CreateAndRefineStrategy,
    "tree_summarization": TreeSummarizationStrategy,
    "async_tree_summarization": AsyncTreeSummarizationStrategy,
}


def get_ctx_synthesis_strategies():
    return list(STRATEGIES.keys())


def get_ctx_synthesis_strategy(strategy_name: str, **kwargs):
    strategy = STRATEGIES.get(strategy_name)

    # validate input
    if strategy is None:
        raise KeyError(strategy_name + " is a not supported synthesis strategy")

    return strategy(**kwargs)
