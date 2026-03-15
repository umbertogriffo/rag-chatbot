```python
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
            context = "\n\n".join(list(text_batch))
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
```
