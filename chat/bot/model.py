import os
from pathlib import Path
from threading import Thread
from typing import Optional

import requests
from bot.model_settings import ModelSettings
from bot.prompt import generate_ctx_prompt, generate_qa_prompt, generate_refine_prompt
from ctransformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers import TextIteratorStreamer, TextStreamer


class Model:
    """
    This Model class encapsulates the initialization of the language model and tokenizer, as well as the generation of
    prompts and outputs.
    You can create an instance of this class and use its methods to handle the specific tasks you need.
    """

    streamer: Optional[TextIteratorStreamer] = None

    def __init__(self, model_folder: Path, model_settings: ModelSettings):
        self.model_settings = model_settings
        self.model_path = model_folder / self.model_settings.file_name
        self.system_template = self.model_settings.system_template
        self.qa_prompt_template = self.model_settings.qa_prompt_template
        self.ctx_prompt_template = self.model_settings.ctx_prompt_template
        self.refined_ctx_prompt_template = self.model_settings.refined_ctx_prompt_template

        self._auto_download()

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=str(model_folder),
            model_file=self.model_settings.file_name,
            model_type=self.model_settings.type,
            config=AutoConfig(config=self.model_settings.config),
            hf=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)

    def _auto_download(self) -> None:
        """
        Downloads a model file based on the provided name and saves it to the specified path.

        Returns:
            None

        Raises:
            Any exceptions raised during the download process will be caught and printed, but not re-raised.

        This function fetches model settings using the provided name, including the model's URL, and then downloads
        the model file from the URL. The download is done in chunks, and a progress bar is displayed to visualize
        the download process.

        """
        file_name = self.model_settings.file_name
        url = self.model_settings.url

        if not os.path.exists(self.model_path):
            # send a GET request to the URL to download the file.
            # Stream it while downloading, since the file is large

            try:
                response = requests.get(url, stream=True)
                # open the file in binary mode and write the contents of the response
                # in chunks.
                with open(self.model_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8912)):
                        if chunk:
                            f.write(chunk)

            except Exception as e:
                print(f"=> Download Failed. Error: {e}")
                return

            print(f"=> Model: {file_name} downloaded successfully 🥳")

    def generate_qa_prompt(self, question) -> str:
        """
        Generates a question-answering (QA) prompt using predefined templates.

        Args:
            question (str): The question for which the prompt is generated.

        Returns:
            str: The generated QA prompt.
        """
        return generate_qa_prompt(
            template=self.qa_prompt_template,
            system=self.system_template,
            question=question,
        )

    def generate_ctx_prompt(self, question, context) -> str:
        """
        Generates a context-based prompt using predefined templates.

        Args:
            question (str): The question for which the prompt is generated.
            context (str): The context information for the prompt.

        Returns:
            str: The generated context-based prompt.
        """
        return generate_ctx_prompt(
            template=self.ctx_prompt_template,
            system=self.system_template,
            question=question,
            context=context,
        )

    def generate_refined_ctx__prompt(self, question, context, existing_answer) -> str:
        """
        Generates a refined prompt for question-answering with existing answer.

        Args:
            question (str): The question for which the prompt is generated.
            context (str): The context information for the prompt.
            existing_answer (str): The existing answer to be refined.

        Returns:
            str: The generated refined prompt.
        """
        return generate_refine_prompt(
            template=self.refined_ctx_prompt_template,
            system=self.system_template,
            question=question,
            context=context,
            existing_answer=existing_answer,
        )

    def encode_prompt(self, prompt: str):
        """
        Encodes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be encoded.

        Returns:
            torch.Tensor: The input IDs tensor generated by the tokenizer.
        """
        return self.tokenizer(text=prompt, return_tensors="pt").input_ids

    def decode_answer(self, prompt_ids, answer_ids):
        """
        Decodes the answer IDs tensor into a human-readable answer using the model's tokenizer.

        Args:
            prompt_ids (torch.Tensor): The prompt IDs tensor used for generating the answer.
            answer_ids (torch.Tensor): The answer IDs tensor generated by the language model.

        Returns:
            str: The decoded answer.
        """
        return self.tokenizer.batch_decode(answer_ids[:, prompt_ids.shape[1]:])[0]

    def generate_answer(self, prompt: str, max_new_tokens: int = 1000) -> str:
        """
        Generates an answer based on the given prompt using the language model.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        Returns:
            str: The generated answer.
        """
        prompt_ids = self.encode_prompt(prompt)
        answer_ids = self.llm.generate(prompt_ids, max_new_tokens=max_new_tokens)
        answer = self.decode_answer(prompt_ids, answer_ids)

        return answer

    def stream_answer(
            self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 1000
    ) -> str:
        """
        Generates an answer by streaming tokens using the TextStreamer.

        Args:
            prompt (str): The input prompt for generating the answer.
            skip_prompt (bool): Whether to skip the prompt tokens during streaming (default is True).
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        Returns:
            str: The generated answer.
        """
        streamer = TextStreamer(tokenizer=self.tokenizer, skip_prompt=skip_prompt)

        prompt_ids = self.encode_prompt(prompt)
        answer_ids = self.llm.generate(
            prompt_ids, streamer=streamer, max_new_tokens=max_new_tokens
        )

        answer = self.decode_answer(prompt_ids, answer_ids)

        return answer

    def start_answer_iterator_streamer(
            self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 1000
    ) -> str:
        """
        Starts an answer iterator streamer thread for generating answers asynchronously.

        Args:
            prompt (str): The input prompt for generating the answer.
            skip_prompt (bool): Whether to skip the prompt tokens during streaming (default is True).
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        Returns:
            str: An empty string as a placeholder for the return value.
        """
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=skip_prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        kwargs = dict(
            input_ids=inputs["input_ids"],
            streamer=self.streamer,
            max_new_tokens=max_new_tokens,
        )
        thread = Thread(target=self.llm.generate, kwargs=kwargs)
        thread.start()

        return ""
