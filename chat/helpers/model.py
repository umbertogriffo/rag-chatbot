from enum import Enum

from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.llms import GPT4All


class Model(Enum):
    lora = 'ggml-model-q4_0.bin'
    wizard = 'ggml-wizardLM-7B.q4_2.bin'

    def __str__(self):
        return self.value


def load_gpt4all(model_path: str, n_threads: int = 4, streaming: bool = True, verbose: bool = True) -> GPT4All:
    """
    Loads the GPT4All model using the LangChain library.

    The LangChain library utilizes the `PyLLaMAcpp` module to load the converted `GPT4All` weights.

    Notes:
    - n_ctx=2048 is the maximum context window size for `gpt4all-lora-quantized-ggml.bin`.
      Refer to the following links for more information:
        - https://github.com/nomic-ai/gpt4all/issues/668#issuecomment-1556353537
        - https://github.com/hwchase17/langchain/issues/2404#issuecomment-1496615372

    Parameters:
    ----------
    model_path : str
        The path to the GPT4All model.

    n_threads : int, optional
        The number of threads to use (default is 4).

    streaming : bool
        Whether to stream the results or not. (default is True).
    verbose : bool
        Whether be verbose or not. (default is True).

    Returns:
    -------
    GPT4All
        The loaded GPT4All model.

    """
    # callbacks =[StreamingStdOutCallbackHandler()]
    callbacks = [FinalStreamingStdOutCallbackHandler(answer_prefix_tokens=["The", "answer", ":"])]
    llm = GPT4All(
        model=model_path,
        streaming=streaming,
        use_mlock=False,
        f16_kv=True,
        callbacks=callbacks,
        n_ctx=512,
        n_threads=n_threads,
        n_batch=1,
        n_predict=256,
        verbose=verbose,
    )
    return llm
