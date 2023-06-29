from langchain.callbacks import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All


def load_gpt4all(model_path: str, n_threads: int = 4, streaming: bool = True, verbose: bool = True) -> GPT4All:
    """
    Loads the GPT4All model using the LangChain library.

    The LangChain library utilizes the `PyLLaMAcpp` module to load the converted `GPT4All` weights.

    Notes:
    - n_ctx=2048 is the maximum context window size for gpt4all-lora-quantized-ggml.bin.
      Refer to the following links for more information:
        - https://github.com/nomic-ai/gpt4all/issues/668#issuecomment-1556353537
        - https://github.com/hwchase17/langchain/issues/2404#issuecomment-1496615372

    - Some suggestions from here https://github.com/abetlen/llama-cpp-python/issues/19#issuecomment-1496254400:
        - n_threads: Anywhere between cpu_count / 2 to cpu_count should be ideal.
        - use_mlock: True
        - f16_cache: True
        - n_batch: Set the max value of this to your n_ctx this effects how many tokens get eval'd at once.

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
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = GPT4All(
        model=model_path,
        streaming=streaming,
        use_mlock=False,
        f16_kv=True,
        callback_manager=callback_manager,
        n_ctx=512,
        n_threads=n_threads,
        n_batch=1,
        n_predict=256,
        verbose=verbose,
    )
    return llm
