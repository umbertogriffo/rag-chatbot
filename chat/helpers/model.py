import os

from langchain.callbacks import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All


def load_gpt4all(model_path: str, n_threads: int = int(os.cpu_count() - 1)) -> GPT4All:
    """

    The [LangChain](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/gpt4all) library
    uses `PyLLaMAcpp` module to load the converted `GPT4All` weights.

    Parameters
    ----------
    model_path
    n_threads

    Returns
    -------

    """
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = GPT4All(
        model=model_path,
        streaming=True,
        callback_manager=callback_manager,
        n_ctx=2048,
        n_threads=n_threads,
        n_predict=1024,
        verbose=True,
    )
    return llm
