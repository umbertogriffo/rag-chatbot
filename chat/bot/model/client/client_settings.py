from bot.model.client.client import LlmClient
from bot.model.client.ctransformers_client import CtransformersClient
from bot.model.client.lama_cpp_client import LamaCppClient

SUPPORTED_CLIENTS = {LlmClient.CTRANSFORMERS.value: CtransformersClient, LlmClient.LAMA_CPP.value: LamaCppClient}


def get_clients():
    """
    Returns a list of supported language model clients.

    Returns:
        list: A list of supported language model clients.
    """
    return list(SUPPORTED_CLIENTS.keys())


def get_client(client_name: str, **kwargs):
    """
    Retrieves a language model client based on the given client name.

    Args:
        client_name (str): The name of the language model client.

    Returns:
        Client: An instance of the requested language model client.

    Raises:
        KeyError: If the requested client is not supported.
    """
    client = SUPPORTED_CLIENTS.get(client_name)

    # validate input
    if client is None:
        raise KeyError(client_name + " is a not supported client")

    return client(**kwargs)
