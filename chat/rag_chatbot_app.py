import argparse
import sys
import time
from pathlib import Path

import streamlit as st
from bot.client.client_settings import get_client, get_clients
from bot.client.llm_client import LlmClient
from bot.conversation.conversation_retrieval import ConversationRetrieval
from bot.conversation.ctx_strategy import (
    BaseSynthesisStrategy,
    get_ctx_synthesis_strategies,
    get_ctx_synthesis_strategy,
)
from bot.memory.embedder import EmbedderHuggingFace
from bot.memory.vector_memory import VectorMemory
from bot.model.model_settings import get_model_setting, get_models
from helpers.log import get_logger
from helpers.prettier import prettify_source

logger = get_logger(__name__)


@st.cache_resource()
def load_llm_client(llm_client_name: str, model_folder: Path, model_name: str) -> LlmClient:
    model_settings = get_model_setting(model_name)
    clients = [client.value for client in model_settings.clients]
    if llm_client_name not in clients:
        llm_client_name = clients[0]
    llm = get_client(llm_client_name, model_folder=model_folder, model_settings=model_settings)

    return llm


@st.cache_resource()
def load_conversational_retrieval(_llm: LlmClient) -> ConversationRetrieval:
    conversation_retrieval = ConversationRetrieval(_llm)
    return conversation_retrieval


@st.cache_resource()
def load_ctx_synthesis_strategy(ctx_synthesis_strategy_name: str, _llm: LlmClient) -> BaseSynthesisStrategy:
    ctx_synthesis_strategy = get_ctx_synthesis_strategy(ctx_synthesis_strategy_name, llm=_llm)
    return ctx_synthesis_strategy


@st.cache_resource()
def load_index(vector_store_path: Path) -> VectorMemory:
    """
    Loads a Vector Memory index based on the specified vector store path.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        VectorMemory: An instance of the VectorMemory class with the loaded index.
    """
    embedding = EmbedderHuggingFace().get_embedding()
    index = VectorMemory(vector_store_path=str(vector_store_path), embedding=embedding)

    return index


def init_page(root_folder: Path) -> None:
    """
    Initializes the page configuration for the application.
    """
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬", initial_sidebar_state="collapsed")
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(root_folder / "images/bot.png"), use_column_width="always")
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
    """
    Initializes a welcome message for the chat interface.
    """
    with st.chat_message("assistant"):
        st.write("How can I help you today?")


def init_messages() -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []


def display_messages_from_history():
    """
    Displays chat messages from the history on app rerun.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main(parameters) -> None:
    """
    Main function to run the RAG Chatbot application.

    Args:
        parameters: Parameters for the application.
    """
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    client_name = parameters.client
    model_name = parameters.model
    synthesis_strategy_name = parameters.synthesis_strategy

    init_page(root_folder)
    llm = load_llm_client(client_name, model_folder, model_name)
    conversational_retrieval = load_conversational_retrieval(_llm=llm)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    index = load_index(vector_store_path)
    init_messages()
    init_welcome_message()
    display_messages_from_history()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display retrieved documents with content previews, and updates the chat interface with the assistant's
        # responses.
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(
                text="Refining the question and Retrieving the docs – hang tight! " "This should take seconds."
            ):
                refined_user_input = conversational_retrieval.refine_question(user_input)
                retrieved_contents, sources = index.similarity_search(query=refined_user_input, k=parameters.k)
                if retrieved_contents:
                    full_response += "Here are the retrieved text chunks with a content preview: \n\n"
                    message_placeholder.markdown(full_response)

                    for source in sources:
                        full_response += prettify_source(source)
                        full_response += "\n\n"
                        message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    full_response += "I did not detect any pertinent chunk of text from the documents. \n\n"
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display assistant response in chat message container
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(
                text="Refining the context and Generating the answer for each text chunk – hang tight! "
                "This should take 1 minute."
            ):
                streamer, fmt_prompts = conversational_retrieval.context_aware_answer(
                    ctx_synthesis_strategy, user_input, retrieved_contents, return_generator=True
                )
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                conversational_retrieval.update_chat_history(user_input, full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chatbot")

    client_list = get_clients()
    default_client = client_list[0]

    model_list = get_models()
    default_model = model_list[0]

    synthesis_strategy_list = get_ctx_synthesis_strategies()
    default_synthesis_strategy = synthesis_strategy_list[0]

    parser.add_argument(
        "--client",
        type=str,
        choices=client_list,
        help=f"Client to be used. Defaults to {default_client}.",
        required=False,
        const=default_client,
        nargs="?",
        default=default_client,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    parser.add_argument(
        "--synthesis-strategy",
        type=str,
        choices=synthesis_strategy_list,
        help=f"Model to be used. Defaults to {default_synthesis_strategy}.",
        required=False,
        const=default_synthesis_strategy,
        nargs="?",
        default=default_synthesis_strategy,
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Number of chunks to return from the similarity search. Defaults to 2.",
        required=False,
        default=2,
    )

    return parser.parse_args()


# streamlit run rag_chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
