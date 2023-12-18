import argparse
import sys
from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.documents import Document

from bot.conversation.conversation import Conversation
from bot.memory.embedder import EmbedderHuggingFace
from bot.memory.vector_memory import VectorMemory
from bot.model import Model
from bot.model_settings import get_model_setting, get_models
from helpers.log import get_logger
from helpers.prettier import prettify_source

logger = get_logger(__name__)


@st.cache_resource()
def load_conversational_retrieval(model_folder: Path, model_name: str) -> Conversation:
    """
    Loads a Conversational Retrieval model based on the specified folder and model name.

    Args:
        model_folder (Path): The path to the folder containing the model.
        model_name (str): The name of the model.

    Returns:
        Conversation: An instance of the Conversation class with the loaded model.
    """
    model_settings = get_model_setting(model_name)
    llm = Model(model_folder, model_settings)

    conversation_retrieval = Conversation(llm)
    return conversation_retrieval


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


def init_page() -> None:
    """
     Initializes the page configuration for the application.
     """
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="💬",
        initial_sidebar_state="collapsed"
    )
    st.header("RAG Chatbot")
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


def get_answer(conversational_retrieval: Conversation, question: str, retrieved_contents: List[Document]) -> str:
    """
    Retrieves the answer from the Conversational Retrieval model for the given question and retrieved contents.

    Args:
        conversational_retrieval (Conversation): An instance of the Conversational Retrieval model.
        question (str): The user's question.
        retrieved_contents: Retrieved contents relevant to the user's question.

    Yields:
        str: A character of the generated answer.
    """
    streamer = conversational_retrieval.answer(question, retrieved_contents, return_generator=True)
    for character in streamer:
        yield character


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

    init_page()
    conversational_retrieval = load_conversational_retrieval(model_folder, parameters.model)
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
            with st.spinner(text="Refining the question and Retrieving the docs – hang tight! "
                                 "This should take seconds."
                            ):
                user_input = conversational_retrieval.refine_question(user_input)
                retrieved_contents, sources = index.similarity_search(query=user_input, k=parameters.k)

                full_response += "Here are the retrieved text chunks with a content preview: \n\n"
                message_placeholder.markdown(full_response)

                for source in sources:
                    full_response += prettify_source(source)
                    full_response += "\n\n"
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(
                    text="Refining the context and Generating the answer for each text chunk – hang tight! "
                         "This should take 1 minute."
            ):
                for chunk in get_answer(conversational_retrieval, user_input, retrieved_contents):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                conversational_retrieval.update_chat_history(user_input, full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chatbot")

    model_list = get_models()
    default_model = model_list[0]

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
