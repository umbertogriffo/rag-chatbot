import argparse
import sys
from pathlib import Path

import streamlit as st

from bot.conversation.conversation_retrieval import ConversationRetrieval
from bot.model.client.client import Client
from bot.model.client.client_settings import get_client, get_clients
from bot.model.model_settings import get_model_setting, get_models
from helpers.log import get_logger

logger = get_logger(__name__)


@st.cache_resource(experimental_allow_widgets=True)
def load_llm(llm_client: Client, model_name: str, model_folder: Path) -> Client:
    """
    Create a LLM session object that points to the model.
    """
    model_settings = get_model_setting(model_name)
    clients = [client.value for client in model_settings.clients]
    if llm_client not in clients:
        llm_client = clients[0]
    llm = get_client(llm_client, model_folder=model_folder, model_settings=model_settings)
    return llm


@st.cache_resource()
def load_conversational_retrieval(_llm: Client) -> ConversationRetrieval:
    conversation_retrieval = ConversationRetrieval(_llm)
    return conversation_retrieval


def init_page() -> None:
    st.set_page_config(
        page_title="Chatbot",
        page_icon="💬",
        initial_sidebar_state="collapsed"
    )
    st.header("Chatbot")
    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
    with st.chat_message("assistant"):
        st.write("How can I help you today?")


def init_messages() -> None:
    """
    Initialize chat history.
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []


def display_messages_from_history():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_answer(llm, messages) -> tuple[str, float]:
    prompt = llm.generate_qa_prompt(question=messages)
    streamer = llm.start_answer_iterator_streamer(
        prompt, max_new_tokens=1000
    )
    for character in streamer:
        yield llm.parse_token(character)


def main(parameters) -> None:
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    client = parameters.client
    model = parameters.model

    init_page()
    llm = load_llm(client, model, model_folder)
    conversational_retrieval = load_conversational_retrieval(_llm=llm)
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

        with st.spinner(text="Refining the question – hang tight! "
                             "This should take seconds."
                        ):
            user_input = conversational_retrieval.refine_question(user_input)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_answer(llm, user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        conversational_retrieval.update_chat_history(user_input, full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot")

    client_list = get_clients()
    default_client = client_list[0]

    model_list = get_models()
    default_model = model_list[0]

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

    return parser.parse_args()


# streamlit run chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
