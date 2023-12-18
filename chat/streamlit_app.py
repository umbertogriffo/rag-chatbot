from pathlib import Path

import streamlit as st

from bot.model import Model
from bot.model_settings import get_model_setting
from helpers.log import get_logger

logger = get_logger(__name__)


@st.cache_resource(experimental_allow_widgets=True)
def load_llm(model_folder: Path) -> Model:
    """
    Create a LLM session object that points to the model.
    """
    model_settings = get_model_setting("zephyr")
    llm = Model(model_folder, model_settings)
    return llm


def init_page() -> None:
    st.set_page_config(
        page_title="Personal ChatGPT",
        page_icon="💬",
        initial_sidebar_state="collapsed"
    )
    st.header("Personal ChatGPT")
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
    _ = llm.start_answer_iterator_streamer(
        prompt, max_new_tokens=1000
    )
    for character in llm.streamer:
        yield character


def main() -> None:
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    init_page()
    llm = load_llm(model_folder)
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

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_answer(llm, user_input):
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# streamlit run app.py
if __name__ == "__main__":
    main()
