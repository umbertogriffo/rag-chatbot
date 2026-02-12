import argparse
import hashlib
import sys
import time
from pathlib import Path

import streamlit as st
from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, extract_content_after_reasoning, refine_question
from bot.conversation.ctx_strategy import (
    BaseSynthesisStrategy,
    get_ctx_synthesis_strategies,
    get_ctx_synthesis_strategy,
)
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import get_model_settings, get_models
from document_loader.format import Format
from document_loader.loader import DirectoryLoader
from document_loader.text_splitter import create_recursive_text_splitter
from helpers.log import get_logger
from helpers.prettier import prettify_source

logger = get_logger(__name__)

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", initial_sidebar_state="collapsed")


@st.cache_resource()
def load_llm_client(model_folder: Path, model_name: str) -> LamaCppClient:
    model_settings = get_model_settings(model_name)
    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)

    return llm


@st.cache_resource()
def init_chat_history(total_length: int = 2) -> ChatHistory:
    chat_history = ChatHistory(total_length=total_length)
    return chat_history


@st.cache_resource()
def load_ctx_synthesis_strategy(ctx_synthesis_strategy_name: str, _llm: LamaCppClient) -> BaseSynthesisStrategy:
    ctx_synthesis_strategy = get_ctx_synthesis_strategy(ctx_synthesis_strategy_name, llm=_llm)
    return ctx_synthesis_strategy


@st.cache_resource()
def load_index(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        Chroma: An instance of the Vector Database.
    """
    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)

    return index


def get_most_recently_modified_document(docs_path: Path) -> Path | None:
    """
    Returns the most recently modified Markdown document in the given directory.

    Args:
        docs_path (Path): The directory containing Markdown documents.

    Returns:
        Path | None: The newest Markdown document, or None if none exist.
    """
    documents = (doc for doc in docs_path.glob("**/*.md") if doc.is_file())
    return max(documents, key=lambda doc: doc.stat().st_mtime, default=None)


def save_and_replace_uploaded_document(docs_path: Path, uploaded_file) -> Path:
    """
    Saves an uploaded Markdown document and removes previously uploaded documents.

    Args:
        docs_path (Path): The directory containing Markdown documents.
        uploaded_file: The uploaded Markdown file object.

    Returns:
        Path: The saved Markdown file path.
    """
    docs_path.mkdir(parents=True, exist_ok=True)
    existing_docs = list(docs_path.glob("**/*.md"))
    if existing_docs:
        logger.info("Replacing uploaded document; clearing %d existing document(s).", len(existing_docs))
    for existing_doc in existing_docs:
        existing_doc.unlink()
    target_path = docs_path / uploaded_file.name
    target_path.write_bytes(uploaded_file.getvalue())
    return target_path


def build_index_from_docs(docs_path: Path, vector_store_path: Path) -> Chroma:
    """
    Rebuilds the Chroma index from Markdown documents in the provided directory.
    This scans the directory after replacement to ensure only the active document is indexed.

    Args:
        docs_path (Path): The directory containing Markdown documents.
        vector_store_path (Path): The directory for the vector store.

    Returns:
        Chroma: The rebuilt vector store index.
    """
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.md",
        show_progress=False,
    )
    sources = loader.load()
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value, chunk_size=512, chunk_overlap=25
    )
    chunks = splitter.split_documents(sources)
    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
    index.reset_collection()
    if chunks:
        index.from_chunks(chunks)
    return index


def get_file_signature(uploaded_file) -> str:
    """
    Computes a SHA-256 hash of the uploaded file to detect content changes.

    Args:
        uploaded_file: The uploaded Markdown file object.

    Returns:
        str: A SHA-256 hex digest of the file contents.
    """
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def init_page(root_folder: Path) -> None:
    """
    Initializes the page configuration for the application.
    """
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


def reset_chat_history(chat_history: ChatHistory) -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        chat_history.clear()


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
    docs_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name = parameters.model
    synthesis_strategy_name = parameters.synthesis_strategy
    max_new_tokens = parameters.max_new_tokens

    init_page(root_folder)
    llm = load_llm_client(model_folder, model_name)
    chat_history = init_chat_history(2)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    reset_chat_history(chat_history)

    uploaded_file = st.sidebar.file_uploader(
        "Upload a document",
        type=["md"],
        help="Upload a Markdown document to replace the current context.",
    )
    if uploaded_file:
        signature = get_file_signature(uploaded_file)
        if signature != st.session_state.get("active_document_signature"):
            with st.spinner(text="Processing the uploaded document..."):
                saved_path = save_and_replace_uploaded_document(docs_path, uploaded_file)
                st.session_state.active_index = build_index_from_docs(docs_path, vector_store_path)
            st.session_state.active_document = saved_path.name
            st.session_state.active_document_signature = signature
            st.session_state.messages = []
            chat_history.clear()

    active_document = st.session_state.get("active_document")
    if not active_document:
        latest_document = get_most_recently_modified_document(docs_path)
        if latest_document:
            active_document = latest_document.name
            st.session_state.active_document = active_document

    if active_document:
        st.sidebar.markdown(f"**Active document:** {active_document}")
    else:
        st.sidebar.info("No document uploaded yet.")

    index = st.session_state.get("active_index") or load_index(vector_store_path)
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
                text="Refining the question and Retrieving the docs – hang tight! This should take seconds."
            ):
                refined_user_input = refine_question(
                    llm, user_input, chat_history=chat_history, max_new_tokens=max_new_tokens
                )
                retrieved_contents, sources = index.similarity_search_with_threshold(
                    query=refined_user_input, k=parameters.k
                )
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
            with st.spinner(text="Refining the context and Generating the answer for each text chunk – hang tight! "):
                streamer, _ = answer_with_context(
                    llm, ctx_synthesis_strategy, user_input, chat_history, retrieved_contents, max_new_tokens
                )
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "▌")

                if llm.model_settings.reasoning:
                    answer = extract_content_after_reasoning(full_response, llm.model_settings.reasoning_stop_tag)
                    if answer == "":
                        answer = "I wasn't able to provide the answer; Do you want me to try again?"
                else:
                    answer = full_response

                chat_history.append(f"question: {user_input}, answer: {answer}")

                message_placeholder.markdown(answer)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chatbot")

    model_list = get_models()
    default_model = model_list[0]

    synthesis_strategy_list = get_ctx_synthesis_strategies()
    default_synthesis_strategy = synthesis_strategy_list[0]

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

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of tokens to generate in the answer. Defaults to 512.",
        required=False,
        default=512,
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
