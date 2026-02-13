import argparse
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
from cleantext import clean
from document_loader.format import Format
from document_loader.text_splitter import create_recursive_text_splitter
from entities.document import Document
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


def process_uploaded_file(uploaded_file, chunk_size: int = 512, chunk_overlap: int = 25) -> list[Document]:
    """
    Process an uploaded markdown file and split it into chunks.

    Args:
        uploaded_file: Streamlit uploaded file object
        chunk_size: Maximum size of each chunk
        chunk_overlap: Amount of overlap between chunks

    Returns:
        List of Document chunks
    """
    # Read the file content
    content = uploaded_file.read().decode("utf-8")

    # Create a Document object
    doc = Document(page_content=content, metadata={"source": uploaded_file.name})

    # Split into chunks
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents([doc])

    return chunks


def add_documents_to_index(index: Chroma, chunks: list[Document]) -> int:
    """
    Add document chunks to the vector database index.

    Args:
        index: Chroma vector database instance
        chunks: List of Document chunks to add

    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0

    texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    index.from_texts(texts=texts, metadatas=metadatas)

    return len(chunks)


def get_indexed_documents(index: Chroma) -> list[str]:
    """
    Get list of unique document sources in the index.

    Args:
        index: Chroma vector database instance

    Returns:
        List of unique source document names
    """
    try:
        # Get all items from collection
        results = index.collection.get()
        if results and "metadatas" in results:
            sources = set()
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
            return sorted(list(sources))
    except Exception as e:
        logger.warning(f"Could not retrieve indexed documents: {e}")
    return []


def handle_document_upload(index: Chroma, chunk_size: int = 512, chunk_overlap: int = 25):
    """
    Handle document upload UI in the sidebar.

    Args:
        index: Chroma vector database instance
        chunk_size: Maximum size of each chunk
        chunk_overlap: Amount of overlap between chunks
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Document Management")

    # Show currently indexed documents
    with st.sidebar.expander("📚 Indexed Documents"):
        indexed_docs = get_indexed_documents(index)
        if indexed_docs:
            for doc in indexed_docs:
                st.text(f"• {doc}")
        else:
            st.text("No documents indexed yet")

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Markdown files",
        type=["md", "markdown"],
        accept_multiple_files=True,
        help="Upload one or more Markdown files to add to the knowledge base",
    )

    if uploaded_files:
        if st.sidebar.button("📥 Add to Knowledge Base", key="add_docs"):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    try:
                        chunks = process_uploaded_file(uploaded_file, chunk_size, chunk_overlap)
                        num_chunks = add_documents_to_index(index, chunks)
                        total_chunks += num_chunks
                        logger.info(f"Added {num_chunks} chunks from {uploaded_file.name}")
                    except Exception as e:
                        st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        logger.error(f"Error processing {uploaded_file.name}: {str(e)}", exc_info=True)

                if total_chunks > 0:
                    st.sidebar.success(f"✅ Added {total_chunks} chunks from {len(uploaded_files)} file(s)")
                    # Force a rerun to refresh the indexed documents list
                    st.rerun()


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
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name = parameters.model
    synthesis_strategy_name = parameters.synthesis_strategy
    max_new_tokens = parameters.max_new_tokens

    init_page(root_folder)
    llm = load_llm_client(model_folder, model_name)
    chat_history = init_chat_history(2)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    index = load_index(vector_store_path)
    reset_chat_history(chat_history)

    # Handle document uploads
    handle_document_upload(index, chunk_size=parameters.chunk_size, chunk_overlap=parameters.chunk_overlap)

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

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="The maximum size of each chunk for document splitting.",
        required=False,
        default=512,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="The amount of overlap between consecutive chunks.",
        required=False,
        default=25,
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
