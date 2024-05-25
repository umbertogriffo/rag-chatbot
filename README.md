# RAG (Retrieval-augmented generation) ChatBot

[![CI](https://github.com/umbertogriffo/rag-chatbot/workflows/CI/badge.svg)](https://github.com/umbertogriffo/rag-chatbot/actions/workflows/ci.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> [!IMPORTANT]
> Disclaimer:
> The code has been tested on:
>   * `Ubuntu 22.04.2 LTS` running on a Lenovo Legion 5 Pro with twenty `12th Gen Intel® Core™ i7-12700H` and
      an `NVIDIA GeForce RTX 3060`.
>   * `MacOS Sonoma 14.3.1` running on a MacBook Pro M1 (2020).
>
> If you are using another Operating System or different hardware, and you can't load the models, please
> take a look either at the official Llama Cpp Python's
> GitHub [issue](https://github.com/abetlen/llama-cpp-python/issues).
> or at the official CTransformers's GitHub [issue](https://github.com/marella/ctransformers/issues)

> [!WARNING]
> Note: it's important to note that the large language model sometimes generates hallucinations or false information.

## Table of contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
    - [Install Poetry](#install-poetry)
- [Bootstrap Environment](#bootstrap-environment)
    - [How to use the make file](#how-to-use-the-make-file)
- [Using the Open-Source Models Locally](#using-the-open-source-models-locally)
    - [Supported Models](#supported-models)
- [Example Data](#example-data)
- [Build the memory index](#build-the-memory-index)
- [Run the Chatbot](#run-the-chatbot)
- [Run the RAG Chatbot](#run-the-rag-chatbot)
- [How to debug the Streamlit app on Pycharm](#how-to-debug-the-streamlit-app-on-pycharm)
- [References](#references)

## Introduction

This project combines the power
of [Lama.cpp](https://github.com/abetlen/llama-cpp-python), [CTransformers](https://github.com/marella/ctransformers),
[LangChain](https://python.langchain.com/docs/get_started/introduction.html) (only used for document chunking and
querying the Vector Database, and we plan to eliminate it entirely),
[Chroma](https://github.com/chroma-core/chroma) and [Streamlit](https://discuss.streamlit.io/) to build:

* a Conversation-aware Chatbot (ChatGPT like experience).
* a RAG (Retrieval-augmented generation) ChatBot.

The RAG Chatbot works by taking a collection of Markdown files as input and, when asked a question, provides the
corresponding answer
based on the context provided by those files.

![rag-chatbot-architecture-1.png](images/rag-chatbot-architecture-1.png)

The `Memory Builder` component of the project loads Markdown pages from the `docs` folder.
It then divides these pages into smaller sections, calculates the embeddings (a numerical representation) of these
sections with the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
`sentence-transformer`, and saves them in an embedding database called [Chroma](https://github.com/chroma-core/chroma)
for later use.

When a user asks a question, the RAG ChatBot retrieves the most relevant sections from the Embedding database.
Since the original question can't be always optimal to retrieve for the LLM, we first prompt an LLM to rewrite the
question,
then conduct retrieval-augmented reading.
The most relevant sections are then used as context to generate the final answer using a local language model (LLM).
Additionally, the chatbot is designed to remember previous interactions. It saves the chat history and considers the
relevant context from previous conversations to provide more accurate answers.

To deal with context overflows, we implemented two approaches:

* `Create And Refine the Context`: synthesize a responses sequentially through all retrieved contents.
    * ![create-and-refine-the-context.png](images/create-and-refine-the-context.png)
* `Hierarchical Summarization of Context`: generate an answer for each relevant section independently, and then
  hierarchically combine the answers.
    * ![hierarchical-summarization.png](images/hierarchical-summarization.png)

## Prerequisites

* Python 3.10+
* GPU supporting CUDA 12 and up
* Poetry 1.7.0

### Install Poetry

Install Poetry with the official installer by following
this [link](https://python-poetry.org/docs/#installing-with-the-official-installer).

You must use the current adopted version of Poetry
defined [here](https://github.com/umbertogriffo/rag-chatbot/blob/main/version/poetry).

If you have poetry already installed and is not the right version, you can downgrade (or upgrade) poetry through:

```
poetry self update <version>
```

## Bootstrap Environment

To easily install the dependencies we created a make file.

### How to use the make file

> [!IMPORTANT]
> Run `Setup` as your init command (or after `Clean`).

* Check: ```make check```
    * Use it to check that `which pip3` and `which python3` points to the right path.
* Setup:
    * Setup with NVIDIA CUDA acceleration: ```make setup_cuda```
        * Creates an environment and installs all dependencies with NVIDIA CUDA acceleration.
    * Setup with Metal GPU acceleration: ```make setup_metal```
        * Creates an environment and installs all dependencies with Metal GPU acceleration for macOS system only.
* Update: ```make update```
    * Update an environment and installs all updated dependencies.
* Tidy up the code: ```make tidy```
    * Run Ruff check and format.
* Clean: ```make clean```
    * Removes the environment and all cached files.
* Test: ```make test```
    * Runs all tests.
    * Using [pytest](https://pypi.org/project/pytest/)

## Using the Open-Source Models Locally

We utilize two open-source libraries, [Lama.cpp](https://github.com/abetlen/llama-cpp-python)
and [CTransformers](https://github.com/marella/ctransformers),
which allow us to work efficiently with transformer-based models efficiently.
Running the LLMs architecture on a local PC is impossible due to the large (~7 billion) number of
parameters. These libraries enable us to run them either on a `CPU` or `GPU`.
Additionally, we use the `Quantization and 4-bit precision` to reduce number of bits required to represent the numbers.
The quantized models are stored in [GGML/GGUF](https://medium.com/@phillipgimmi/what-is-gguf-and-ggml-e364834d241c)
format.

### Supported Models

| 🤖 Model                                       | Supported | Model Size | Notes and link to the model                                                                                                                                          |
|------------------------------------------------|-----------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `llama-3` Meta Llama 3 Instruct                | ✅         | 8B         | Less accurate than OpenChat - [link](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)                                                                 |
| `openchat` **Recommended** - OpenChat 3.5 0106 | ✅         | 7B         | [link](https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF)                                                                                                       |
| `starling` Starling Beta                       | ✅         | 7B         | Is trained from `Openchat-3.5-0106`. It's recommended if you prefer more verbosity over OpenChat - [link](https://huggingface.co/bartowski/Starling-LM-7B-beta-GGUF) |
| `neural-beagle` NeuralBeagle14                 | ✅         | 7B         | [link](https://huggingface.co/TheBloke/NeuralBeagle14-7B-GGUF)                                                                                                       |
| `dolphin` Dolphin 2.6 Mistral DPO Laser        | ✅         | 7B         | [link](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF)                                                                                        |
| `zephyr` Zephyr Beta                           | ✅         | 7B         | [link](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF)                                                                                                          |
| `mistral` Mistral OpenOrca                     | ✅         | 7B         | [link](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)                                                                                                     |
| `phi-3` Phi-3 Mini 4K Instruct                 | ✅         | 3.8B       | [link](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)                                                                                                 |
| `stablelm-zephyr` StableLM Zephyr OpenOrca     | ✅         | 3B         | [link](https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF)                                                                                                      |

## Example Data

You could download some Markdown pages from
the [Blendle Employee Handbook](https://blendle.notion.site/Blendle-s-Employee-Handbook-7692ffe24f07450785f093b94bbe1a09)
and put them under `docs`.

## Build the memory index

Run:

```shell
python chat/memory_builder.py --chunk-size 1000
```

## Run the Chatbot

To interact with a GUI type:

```shell
streamlit run chatbot/chatbot_app.py -- --model openchat
```

![conversation-aware-chatbot.gif](images/conversation-aware-chatbot.gif)

## Run the RAG Chatbot

To interact with a GUI type:

```shell
streamlit run chatbot/rag_chatbot_app.py -- --model openchat --k 2 --synthesis-strategy async_tree_summarization
```

![rag_chatbot_example.gif](images%2Frag_chatbot_example.gif)

## How to debug the Streamlit app on Pycharm

![debug_streamlit.png](images/debug_streamlit.png)

## References

* LLMs:
    * [Calculating GPU memory for serving LLMs](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm/)
    * [Building Response Synthesis from Scratch](https://gpt-index.readthedocs.io/en/latest/examples/low_level/response_synthesis.html#)
    * [Attention Sinks in LLMs for endless fluency](https://huggingface.co/blog/tomaarsen/attention-sinks)
    * [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
    * [Introduction to Weight Quantization](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)
* LLM integration and Modules:
    * [LangChain](https://python.langchain.com/docs/get_started/introduction.html):
        * [MarkdownTextSplitter](https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#MarkdownTextSplitter)
        * [Chroma Integration](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma)
        * [The Problem With LangChain](https://minimaxir.com/2023/07/langchain-problem/#:~:text=The%20problem%20with%20LangChain%20is,don't%20start%20with%20LangChain)
* Embeddings:
    * [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
        * This is a `sentence-transformers` model: It maps sentences & paragraphs to a 384 dimensional dense vector
          space and can be used for tasks like clustering or semantic search.
* Vector Databases:
    * [Chroma](https://www.trychroma.com/)
    * [Food Discovery with Qdrant](https://qdrant.tech/articles/new-recommendation-api/#)
    * Indexing algorithms:
        * There are many algorithms for building indexes to optimize vector search. Most vector databases
          implement `Hierarchical Navigable Small World (HNSW)` and/or `Inverted File Index (IVF)`. Here are some great
          articles explaining them, and the trade-off between `speed`, `memory` and `quality`:
            * [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/)
            * [Hierarchical Navigable Small World (HNSW)](https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37)
            * [From NVIDIA - Accelerating Vector Search: Using GPU-Powered Indexes with RAPIDS RAFT](https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/)
            * [From NVIDIA - Accelerating Vector Search: Fine-Tuning GPU Index Algorithms](https://developer.nvidia.com/blog/accelerating-vector-search-fine-tuning-gpu-index-algorithms/)
            * > PS: Flat indexes (i.e. no optimisation) can be used to maintain 100% recall and precision, at the
              expense of speed.
* Retrieval Augmented Generation (RAG):
    * [Rewrite-Retrieve-Read](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb)
        * > Because the original query can not be always optimal to retrieve for the LLM, especially in the real world,
          we first prompt an LLM to rewrite the queries, then conduct retrieval-augmented reading.
    * [Rerank](https://txt.cohere.com/rag-chatbot/#implement-reranking)
    * [Conversational awareness](https://langstream.ai/2023/10/13/rag-chatbot-with-conversation/)
    * [Summarization: Improving RAG quality in LLM apps while minimizing vector storage costs](https://www.ninetack.io/post/improving-rag-quality-by-summarization)
* Chatbot Development:
    * [Streamlit](https://discuss.streamlit.io/):
        * [Build a basic LLM chat app](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-chatgpt-like-app)
        * [Layouts and Containers](https://docs.streamlit.io/library/api-reference/layout)
        * [st.chat_message](https://docs.streamlit.io/library/api-reference/chat/st.chat_message)
        * [Add statefulness to apps](https://docs.streamlit.io/library/advanced-features/session-state)
            * [Why session state is not persisting between refresh?](https://discuss.streamlit.io/t/why-session-state-is-not-persisting-between-refresh/32020)
        * [st.cache_resource](https://docs.streamlit.io/library/api-reference/performance/st.cache_resource)
        * [Handling External Command Line Arguments](https://github.com/streamlit/streamlit/issues/337)
    * [(Investigate) FastServe - Serve Llama-cpp with FastAPI](https://github.com/aniketmaurya/fastserve)
    * [(Investigate) Chat Templates to standardise the format](https://huggingface.co/blog/chat-templates)
    * [(Investigate) Ollama](https://github.com/ollama/ollama)
* Text Processing and Cleaning:
    * [clean-text](https://github.com/jfilter/clean-text/tree/main)
* Open Source Repositories:
    * [CTransformers](https://github.com/marella/ctransformers)
    * [GPT4All](https://github.com/nomic-ai/gpt4all)
    * [llama.cpp](https://github.com/ggerganov/llama.cpp)
    * [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
    * [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp)
    * [chroma](https://github.com/chroma-core/chroma)
    * Inspirational repos:
        * [lit-gpt](https://github.com/Lightning-AI/lit-gpt)
        * [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
        * [PrivateDocBot](https://github.com/Abhi5h3k/PrivateDocBot)
        * [Rag_bot - Adaptive Intelligence Chatbot](https://github.com/kylejtobin/rag_bot)
