# RAG(Retrieval-augmented generation) ChatBot with CTransformers, LangChain and Chroma

This project combines the power of [CTransformers](https://github.com/marella/ctransformers), [LangChain](https://python.langchain.com/docs/get_started/introduction.html) and 
[Chroma](https://github.com/chroma-core/chroma) to accomplish a specific task.
It works by taking a collection of Markdown files as input and, when asked a question, provides the corresponding answer
based on the context provided by those files.

![architecture.png](images/contextual-chatbot-gpt4all.png)

The `Memory Builder` component of the project loads Markdown pages from the `docs` folder.
It then divides these pages into smaller sections, calculates the embeddings (a numerical representation) of these
sections with the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 
`sentence-transformer`, and saves them in an embedding database called [Chroma](https://github.com/chroma-core/chroma) for later use.

When a user asks a question, the ChatBot retrieves the most relevant sections from the Embedding database.
These sections are then used as context to generate the final answer using a local language model (LLM).

Additionally, the chatbot is designed to remember previous interactions. It saves the chat history and considers the
relevant context from previous conversations to provide more accurate answers. 

> [!IMPORTANT]
> Disclaimer: The code has been tested on `Ubuntu 22.04.2 LTS` running on a Lenovo Legion 5 Pro
> with twenty `12th Gen Intel® Core™ i7-12700H` and an `NVIDIA GeForce RTX 3060`.
> If you are using another Operating System or different hardware, and you can't load the models, please
> take a look at the official CTransformers's GitHub [issue](https://github.com/marella/ctransformers/issues).

> [!WARNING]
> Note: it's important to note that the large language model sometimes generates hallucinations or false information.

## Table of contents

- [Prerequisites](#prerequisites)
  - [Install Poetry](#install-poetry)
- [🛠 Bootstrap Environment](#-bootstrap-environment)
  - [How to use the make file](#how-to-use-the-make-file)
- [Using the Open-Source GPT4All's Models Locally](#using-the-open-source-gpt4alls-models-locally)
  - [Supported Models](#supported-models)
- [Example Data](#example-data)
- [Build the memory index](#build-the-memory-index)
- [Run a simple Chatbot](#run-a-simple-chatbot)
- [Run the Contextual Chatbot](#run-the-contextual-chatbot)
- [References](#references)


## Prerequisites

Python 3.10+ and Poetry.

### Install Poetry

Install Poetry by following this [link](https://python-poetry.org/docs/).

## 🛠 Bootstrap Environment

To easily install the dependencies I created a make file.

### How to use the make file

> [!IMPORTANT]
> Run ```make install``` to install `sentence-transformers` with pip to avoid poetry's issues in installing torch 
> (it doesn't install CUDA dependencies).

* Check: ```make check```
  * Use It to check that `which pip3` and `which python3` points to the right path.
* Install: ```make install```
  * Creates an environment and installs all dependencies.
* Update: ```make update```
  * Update an environment and installs all updated dependencies.
* Tidy up the code: ```make tidy```
  * Run Isort, Black and Flake8.
* Clean: ```make clean```
  * Removes the environment and all cached files.

> Note: Run `Install` as your init command (or after `Clean`).

## Using the Open-Source GPT4All's Models Locally

We use [CTransformers](https://github.com/marella/ctransformers), an open-source library that allows working with 
transformer-based models efficiently.
Running the LLMs architecture on your local PC is impossible due to the large (~7 billion) number of 
parameters. The main contribution of `CTransformers` models is the ability to run them either on a `CPU` or `GPU`.
The authors applied `Quantization and 4-bit precision` using the [GGML/GGUF](https://medium.com/@phillipgimmi/what-is-gguf-and-ggml-e364834d241c) format.
Basically, the model uses fewer bits to represent the numbers.

### Supported Models
* [Zephyr 7B Beta - GGUF](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF)
* [Mistral 7B OpenOrca - GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)

## Example Data

You could download some Markdown pages from the [Blendle Employee Handbook](https://blendle.notion.site/Blendle-s-Employee-Handbook-7692ffe24f07450785f093b94bbe1a09) 
and put them under `docs`.

## Build the memory index

Run:
```shell
python chat/memory_builder.py --chunk-size 1000
```

## Run a simple Chatbot

Type:
```shell
python chat/app.py
```
![app.gif](images/app.gif)

## Run the RAG Chatbot

Run:
```shell
python chat/rag_bot.py --k 1
```

## References

* LLMs:
  * [Calculating GPU memory for serving LLMs](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm/)
  * [Building Response Synthesis from Scratch](https://gpt-index.readthedocs.io/en/latest/examples/low_level/response_synthesis.html#)
  * [Attention Sinks in LLMs for endless fluency](https://huggingface.co/blog/tomaarsen/attention-sinks)
* LLM integration and Modules:
  * [LangChain](https://python.langchain.com/docs/get_started/introduction.html):
    * [Memory](https://python.langchain.com/docs/modules/memory.html)
    * [MarkdownTextSplitter](https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#MarkdownTextSplitter)
    * [Chroma Integration](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma)
    * [GPT4All](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/gpt4all.html)
      * uses `pyllamacpp`
    * [Llama-cpp](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/llamacpp)
      * uses `llama-cpp-python`
    * [C Transformers](https://python.langchain.com/docs/integrations/llms/ctransformers.html)
* Embeddings:
  * [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    * This is a `sentence-transformers` model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
* Chatbot Development:
  * [How to Create a Custom Chatbot with Gradio Blocks](https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks#add-streaming-to-your-chatbot)
* Text Processing and Cleaning:
  * [clean-text](https://github.com/jfilter/clean-text/tree/main)
* Open Source Repositories:
  * [CTransformers](https://github.com/marella/ctransformers)
  * [GPT4All](https://github.com/nomic-ai/gpt4all)
  * [llama.cpp](https://github.com/ggerganov/llama.cpp)
  * [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp)
  * [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
  * [lit-gpt](https://github.com/Lightning-AI/lit-gpt)
  * [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
  * [PrivateDocBot](https://github.com/Abhi5h3k/PrivateDocBot)
