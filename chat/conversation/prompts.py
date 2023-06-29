# flake8: noqa
from langchain import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the Engineering Handbook.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


template = """You are an AI assistant for answering questions about the Engineering Handbook.
You are given a question and the following context containing the extracted parts of a long document. Provide a conversational answer.
If you don't know the answer, say, "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the Engineering Handbook, please let them know you are tuned to only answer questions about the Engineering Handbook.

Question: {question}
=========
Context: {context}
=========
Answer:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

summarization_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
SUMMARIZATION_PROMPT = PromptTemplate(
    template=summarization_template, input_variables=["text"]
)
