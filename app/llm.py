import os
import tempfile
import time
from typing import Generator, cast

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.graph.graph import CompiledGraph
from loguru import logger
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from util import Config, get_session

RESPONSE_GEN_T = Generator[str, None, None]


def load_chat_llm() -> AzureChatOpenAI:
    return cast(AzureChatOpenAI, st.session_state["llm"])


def load_embedding_llm() -> AzureOpenAIEmbeddings:
    return cast(AzureOpenAIEmbeddings, st.session_state["embedding_llm"])


def load_react_agent() -> CompiledGraph:
    return cast(CompiledGraph, st.session_state["agent"])


def chroma_client():
    return cast(Chroma, st.session_state["chroma_client"])


def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.Char_Text_Splitter_Chunk_Size,
        chunk_overlap=Config.Char_Text_Splitter_Chunk_Overlap,
        length_function=load_chat_llm().get_num_tokens,
        is_separator_regex=False,
    )


def _load_txt(session_id: str, file: UploadedFile) -> list[Document]:
    splitter = text_splitter()
    metadata = {"session_id": session_id, "file_name": file.name}
    texts = splitter.split_text(file.getvalue().decode("utf-8"))
    return splitter.create_documents(texts, metadatas=[metadata] * len(texts))  # type: ignore


def _extend_docs_metadata(docs: list[Document], metadata_extension: dict[str, str]):
    for doc in docs:
        doc.metadata.update(**metadata_extension)  # type: ignore


def _load_pdf(session_id: str, file: UploadedFile) -> list[Document]:
    splitter = text_splitter()

    file_name = None
    try:
        with tempfile.NamedTemporaryFile("w+b", delete=False) as temp_file:
            temp_file.write(file.getvalue())
            file_name = temp_file.name
        docs = PyPDFLoader(temp_file.name).load_and_split(splitter)
        _extend_docs_metadata(docs, {"session_id": session_id})
        return docs
    finally:
        if file_name is not None:
            os.remove(file_name)


def _load_md(session_id: str, file: UploadedFile) -> list[Document]:
    headers_to_split_on: list[tuple[str, str]] = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    docs = splitter.split_text(file.getvalue().decode("utf-8"))
    _extend_docs_metadata(docs, {"session_id": session_id})
    return docs


@logger.catch
def embed_file(session_id: str, file: UploadedFile):
    assert "." in file.name, ""
    match file.name.split(".")[-1].lower():
        case "txt":
            documents = _load_txt(session_id, file)
        case "pdf":
            documents = _load_pdf(session_id, file)
        case "md":
            documents = _load_md(session_id, file)
        case _:
            raise ValueError(f"File type {file.type} is not supported")
    chroma_client().add_documents(documents)


def format_docs(docs: list[Document]):
    return f"{'='*15}\n\n".join(f"{doc.metadata}\n{doc.page_content}" for doc in docs)  # type: ignore


@logger.catch
def query_chroma(query: str) -> str:
    """Use this function for debugging chroma"""
    return format_docs(
        chroma_client()
        .as_retriever(
            search_type="similarity",
            score_threshold=Config.Chroma_Search_Min_Relevance_Thr,
            search_kwargs=dict(k=Config.Chroma_Search_Top_K, filter={"session_id": get_session()}),
        )
        .invoke(query)
    )


@logger.catch
def infer_response_stream(query: str) -> RESPONSE_GEN_T:
    # IF WE WANT TO SEE STEP BY STEP HOW IT EXECUTES WE CAN USE BELOW CODE
    # WE CAN SEE WHEN IT CALLS RAG AND WHEN ANSWERS WITHOUT IT

    # from langchain_core.messages import HumanMessage
    # for s in load_react_agent().stream(
    #     {"messages": [HumanMessage(content=query)]} , config={"configurable": {"thread_id": get_session()}}
    # ):
    #     print(s)
    #     print("----")
    config = {"configurable": {"thread_id": get_session()}}
    response = load_react_agent().invoke({"messages": [query]}, config=config)
    return_text = response["messages"][-1].content
    yield from echo_response_stream(
        return_text, timeout=0
    )  # TODO: Needs to find way for streaming, maybe implement custom callback


@logger.catch
def echo_response_stream(query: str, timeout: float = 0.05) -> RESPONSE_GEN_T:
    for letter in query:
        yield letter
        time.sleep(timeout)
