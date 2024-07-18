"""
Streamlit App
"""
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from llm import embed_file, infer_response_stream, query_chroma
import streamlit as st
from util import Config, get_session, start_beating

load_dotenv()
st.set_page_config(page_title="LLM App Example", page_icon="🦜")
st.title("LLM App Example")


@st.cache_resource
def load_chat_llm():
    return AzureChatOpenAI(temperature=0.7, azure_deployment="gpt-4-0613")


@st.cache_resource
def load_embedding_llm():
    match Config.Embedding_Model:
        case "ada":
            return AzureOpenAIEmbeddings(model="text-embedding-ada-002")
        case _:
            from langchain_community.embeddings.sentence_transformer import (
                SentenceTransformerEmbeddings,
            )

            return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def chroma_client():
    client = Chroma(embedding_function=load_embedding_llm())
    return client


def load_tools() -> list[Tool]:
    tool = create_retriever_tool(
        chroma_client().as_retriever(
            search_type="similarity",
            score_threshold=Config.Chroma_Search_Min_Relevance_Thr,
            search_kwargs=dict(k=Config.Chroma_Search_Top_K, filter={"session_id": get_session()}),
        ),
        "document_retreiver",
        "Retreives documents from Chroma if needed",
    )
    return [tool]


def load_memory() -> SqliteSaver:
    memory = SqliteSaver.from_conn_string(":memory:")
    return memory


if "llm" not in st.session_state:
    st.session_state["llm"] = load_chat_llm()

if "memory" not in st.session_state:
    st.session_state["memory"] = load_memory()

if "tools" not in st.session_state:
    st.session_state["tools"] = load_tools()

if "embedding_llm" not in st.session_state:
    st.session_state["embedding_llm"] = load_embedding_llm()

if "agent" not in st.session_state:
    ### Build retriever tool ###
    tools = load_tools()
    agent_executor = create_react_agent(
        load_chat_llm(),
        tools,
        checkpointer=st.session_state["memory"],
        messages_modifier=SystemMessage(
            content=(
                "You're a helpful chat assistant. You help users find answers based on "
                "documents you receive using document_retreiver tool "
                # Ask it to use only information from the context
                "Your answers must contain facts solely based on the "
                "context of this conversation. "
                # Adding an option for where there is no information
                # helps to prevent some halucinations
                "If provided context is irrelevant to user's question reply \"Sorry, I can't answer this question.\" "
                "and stop."
            )
        ),
    )
    st.session_state["agent"] = agent_executor

if "chroma_client" not in st.session_state:
    st.session_state["chroma_client"] = chroma_client()

if "heart_beat" not in st.session_state:
    st.session_state["heart_beat"] = True
    start_beating(get_session())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    if prompt.startswith("chroma:"):
        with st.chat_message("ai"):
            response = query_chroma(str(prompt))
            st.write(response)
    else:
        with st.chat_message("ai"):
            response = st.write_stream(infer_response_stream(str(prompt)))
    st.session_state.messages.append({"role": "ai", "content": response})

if file := st.file_uploader("Upload document", type=("txt", "pdf", "md")):
    embed_file(get_session(), file)
