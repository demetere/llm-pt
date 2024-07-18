import threading

from loguru import logger
from streamlit.runtime import get_instance

# Super unsafe but I couldn't find any other way to clear cache
from streamlit.runtime.caching import _resource_caches  # type: ignore
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


class Config:
    Chroma_Search_Min_Relevance_Thr = 0.0
    Chroma_Search_Top_K = 4
    Char_Text_Splitter_Chunk_Size = 4000  # XXX: NOT CHARS, BUT TOKENS
    Char_Text_Splitter_Chunk_Overlap = 200  # XXX: NOT CHARS, BUT TOKENS
    Embedding_Model = "ada"


def get_session():
    if ctx := get_script_run_ctx():
        return ctx.session_id
    else:
        raise ValueError("Thread doesn't have a state")


def start_beating(session_id: str):
    """Original
    https://discuss.streamlit.io/t/detecting-user-exit-browser-tab-closed-session-end/62066/2
    """
    # gotta import that late, so that we get cached_class
    # which we create and set for `llm` module in main.py
    from llm import chroma_client

    thread = threading.Timer(interval=2, function=start_beating, args=(session_id,))

    # insert context to the current thread, needed for
    # getting session specific attributes like st.session_state
    add_script_run_ctx(thread)
    runtime = get_instance()  # this is the main runtime, contains all the sessions

    if runtime.is_active_session(session_id):
        thread.start()
    else:
        logger.info(f"Session: {session_id} has been closed")
        chroma_client()._collection.delete(where={"session_id": session_id})  # type: ignore
        logger.info("Cache log: ")
        logger.info(_resource_caches._function_caches.keys())  # type: ignore
        return
