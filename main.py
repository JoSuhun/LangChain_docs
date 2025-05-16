import streamlit as st
from streamlit_chat import message
from backend.core import run_agent, create_sources_string

st.header("Langchain-Doc Helper Bot")

# Streamlit에서 입력받는 prompt
prompt = st.text_input("Prompt", placeholder="Enter your prompt here . . .")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if prompt:
    with st.spinner("Generating response .."):
        # core.py run_agent 실행
        res = run_agent(prompt)
        formatted_response = f"{res['result']}\n\n" + create_sources_string(set(res["sources"]))

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for response, user_msg in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(user_msg, is_user=True)
        message(response)
