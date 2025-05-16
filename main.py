import streamlit as st
from streamlit_chat import message
from backend.core import run_agent

st.header("LangChain Doc & Web Assistant 🤖")

prompt = st.text_input("Prompt", placeholder="Ask anything...")

# 세션 상태 초기화
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating response..."):
        response = run_agent(prompt)

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", response))

if st.session_state["chat_answers_history"]:
    for query, answer in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]):
        message(query, is_user=True)
        message(answer)