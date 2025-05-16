from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search.tool import TavilySearchResults
import streamlit as st
from streamlit_chat import message

# Load environment variables
load_dotenv()
INDEX_NAME = "langchain-doc-index"

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# Build Agent Executor with tools
def build_agent_executor():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="document_search",
        description="Use this tool to answer questions about the LangChain documentation."
    )

    web_search_tool = TavilySearchResults(k=3)
    tools = [retriever_tool, web_search_tool]

    prompt = hub.pull("hwchase17/react")
    chat_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    agent = create_react_agent(llm=chat_model, tools=tools, prompt=prompt)

    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return executor, retriever

def run_agent_with_sources(query: str) -> Dict[str, Any]:
    executor, retriever = build_agent_executor()

    # 에이전트 실행
    result = executor.invoke({"input": query})
    output = result["output"]

    used_document_search = any(
        step[0].tool == "document_search" for step in result.get("intermediate_steps", [])
    )

    sources = []
    if used_document_search:
        docs = retriever.get_relevant_documents(query)
        sources = sorted(set(doc.metadata.get("source", "unknown") for doc in docs))

    return {
        "result": output,
        "sources": sources
    }