from typing import Dict, Any, Set
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search.tool import TavilySearchResults

load_dotenv()
INDEX_NAME = "langchain-doc-index"

# document_search > sources formatting
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# Agent Executor
def build_agent_executor():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 문서 검색 retirever tool
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="document_search",
        description="Use this tool to answer questions about the LangChain documentation."
    )

    # 웹 검색 tool
    web_search_tool = TavilySearchResults(k=3)

    tools = [retriever_tool, web_search_tool]

    prompt = hub.pull("hwchase17/react")
    chat_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    agent = create_react_agent(llm=chat_model, tools=tools, prompt=prompt)

    # Agent Executor 생성
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return executor, retriever

# run agent 함수 -> main 에서 실행
def run_agent(query: str) -> Dict[str, Any]:
    executor, retriever = build_agent_executor()

    result = executor.invoke({"input": query})
    output = result["output"]

    used_document_search = any(
        step[0].tool == "document_search" for step in result.get("intermediate_steps", [])
    )

    # tool로 "document_search"이 사용된 경우 sources를 표시
    sources = []
    if used_document_search:
        docs = retriever.get_relevant_documents(query)
        sources = sorted(set(doc.metadata.get("source", "unknown") for doc in docs))

    return {
        "result": output,
        "sources": sources
    }