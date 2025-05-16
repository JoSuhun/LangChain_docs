from dotenv import load_dotenv

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.tavily_search import TavilySearchResults

load_dotenv()
INDEX_NAME = "langchain-doc-index"

def build_agent():
    # 모델, 임베딩, 문서 벡터 스토어
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 문서 검색 툴
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="document_search",
        description="Use this tool to answer questions about the LangChain documentation."
    )

    # 웹 검색 툴
    web_search_tool = TavilySearchResults(k=3)

    tools = [retriever_tool, web_search_tool]

    # 프롬프트 + Agent 생성
    prompt = hub.pull("hwchase17/react")
    chat_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    agent = create_react_agent(llm=chat_model, tools=tools, prompt=prompt)

    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

AGENT_EXECUTOR = build_agent()

def run_agent(query: str) -> str:
    result = AGENT_EXECUTOR.invoke({"input": query})
    return result["output"]
