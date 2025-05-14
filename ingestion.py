import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"{len(documents)} to Pinecone")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("langchain-doc-index")

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # 배치 처리
    for i in range(0, len(documents), 100):
        batch = documents[i:i + 100]
        vectorstore.add_documents(batch)
        print(f"Added batch {i//100 + 1} with {len(batch)} documents")

    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()
