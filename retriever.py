import chromadb
from embedder import get_embedding

client = chromadb.Client()
collection = client.get_or_create_collection("rag_docs")

def index_documents(docs: list[str]):
    embeddings = [get_embedding(doc) for doc in docs]
    ids = [f"doc-{i}" for i in range(len(docs))]
    collection.add(documents=docs, embeddings=embeddings, ids=ids)

def retrieve(query: str, k=2) -> list[str]:
    query_vec = get_embedding(query)
    results = collection.query(query_embeddings=[query_vec], n_results=k)
    return results["documents"][0]
