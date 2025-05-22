from retriever import index_documents
from rag_pipeline import ask_question

# Load documents from file
with open("data/sample.txt", "r") as f:
    raw_docs = [line.strip() for line in f.readlines() if line.strip()]

# Index documents (run once)
index_documents(raw_docs)

# Example RAG usage
query = input("Ask a question: ")
answer = ask_question(query)
print("\nAnswer:", answer)
