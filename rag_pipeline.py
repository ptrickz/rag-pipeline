from retriever import retrieve
from generator import build_prompt, generate_response

def ask_question(query: str) -> str:
    context = retrieve(query)
    prompt = build_prompt(context, query)
    return generate_response(prompt)
