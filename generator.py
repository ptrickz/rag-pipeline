from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def build_prompt(contexts: list[str], query: str) -> str:
    context_block = "\n".join(contexts)
    return f"""You are a helpful assistant.

Context:
{context_block}

Question:
{query}

Answer:"""

def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt):].strip()
