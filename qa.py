import faiss-cpu
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ingest import load_index_and_chunks

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

def retrieve(query, top_k=4):
    index, chunks = load_index_and_chunks()
    if index is None or chunks is None:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), top_k)
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx].page_content)
    return results

def generate_answer(question, contexts, max_length=256):
    context_text = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are an assistant that answers questions using only the provided context. "
        "If the answer is not contained in the context, say 'I don't know.' "
"Keep the answer concise.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ans

def answer_question(query):
    contexts = retrieve(query, top_k=4)
    if not contexts:
        return "No documents ingested yet. Please upload documents first."
    answer = generate_answer(query, contexts)
    return answer,contexts
