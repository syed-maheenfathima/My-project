import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils import read_pdf, save_pickle, load_pickle

# === Paths ===
PERSIST_DIR = "persistence"
FAISS_INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")


# === Split PDF text into small chunks ===
def chunk_texts(texts, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []

    for text in texts:
        docs = [Document(page_content=text)]
        split_chunks = splitter.split_documents(docs)
        chunks.extend(split_chunks)

    return chunks


# === Create FAISS embeddings index ===
def build_embeddings_and_index(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))

    return index, embeddings


# === Ingest PDF file, chunk, embed, and save ===
def ingest_file(file_path):
    text = read_pdf(file_path)
    chunks = chunk_texts([text])
    index, embeddings = build_embeddings_and_index(chunks)

    os.makedirs(PERSIST_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_pickle(chunks, CHUNKS_PATH)

    print(f"Ingested and saved {len(chunks)} chunks.")
    return True


# === Load previously saved FAISS index and chunks ===
def load_index_and_chunks():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("No saved index/chunks found.")
        return None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    chunks = load_pickle(CHUNKS_PATH)
    return index, chunks