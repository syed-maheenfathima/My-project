import os
import pickle
from pypdf import PdfReader

def read_pdf(path):
    text_pages = []
    reader = PdfReader(path)
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        if text:
            text_pages.append(text)
    return "\n".join(text_pages)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
