import os
import tempfile
import streamlit as st
from ingest import ingest_file, load_index_and_chunks
from qa import answer_question

st.set_page_config(page_title="StudyMate AI", layout="wide")
st.title("StudyMate AI - Your Personal Academic Chatbot")
st.markdown("Upload PDFs and ask questions. Powered by LLMs and Vector Databases.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(uploaded.read())
    tmp_file.flush()
    tmp_file.close()
    with st.spinner("Ingesting document (creating embeddings)..."):
        ingest_file(tmp_file.name)
    st.success("Document ingested. You can now ask questions.")

index, chunks = load_index_and_chunks()
if index is None:
    st.info("No documents ingested yet. Upload a PDF to start.")
else:
    query = st.text_input("Enter your question about the uploaded documents")
    if st.button("Ask"):
        with st.spinner("Retrieving answer..."):
            result = answer_question(query)
            if isinstance(result, tuple):
                ans, contexts = result
                st.markdown("### 🧠 Answer")
                st.write(ans)
                st.markdown("### 📚 Retrieved contexts")
                for i, c in enumerate(contexts):
                    st.markdown(f"*Context {i+1}:*")
                    st.write(c[:1000] + ("..." if len(c) > 1000 else ""))
            else:
                st.write(result)

