import streamlit as st
import os

# Inject streamlit secrets into env before config loads (falls back to .env locally)
try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
    os.environ.setdefault("LANCE_DB_DIR", st.secrets.get("LANCE_DB_DIR", "./.lancedb"))
    os.environ.setdefault("TABLE_NAME", st.secrets.get("TABLE_NAME", "document_chunks"))
except Exception:
    pass  # Fall back to .env file via pydantic-settings

from database import vector_db
from embeddings import embedding_service
from llm_service import llm_service

st.set_page_config(page_title="RAG App", page_icon="🔍")
st.title("🔍 Cost-Efficient RAG Application")

tab1, tab2 = st.tabs(["📥 Ingest", "💬 Query"])

with tab1:
    st.subheader("Ingest Document Chunks")
    text = st.text_area("Document Text", placeholder="Paste your document text here...")
    doc_id = st.text_input("Document ID", placeholder="e.g. doc_1")
    file_type = st.selectbox("File Type", ["pdf", "html", "md"])

    if st.button("Ingest"):
        if not text or not doc_id:
            st.warning("Please fill in both text and document ID.")
        else:
            with st.spinner("Ingesting..."):
                vector_db.ingest_chunks([{"text": text, "doc_id": doc_id, "file_type": file_type}])
            st.success("✅ Ingested successfully!")

with tab2:
    st.subheader("Query the RAG Service")
    query = st.text_input("Your Question", placeholder="e.g. What is the retention policy?")
    k = st.slider("Top-k chunks", min_value=1, max_value=10, value=3)
    file_filter = st.selectbox("Filter by file type (optional)", ["None", "pdf", "html", "md"])

    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                query_vector = embedding_service.embed_texts([query])[0]
                filter_val = None if file_filter == "None" else file_filter
                contexts = vector_db.search(query_vector, k=k, file_type_filter=filter_val)
                answer, metrics = llm_service.generate_answer(query, contexts)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Citations")
            st.write(", ".join([c["doc_id"] for c in contexts]) if contexts else "No citations")

            st.markdown("### Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Latency", f"{metrics['latency_ms']:.0f} ms")
            col2.metric("Chunks Retrieved", len(contexts))
            col3.metric("Prompt Tokens", metrics["prompt_tokens"])
