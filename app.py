import streamlit as st
import requests

API = "http://localhost:8000"

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
                res = requests.post(f"{API}/ingest", json={
                    "chunks": [{"text": text, "doc_id": doc_id, "file_type": file_type}]
                })
            if res.status_code == 200:
                st.success(f"✅ Ingested successfully! Chunks processed: {res.json()['processed_chunks']}")
            else:
                st.error(f"Error: {res.text}")

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
                payload = {"query": query, "k": k}
                if file_filter != "None":
                    payload["file_type_filter"] = file_filter
                res = requests.post(f"{API}/query", json=payload)

            if res.status_code == 200:
                data = res.json()
                st.markdown("### Answer")
                st.write(data["answer"])

                st.markdown("### Citations")
                st.write(", ".join(data["citations"]) if data["citations"] else "No citations")

                st.markdown("### Metrics")
                m = data["metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Latency", f"{m['total_latency_ms']:.0f} ms")
                col2.metric("Chunks Retrieved", m["chunk_count"])
                col3.metric("Prompt Tokens", m["token_usage"]["prompt_tokens"])
            else:
                st.error(f"Error: {res.text}")
