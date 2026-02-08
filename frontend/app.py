import requests
import streamlit as st

st.set_page_config(page_title="Paradox Works • RAG Mini App", page_icon="🧠", layout="centered")

st.title("RAG Mini App")
st.caption("Local RAG demo: upload docs → ask questions → get answers with citations.")

# Safer default: lock to localhost unless you intentionally change it
LOCK_LOCALHOST = True

if LOCK_LOCALHOST:
    backend_url = "http://127.0.0.1:8000"
    st.sidebar.text_input("Backend URL (locked)", value=backend_url, disabled=True)
else:
    backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000").strip()

top_k = st.sidebar.slider("Top K chunks", min_value=2, max_value=8, value=4)

# Optional API key support (matches FastAPI header x-api-key)
api_key = st.sidebar.text_input("API key (optional)", type="password")
headers = {"x-api-key": api_key} if api_key else {}

st.sidebar.markdown("---")
if st.sidebar.button("Reset index (clear docs)"):
    try:
        r = requests.post(f"{backend_url}/reset", headers=headers, timeout=30)
        st.sidebar.success("Index reset." if r.ok else f"Reset failed: {r.status_code}")
    except requests.RequestException as e:
        st.sidebar.error(f"Reset error: {e}")

st.subheader("1) Upload a document")
uploaded = st.file_uploader("Upload .pdf or .txt", type=["pdf", "txt"])

if uploaded is not None:
    with st.spinner("Uploading + indexing..."):
        file_bytes = uploaded.getvalue()
        files = {"file": (uploaded.name, file_bytes)}
        try:
            r = requests.post(f"{backend_url}/ingest", headers=headers, files=files, timeout=180)
        except requests.RequestException as e:
            r = None
            st.error(f"Upload error: {e}")

    if r is not None:
        if r.ok:
            data = r.json()
            st.success(f"Indexed {data.get('added_chunks', 0)} chunks from {data.get('source')}.")
        else:
            st.error(f"Upload failed: {r.status_code} {r.text}")

st.subheader("2) Ask a question")
question = st.text_input("Question", placeholder="Ask something about the uploaded document...")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        payload = {"question": question.strip(), "top_k": top_k}
        try:
            r = requests.post(f"{backend_url}/query", headers=headers, json=payload, timeout=180)
        except requests.RequestException as e:
            r = None
            st.error(f"Query error: {e}")

    if r is not None:
        if not r.ok:
            st.error(f"Query failed: {r.status_code} {r.text}")
        else:
            data = r.json()
            st.markdown("### Answer")
            st.write(data.get("answer", ""))

            sources = data.get("sources", [])
            if sources:
                st.markdown("### Sources")
                for s in sources:
                    st.markdown(
                        f"**[{s['n']}] {s['source']}** — chunk {s['chunk_id']} (score {s['score']})\n\n"
                        f"> {s['preview']}"
                    )
            else:
                st.info("No sources found (index may be empty).")
