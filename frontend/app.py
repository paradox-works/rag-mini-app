import requests
import streamlit as st

st.set_page_config(page_title="Paradox Works • RAG Mini App", page_icon="🧠", layout="centered")

st.title("RAG Mini App")
st.caption("Local RAG demo: upload docs → ask questions → get answers with citations.")

backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000")
top_k = st.sidebar.slider("Top K chunks", min_value=2, max_value=8, value=4)

st.sidebar.markdown("---")
if st.sidebar.button("Reset index (clear docs)"):
    r = requests.post(f"{backend_url}/reset", timeout=30)
    st.sidebar.success("Index reset." if r.ok else "Reset failed.")

st.subheader("1) Upload a document")
uploaded = st.file_uploader("Upload .pdf or .txt", type=["pdf", "txt"])

if uploaded is not None:
    with st.spinner("Uploading + indexing..."):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        r = requests.post(f"{backend_url}/ingest", files=files, timeout=120)
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
        r = requests.post(f"{backend_url}/query", json=payload, timeout=180)

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
