import streamlit as st
import requests

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant — Step 2")

st.subheader("Upload document (PDF/TXT) for RAG")
uploaded_file = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"])
if uploaded_file is not None:
    if st.button("Upload Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            resp = requests.post("http://localhost:8000/upload", files=files)
            st.success(f"Upload response: {resp.json()}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

query = st.text_input("Ask something:")

if st.button("Send to API"):
    try:
        resp = requests.post(
            "http://localhost:8000/ask",
            json={"query": query}
        )
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Answer")
            st.write(data["answer"])
        if data["citations"]:
            for c in data["citations"]:
                score = c.get("score", None)
                if score is not None:
                    st.markdown(f"- **{c['title']}** (score={score:.4f}) — [link]({c['url']})")
                # else:
                #     st.markdown(f"- **{c['title']}** — [link]({c['url']})")
            # st.subheader("Citations")
            # for c in data["citations"]:
            #     st.markdown(f"- [{c['title']}]({c['url']})")
        else:
            st.error(f"API error: {resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"Could not reach API: {e}")

