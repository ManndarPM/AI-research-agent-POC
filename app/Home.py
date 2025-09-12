import streamlit as st
import requests

st.set_page_config(page_title="AI Research Assistant - Demo", layout="wide")
st.title("AI Research Assistant — Demo (Step 1)")

st.write("This is a minimal Streamlit UI. It will call FastAPI below.")

query = st.text_input("Ask something (demo):", "")
if st.button("Send to API"):
    try:
        resp = requests.get("http://localhost:8000/")
        st.write("API response:", resp.json())
    except Exception as e:
        st.error(f"Could not reach API: {e}")

st.write("---")
st.write("When FastAPI is running at http://localhost:8000/ you should see a JSON response above.")
