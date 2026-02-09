"""
Streamlit Chat UI for Clinical Decision Support Agent.
Simple chat interface with citations display.
Compatible with Streamlit 1.9.0.
"""
import streamlit as st
import requests
import uuid
from pathlib import Path


API_URL = "http://localhost:8000"

st.set_page_config(page_title="Clinical Agent", layout="centered")


css_file = Path(__file__).parent / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("Chat Agent")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    
    def clear_history():
        try:
            requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
        except:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
            
    st.button("Clear History", on_click=clear_history)


st.title("Clinical Decision Support")


for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**Assistant:** {content}")
        
        # Show citations if present
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    page = c.get("page", "?")
                    excerpt = c.get("excerpt", "")[:150]
                    st.markdown(f"**[NG12 p.{page}]** {excerpt}...")

st.markdown("---")


with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Ask a clinical question:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and prompt:
    print(f"[UI] User submitted: {prompt}")

    st.session_state.messages.append({"role": "user", "content": prompt})
    print(f"[UI] Added to history. Current len: {len(st.session_state.messages)}")
    

    with st.spinner("Thinking..."):
        try:
            print(f"[UI] Sending POST to {API_URL}/chat...")
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "message": prompt
                },
                timeout=300
            )
            print(f"[UI] Response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            print(f"[UI] Response data keys: {data.keys()}")
            
            answer = data.get("answer", "No response")
            citations = data.get("citations", [])
            
            # Save to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            print(f"[UI] Added assessment to history. New len: {len(st.session_state.messages)}")
            
        except requests.exceptions.ConnectionError:
            print("[UI ERROR] ConnectionError")
            st.error("Cannot connect to API. Please make sure the server is running.")
        except requests.exceptions.Timeout:
            print("[UI ERROR] Timeout")
            st.error("Request timed out. The agent might be busy or retrying. Please try again.")
        except Exception as e:
            print(f"[UI ERROR] Exception: {e}")
            st.error(f"Error: {str(e)}. Please try again.")
    
    print("[UI] Rerunning app...")
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
