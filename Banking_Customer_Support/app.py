import streamlit as st
import sys
import os

# -------------------------
# FIX IMPORT ISSUE
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main import ask_question

# -------------------------
# AVATARS (NEW)
# -------------------------
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

# -------------------------
# PAGE SETTINGS
# -------------------------
st.set_page_config(
    page_title="Banking RAG Bot",
    layout="centered"
)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("Settings")
    st.markdown("RAG System: Groq + HITL")
    if st.button("Clear Chat"):
        st.session_state.messages = []

# -------------------------
# TITLE
# -------------------------
st.title("Customer Support Assistant")
st.markdown("Ask anything about Banking.")

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# DISPLAY CHAT HISTORY (UPDATED)
# -------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # USER MESSAGE
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    # BOT RESPONSE
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                response = ask_question(user_input)

                # HITL UI (unchanged logic)
                if "🚨" in response:
                    st.error(response)
                else:
                    st.success(response)

            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})