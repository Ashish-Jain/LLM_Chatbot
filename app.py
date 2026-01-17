import streamlit as st
from langchain.memory import ConversationBufferMemory
from conversationbot_graph import start_chat
import os
#from flight_bot import start_chat

st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="centered"
)
bot_reply = None

st.title("🤖 Groq Chatbot")
st.markdown("We are using moonshotai/kimi-k2-instruct-0905 model")

with st.form("key_form"):
    key = st.text_input(
        "Groq API Key",
        type="password",
        help="Key is stored in session memory only"
    )
    key = key.strip()
    submit = st.form_submit_button("Save")

if submit:
    key = key.strip()
    if not key.startswith("gsk_"):
        st.error("Invalid Groq API key format")
        st.stop()

    os.environ["GROQ_API_KEY"] = key
    st.session_state.api_key = key
    st.success("Groq key saved")

if "GROQ_API_KEY" not in os.environ:
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True
    )

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")
if user_input and user_input.strip():
    session_id = "11111"
    if "api_key" in st.session_state:
        bot_reply = start_chat(user_input,session_id,st.session_state.api_key)
    else:
        st.chat_message("assistant").markdown("API Key not provided or invalid")

if user_input is not None:
    st.chat_message("assistant").markdown(user_input)
if bot_reply is not None:
    st.chat_message("assistant").markdown(bot_reply)

st.session_state.messages.append(
    {"role": "assistant", "content": bot_reply}
)

# Add to memory
if bot_reply and isinstance(bot_reply, str):
    st.session_state.memory.chat_memory.add_ai_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(bot_reply)





