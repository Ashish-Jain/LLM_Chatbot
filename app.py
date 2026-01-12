import streamlit as st
from langchain.memory import ConversationBufferMemory
from conversationbot_graph import start_chat

st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="centered"
)
bot_reply = None

st.title("🤖 Groq Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")
if user_input and user_input.strip():
    session_id = "11111"
    bot_reply = start_chat(user_input,session_id)

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





