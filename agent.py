import uuid
from conversationbot_graph import start_chat

def run_agent(message: str, session_id: str | None):
    if not session_id:
        session_id = str(uuid.uuid4())

    # --- AI LOGIC GOES HERE ---
    reply = start_chat(message,session_id)
    #reply = f"You said: {message}"

    return reply, session_id