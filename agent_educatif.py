# =========================== #
# AGENT EDUCATEUR INTELLIGENT # 
# =========================== #

import os
import gradio as gr


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

import gradio as gr


# Déclaration de la clef d'API
os.environ["GOOGLE_API_KEY"] = "MA_CLE_GOOGLE_ICI"

# Création d'une instance de chat
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=None)

# Mémoire
memory = ConversationBufferMemory(memory_key="history", return_messages=True)





# ================ #
# Interface Gradio #
# ================ #
def agent_chat(message, history):
    """Chat  avec l'agent."""
    if not message:
        return history, ""
    history = history or []
    history.append((message))
    return history, ""



# ==================== #
# Construction de l’UI #
# ==================== #
with gr.Blocks(title="Agent Éducatif Intelligent") as interface:
    gr.Markdown("Ton Agent Éducatif Intelligent")

    with gr.Tab("Discussion"):
        chatbox = gr.Chatbot(height=400)
        msg = gr.Textbox(label="Message")
        send = gr.Button("Envoyer")
        send.click(fn=agent_chat, inputs=[msg, chatbox], outputs=[chatbox, msg])


interface.launch(share=True)