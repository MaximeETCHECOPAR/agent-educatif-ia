# =========================== #
# AGENT EDUCATEUR INTELLIGENT #
# Auteur : Maxime ETCHECOPAR  # 
# =========================== #

import os
import math
import re
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv # Import pour charger le fichier .env

# Importations LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from gtts import gTTS

# ============================ #
# 1. Configuration & API Keys  #
# ============================ #

# Charge les clés depuis le fichier .env
load_dotenv()

# Vérification de sécurité 
if not os.getenv("GOOGLE_API_KEY"):
    print("ERREUR : Clé Google non trouvée. Vérifie le fichier .env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=None)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# ============================ #
# 2. Définition des Outils     #
# ============================ #

wikipedia = WikipediaAPIWrapper(lang='fr') 
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia.run,
    description="Obligatoire pour générer le contenu du cours."
)

search_tool = Tool(
    name="Recherche Web",
    func=TavilySearchResults(max_results=3).run,
    description="Pour des infos d'actualité."
)

def safe_calculator(expression):
    expression = re.sub(r'[^0-9+\-*/().]', '', expression)
    try:
        result = eval(expression, {"__builtins__": None}, {"math": math})
        return f"Résultat: {result}"
    except Exception as e:
        return f"Erreur: {str(e)}"

calculator_tool = Tool(
    name="Calculatrice",
    func=safe_calculator,
    description="OBLIGATOIRE pour calculer le score du QCM en pourcentage."
)

python_tool = PythonREPLTool()
python_tool.name = "Python Executor"
python_tool.description = "Pour exécuter du code"

def get_current_time(query=""):
    return datetime.now().strftime("%H:%M:%S")

time_tool = Tool(
    name="Heure Actuelle",
    func=get_current_time,
    description="Donne l'heure."
)

def speak_text(text: str):
    try:
        filename = "output_audio.mp3"
        clean_text = text.replace("*", "").replace("#", "")[:400] 
        tts = gTTS(text=clean_text, lang='fr')
        tts.save(filename)
        return filename
    except Exception as e:
        return None

tts_tool = Tool(
    name="Lecteur Vocal",
    func=speak_text,
    description="Pour lire du texte."
)

tools = [wikipedia_tool, search_tool, calculator_tool, python_tool, time_tool, tts_tool]

# ============================ #
# 3. Agent & Logique Métier    #
# ============================ #

# --- PROMPT SYSTÈME ---
system_message = """
Tu es un Professeur expert et pédagogue.

RÈGLES DE COMPORTEMENT :
1. MODE COURS : Utilise Wikipedia pour expliquer le sujet clairement.

2. MODE QCM (FORMATAGE CRITIQUE) :
   Génère les questions demandées.
   Tu DOIS utiliser des puces Markdown (-) pour les réponses afin qu'elles soient en liste.
   
   Exemple de format attendu :
   **Question 1 : Quelle est la capitale de la France ?**
   
   - A) Lyon
   - B) Paris
   - C) Marseille
   
   (Fais bien des sauts de ligne vides entre la question et les choix).

3. MODE CORRECTION :
   - Dis si c'est Correct ou Incorrect pour chaque question.
   - Explique POURQUOI c'est faux si nécessaire.
   - Utilise l'outil CALCULATRICE pour le score.
   - Affiche le score : "Score global : X%".
   - Termine par une section : "Ce qu'il faut réviser :" avec des mots-clés.
"""

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_message}
)

# --- Fonctions Auxiliaires ---

def extraire_score(texte):
    match = re.search(r'(\d+(?:[\.,]\d+)?)\s*%', texte)
    if match:
        return f"{match.group(1)}%"
    return "Non détecté"

# --- Fonctions Gradio ---

def chat_cours(message, history):
    """Fonction Chatbot"""
    if not message: return "", history
    prompt = f"Mode COURS. L'utilisateur dit : {message}"
    try:
        response = agent_chain.run(input=prompt)
    except Exception as e:
        response = f"Erreur : {str(e)}"
    history.append((message, response))
    return "", history

def generer_qcm(sujet, niveau, nombre_questions, progress=gr.Progress()):
    if not sujet: return "Attention aucun sujet défini ! Veuillez entrer un sujet ci-dessus."
    
    progress(0, desc="Création du quiz...")

    prompt = f"""
    Agis en MODE QCM.
    Sujet : {sujet}
    Difficulté : {niveau}
    Nombre de questions : {nombre_questions}
    
    Génère les questions maintenant.
    
    INSTRUCTION DE MISE EN PAGE OBLIGATOIRE :
    1. Écris la question en GRAS.
    2. SAUTE UNE LIGNE VIDE.
    3. Écris chaque choix de réponse (A, B, C) sur une NOUVELLE LIGNE précédée d'un tiret (-).
    4. SAUTE UNE LIGNE VIDE entre chaque question.
    
    Ne donne PAS les réponses maintenant.
    """
    reponse = agent_chain.run(input=prompt)
    return reponse

def corriger_et_archiver(reponses_utilisateur, sujet, niveau, historique_data, progress=gr.Progress()):
    if not reponses_utilisateur: return "Veuillez entrer vos réponses.", historique_data
    
    progress(0, desc="Correction en cours...")

    prompt = f"""
    Agis en MODE CORRECTION.
    Voici mes réponses : {reponses_utilisateur}.
    1. Corrige chaque question en expliquant pourquoi c'est juste ou faux.
    2. Utilise la calculatrice pour calculer le score exact en pourcentage.
    3. Donne-moi des pistes de révision précises (sujets à approfondir) basées sur mes erreurs.
    """
    reponse_ia = agent_chain.run(input=prompt)
    
    score = extraire_score(reponse_ia)
    date_jour = datetime.now().strftime("%Y-%m-%d %H:%M")
    nouvelle_ligne = [date_jour, sujet, niveau, score]
    
    if historique_data is None:
        historique_data = []
    historique_data.insert(0, nouvelle_ligne)
    
    return reponse_ia, historique_data

def refresh_table(historique_data):
    return historique_data

# ============================ #
# 4. Interface Multi-Onglets   #
# ============================ #

with gr.Blocks(title="Plateforme Éducative IA", theme=gr.themes.Soft()) as interface:
    
    history_state = gr.State([])         

    gr.Markdown("Plateforme Éducative Intelligente")
    
    # --- ONGLET 1 : CHATBOT ---
    with gr.Tab("Discussion & Cours"):
        gr.Markdown("### Discutez avec le professeur pour apprendre")
        chatbot_cours = gr.Chatbot(height=400, label="Professeur")
        msg_input = gr.Textbox(label="Votre message / Sujet", placeholder="Bonjour, explique-moi la photosynthèse...")
        btn_send = gr.Button("Envoyer", variant="primary")
        
        btn_send.click(
            fn=chat_cours, 
            inputs=[msg_input, chatbot_cours], 
            outputs=[msg_input, chatbot_cours]
        )
        msg_input.submit(
            fn=chat_cours, 
            inputs=[msg_input, chatbot_cours], 
            outputs=[msg_input, chatbot_cours]
        )

    # --- ONGLET 2 : QCM ---
    with gr.Tab("Entraînement QCM"):
        gr.Markdown("### Configurez votre Quiz")
        
        with gr.Row():
            qcm_sujet_input = gr.Textbox(label="Sujet du Quiz", placeholder="Entrez le sujet ici (ex: Les Volcans)", interactive=True)
        
        with gr.Row():
            qcm_niveau = gr.Radio(["Facile", "Moyen", "Difficile", "Expert"], label="Niveau", value="Moyen")
            qcm_nombre = gr.Slider(1, 10, value=3, step=1, label="Nombre de questions")
        
        btn_start_qcm = gr.Button("Générer le QCM", variant="primary")
        qcm_display = gr.Markdown(label="Questions")
        
        gr.Markdown("---")
        gr.Markdown("### Vos Réponses")
        with gr.Row():
            input_reponses = gr.Textbox(label="Réponses", placeholder="Ex: 1A, 2C")
            btn_valider = gr.Button("Valider & Corriger ✅", variant="secondary")
        
        correction_display = gr.Markdown(label="Correction & Conseils")
        
        # Actions QCM
        btn_start_qcm.click(
            fn=generer_qcm,
            inputs=[qcm_sujet_input, qcm_niveau, qcm_nombre],
            outputs=[qcm_display],
            show_progress="minimal" 
        )
        
        btn_valider.click(
            fn=corriger_et_archiver,
            inputs=[input_reponses, qcm_sujet_input, qcm_niveau, history_state],
            outputs=[correction_display, history_state],
            show_progress="minimal"
        )

    # --- ONGLET 3 : HISTORIQUE ---
    with gr.Tab("Historique & Progression"):
        gr.Markdown("### Vos résultats aux précédents QCM")
        btn_refresh = gr.Button("Actualiser le tableau 🔄")
        
        history_table = gr.Dataframe(
            headers=["Date", "Sujet", "Niveau", "Score"],
            datatype=["str", "str", "str", "str"],
            value=[],
            label="Journal de bord"
        )
        
        btn_refresh.click(fn=refresh_table, inputs=[history_state], outputs=[history_table])

if __name__ == "__main__":
    interface.queue().launch(share=True)