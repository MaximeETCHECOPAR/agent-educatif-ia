# =========================== #
# Projet IA : AGENT EDUCATIF  #
# Auteur : Maxime ETCHECOPAR  #
# Date de rendu : 19/12/2025  # 
# =========================== #

import os
import math
import re
import time
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv

# Importations LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from gtts import gTTS

# ================= #
# 1. Configuration  #
# ================= #

# Charge les variables d'environnement (clé api)
load_dotenv()

# Verif clé
if not os.getenv("GOOGLE_API_KEY"):
    print("ERREUR : Clé Google non trouvée, vérifier le fichier .env")

# Initialisation du LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=None)

# Initialisation de la mémoire
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# ============================ #
# 2. Définition des Outils     #
# ============================ #

# Outil 1 : Wikipedia pour le savoir
wikipedia = WikipediaAPIWrapper(lang='fr') 
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia.run
)

# Outil 2 : Tavily pour les recherches web.
search_tool = Tool(
    name="Recherche Web",
    func=TavilySearchResults(max_results=3).run
)

# Fonction pour la calculatrice
def calculator(expression):
    """
    Nettoie les expressions mathématiques (enlève les lettres par exemple) et la calcule
    Utilise "eval".
    """
    expression = re.sub(r'[^0-9+\-*/().]', '', expression)
    try:
        result = eval(expression, {"__builtins__": None}, {"math": math})
        return f"Résultat: {result}"
    except Exception as e:
        return f"Erreur: {str(e)}"

# Outil 3 : La Calculatrice
calculator_tool = Tool(
    name="Calculatrice",
    func=calculator
)

# Outil 4 : Exécuteur Python
python_tool = PythonREPLTool()
python_tool.name = "Python Executor"

# Fonction pour l'heure
def get_current_time(query=""):
    """Retourne l'heure actuelle au format HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")

# Outil 5 : Heure actuelle
time_tool = Tool(
    name="Heure Actuelle",
    func=get_current_time,
)

# Fonction pour le TTS (Text-To-Speech)
def speak_text(text: str):
    """
    Convertit un texte donné en fichier audio MP3.
    Retourne le chemin du fichier généré pour l'afficher.
    """
    try:
        filename = "correction_audio.mp3"
        # Nettoyage du texte pour éviter de lire les caractères spéciaux
        clean_text = text.replace("*", "").replace("#", "").replace("-", "")[:1000] 
        tts = gTTS(text=clean_text, lang='fr')
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Erreur audio: {e}")
        return None

# Outil 6 : Lecteur Vocal
tts_tool = Tool(
    name="Lecteur Vocal",
    func=speak_text,
    description="Pour lire du texte."
)

# Liste complète des outils fournis à l'agent
tools = [wikipedia_tool, search_tool, calculator_tool, python_tool, time_tool, tts_tool]

# ============================ #
# 3. Agent & Logique Métier    #
# ============================ #

# Prompt Système
system_message = """
Tu es un Professeur expert et pédagogue.

RÈGLES DE COMPORTEMENT :
1. MODE COURS : Utilise Wikipedia pour expliquer le sujet clairement.

2. MODE QCM :
   Génère les questions demandées.
   Tu DOIS utiliser des puces Markdown (-) pour les réponses afin qu'elles soient en liste.
   SAUTE UNE LIGNE VIDE entre chaque question.

3. MODE CORRECTION (Formatage STRICT) :
   - Pour CHAQUE question, tu dois suivre EXACTEMENT ce format :
     "**Question X :** [Texte de la question]"
     "**Votre réponse : [Lettre] - [Correct ou Incorrect]**"
     "**Explication :** [Ton explication pédagogique]"
   
   - Ensuite, utilise l'outil CALCULATRICE pour le score.
   - Affiche le score : "Score global : X%".
   - Termine par : "Ce qu'il faut réviser :" avec des mots-clés.
"""

# Initialisation de l'agent ReAct
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_message}
)

# --- Fonctions autres ---

def extraire_score(texte):
    """
    Sert à remplir le tableau d'historique
    """
    match = re.search(r'(\d+(?:[\.,]\d+)?)\s*%', texte)


def nettoyer_sortie(texte):
    """
    Nettoie la réponse finale de l'agent parce que l'agent laisse parfois des traces de code JSON.
    Cette fonction supprime ces ces problèmes pour avoir un affichage propre.
    """
    texte = re.sub(r'```json.*?```', '', texte, flags=re.DOTALL)
    texte = re.sub(r'\{"action":.*?\}', '', texte, flags=re.DOTALL)
    return texte.strip()

# --- Fonctions pour l'interface ---

def chat_cours(message, history):
    """
    Gère l'onglet 1 (la discussion).
    Envoie le message à l'agent avec le contexte 'Mode COURS'.
    """
    if not message: return "", history
    prompt = f"Mode COURS. L'utilisateur dit : {message}"
    try:
        response = agent_chain.run(input=prompt)
    except Exception as e:
        response = f"Erreur : {str(e)}"
    
    response_propre = nettoyer_sortie(response)
    history.append((message, response_propre))
    return "", history

def generer_qcm(sujet, niveau, nombre_questions, progress=gr.Progress()):
    """
    Gère dans l'onglet 2 (le Bouton Générer qcm).
    Crée le QCM en fonction des paramètres.
    Utilise 'progress' pour afficher une barre de chargement.
    """
    if not sujet: return "Attention aucun sujet défini ! Veuillez entrer un sujet."
    
    progress(0.1, desc="Initialisation...")
    time.sleep(1) 
    progress(0.4, desc="Rédaction des questions...")

    prompt = f"""
    Agis en MODE QCM.
    Sujet : {sujet}
    Difficulté : {niveau}
    Nombre de questions : {nombre_questions}
    
    Génère les questions maintenant.
    Format : Question en gras, sauts de lignes, réponses en liste à puces (- A)...).
    Ne donne PAS les réponses maintenant.
    """
    reponse = agent_chain.run(input=prompt)
    return nettoyer_sortie(reponse)

def corriger_et_archiver(reponses_utilisateur, sujet, niveau, historique_data, progress=gr.Progress()):
    """
    Gère dans l'onglet 2 ( le bouton Valider pour la correction).
    1. Corrige les réponses.
    2. Calcule le score.
    3. Génère l'audio de la correction.
    4. Met à jour l'historique (tableau).
    """
    if not reponses_utilisateur: return "Veuillez entrer vos réponses.", historique_data, None
    
    progress(0.1, desc="Analyse de vos réponses...")
    time.sleep(1)
    progress(0.5, desc="Calcul du score...")

    prompt = f"""
    Agis en MODE CORRECTION.
    Voici mes réponses : {reponses_utilisateur}.
    
    FORMAT OBLIGATOIRE pour chaque question :
    "Votre réponse : [Lettre] - [Correct ou Incorrect]"
    
    Ensuite explique pourquoi c'est juste ou faux.
    Calcule le score avec la calculatrice.
    """
    reponse_ia = agent_chain.run(input=prompt)
    
    reponse_propre = nettoyer_sortie(reponse_ia)
    
    # Génération de l'audio
    progress(0.9, desc="Génération de l'audio...")
    chemin_audio = speak_text(reponse_propre)

    # Extraction du score et mise à jour de l'historique
    score = extraire_score(reponse_propre)
    date_jour = datetime.now().strftime("%Y-%m-%d %H:%M")
    nouvelle_ligne = [date_jour, sujet, niveau, score]
    
    if historique_data is None:
        historique_data = []
    historique_data.insert(0, nouvelle_ligne)
    
    # On retourne 3 éléments pour les mettre à jour
    return reponse_propre, historique_data, chemin_audio

def refresh_table(historique_data):
    """Simple fonction pour rafraîchir la vue du tableau d'historique."""
    return historique_data

# ============================ #
# 4. Interface Multi-onglets   #
# ============================ #

with gr.Blocks(title="Plateforme Éducative IA", theme=gr.themes.Soft()) as interface:
    
    history_state = gr.State([])        

    gr.Markdown("Plateforme Éducative Intelligente")
    
    # --- ONGLET 1 : CHATBOT ---
    with gr.Tab("Discussion & Cours"):
        gr.Markdown("### Discutez avec le professeur")
        chatbot_cours = gr.Chatbot(height=400, label="Professeur")
        msg_input = gr.Textbox(label="Votre message", placeholder="Bonjour, explique-moi la photosynthèse...")
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
            btn_valider = gr.Button("Valider & Corriger", variant="secondary")
        
        correction_display = gr.Markdown(label="Correction & Conseils")
        
        # Lecteur Audio
        audio_player = gr.Audio(label="Écouter la correction", type="filepath", interactive=False)
        
        # Actions QCM
        btn_start_qcm.click(
            fn=generer_qcm,
            inputs=[qcm_sujet_input, qcm_niveau, qcm_nombre],
            outputs=[qcm_display]
        )
        
        # Quand on clic sur valider, on met tout à jour
        btn_valider.click(
            fn=corriger_et_archiver,
            inputs=[input_reponses, qcm_sujet_input, qcm_niveau, history_state],
            outputs=[correction_display, history_state, audio_player]
        )

    # --- ONGLET 3 : HISTORIQUE ---
    with gr.Tab("Historique & Progression"):
        gr.Markdown("### Vos résultats aux précédents QCM")
        btn_refresh = gr.Button("Actualiser le tableau")
        
        history_table = gr.Dataframe(
            headers=["Date", "Sujet", "Niveau", "Score"],
            datatype=["str", "str", "str", "str"],
            value=[],
            label="Journal de bord"
        )
        
        btn_refresh.click(fn=refresh_table, inputs=[history_state], outputs=[history_table])

if __name__ == "__main__":
    interface.queue().launch(share=True)