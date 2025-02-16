__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from duckduckgo_search import DDGS
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.document_loaders import PyPDFLoader
import os

OPENAI_API_KEY = "your-api-key-here"

# Funzione per cercare informazioni aggiornate su DuckDuckGo
#def cerca_su_web(query: str) -> str:
#    """Esegue una ricerca su DuckDuckGo e restituisce i primi risultati."""
#    with DDGS() as ddgs:
#        risultati = [f"{r['title']} - {r['href']}" for r in ddgs.text(query, max_results=3)]
#    return "\n".join(risultati)

# Tool valido per CrewAI
#search_tool = Tool(
#    name="Ricerca Web",
#    func=lambda query: cerca_su_web(query),
#    description="Usa DuckDuckGo per trovare informazioni aggiornate su un argomento."
#)

# Funzione per leggere il testo da un PDF
def estrai_testo_da_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pagine = loader.load()
    testo_completo = "\n".join([pagina.page_content for pagina in pagine])
    st.write(testo_completo)
    return testo_completo

def create_agents(use_web,x,y, pdf_text=None):
    """Crea agenti CrewAI per Ricerca Web o Analisi PDF"""
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Puoi usare "gpt-4"
        openai_api_key=OPENAI_API_KEY
    )

    analyst = Agent(
    role="Analista Tematico",
    goal=f"Identificare i {x} temi principali nel testo {pdf_text}." if use_quiz else "TBD per storie",
    backstory="Esperto in analisi testuale e individuazione di argomenti chiave.",
    verbose=True,
    memory=True,
    allow_delegation=True,
    llm=llm  
    )

    quiz_creator = Agent(
    role="Costruttore di Quiz",
    goal="Creare {y} domande in italiano per ognuno degli {x} temi individuati, con 4 opzioni di risposta: una corretta, una parzialmente corretta, una errata, una errata e dannosa.",
    backstory="Esperto nella creazione di quiz e test di valutazione.",
    verbose=True,
    memory=True,
    allow_delegation=True,
    llm=llm
    )

    answer_evaluator = Agent(
    role="Valutatore Risposte",
    goal="Assegnare un punteggio tra quelli disponibili a ciascuna opzione di risposta in base al grado di correttezza: 5 per risposte corrette, 2 per risposte parzialmente corrette, 0 per risposte errate, -5 per risposte errate e dannose.",
    backstory="Esperto nella valutazione di domande a scelta multipla.",
    verbose=True,
    memory=True,
    allow_delegation=True,
    llm=llm
    )

    return analyst, quiz_creator, answer_evaluator

def create_crew(use_quiz,x,y, pdf_text=None):
    """Crea il CrewAI e definisce i task in base alla scelta dell'utente."""
    
    analyst, quiz_creator, answer_evaluator = create_agents(use_quiz,x,y, pdf_text)

    if use_quiz:
        extract_themes_task = Task(
        description=f"Analizza il contenuto del testo {pdf_text} e identifica i {x} temi più importanti.",
        agent=analyst,
        expected_output="Elenco di 3 temi in italiano"
        )

        generate_questions_task = Task(
        description=f"""Per ognuno dei {x} temi individuati, genera esattamente {y} domande in italiano in base alle informazioni contenute nel documento {pdf_text}.  
    Ogni domanda deve avere 4 opzioni di risposta in italiano:
    - Una corretta ✅
    - Una parzialmente corretta ⚠️
    - Una errata ❌
    - Una errata e dannosa ❌❌  
    Le domande devono essere a difficoltà crescente, ovvero fare riferimento ad elementi e concetti sempre più specifici del documento. 
    Restituisci l'output in italiano e strutturato come segue:  
    Tema1: [Nome del tema 1]  
    1. [Domanda 1]  
       A) [Opzione 1]-[Punteggio 1]
       B) [Opzione 2]-[Punteggio 2]
       C) [Opzione 3]-[Punteggio 3]
       D) [Opzione 4]-[Punteggio 4]
    2. [Domanda 2]  
       A) [Opzione 1]-[Punteggio 1]
       B) [Opzione 2]-[Punteggio 2]
       C) [Opzione 3]-[Punteggio 3]
       D) [Opzione 4]-[Punteggio 4]
    (Ripeti per le {y} domande del tema 1)
    Tema2: [Nome del tema 2]  
    1. [Domanda 1]  
       A) [Opzione 1]-[Punteggio 1]
       B) [Opzione 2]-[Punteggio 2]
       C) [Opzione 3]-[Punteggio 3]
       D) [Opzione 4]-[Punteggio 4]
    (Ripeti per le {y} domande del tema 2)
    (Ripeti per gli altri temi)""",
        agent=quiz_creator,
        depends_on=[extract_themes_task] , # Dipende dall'estrazione dei temi
        expected_output=f"Elenco di {x} temi e, per ogni tema, un elenco di {y} domande e 4 opzioni di risposta di difficoltà crescente. "
        )

        score_answers_task = Task(
        description="""Valuta il grado di correttezza delle opzioni di risposta e assegna **sempre**  
    uno dei seguenti punteggi:  
    - ✅ Corretta: **5**  
    - ⚠️ Parzialmente corretta: **2**  
    - ❌ Errata: **0**  
    - ❌❌ Errata e dannosa: **-5**  
    **Assicurati che ogni domanda abbia esattamente una risposta con ogni punteggio**.  
    Non assegnare lo stesso punteggio a più di un'opzione.
    Restituisci l'output in italiano e strutturato come un csv con le colonne Tema, Domanda, Risposta 1, Punteggio 1, Risposta 2, Punteggio 2, Risposta 3, Punteggio 3, Risposta 4, Punteggio 4""",
        agent=answer_evaluator,
        depends_on=[generate_questions_task] , # Dipende dalla generazione delle domande
        expected_output=f"Elenco in italiano di {x} temi e per ogni tema un elenco di {y} domande, 4 opzioni di risposte e un punteggio per ogni opzione di risposta"
        )
    else:
        extract_themes_task = Task(
        description="TBD per storie",
        agent=analyst,
        expected_output="TBD per storie"
        )        

        generate_questions_task = Task(
        description="TBD per storie",
        agent=quiz_creator,
        depends_on=[extract_themes_task] , 
        expected_output="TBD per storie"
        )

        score_answers_task = Task(
        description="TBD per storie",
        agent=answer_evaluator,
        depends_on=[generate_questions_task],  
        expected_output="TBD per storie"
        )

    crew = Crew(
        agents=[analyst, quiz_creator, answer_evaluator],
        tasks=[extract_themes_task , generate_questions_task, score_answers_task]
    )

    return crew

# --- INTERFACCIA STREAMLIT ---
st.title("🤖 AI Collaborativa con CrewAI e Streamlit")

# Selettore per la modalità di ricerca
option = st.radio("Scegli tra:", ["Quiz", "Storie"])

uploaded_file = st.file_uploader("Carica un file PDF", type="pdf")

pdf_text = None  # Inizializza la variabile per il testo del PDF

if option == "Quiz":
    if uploaded_file is not None:
        pdf_path = os.path.join("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pdf_text = estrai_testo_da_pdf(pdf_path)  # Estrai il testo dal PDF
        os.remove(pdf_path)  # Cancella il file temporaneo dopo la lettura

        use_quiz= True
        x=2
        y=5
        crew = create_crew(use_quiz,x,y, pdf_text)
    
        st.write("### 🚀 Elaborazione della risposta...")
        result = crew.kickoff()
    
        st.subheader("📝 Risposta Generata")
        st.write(result)
