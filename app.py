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

def create_agents(use_quiz,x,y, pdf_text=None):
    """Crea agenti CrewAI per Ricerca Web o Analisi PDF"""
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Puoi usare "gpt-4"
        openai_api_key=OPENAI_API_KEY
    )

    analyst = Agent(
    role="Analista Tematico",
    goal=f"Identificare i {x} temi principali nel testo {pdf_text}. Rispondi sempre e solo in italiano." if use_quiz else "TBD per storie",
    backstory="Esperto in analisi testuale e individuazione di argomenti chiave.",
    verbose=True,
    allow_delegation=True,
    llm=llm  
    )

    content_organizer = Agent(
    role="Organizzatore di contenuti",
    goal=f"Per ognuno degli {x} temi individuati, estrapolare tutte le parti del testo che fanno riferiemento ad ogni tema.",
    backstory="Esperto nella organizzazione di contenuti e nell'analisi di testi complessi.",
    verbose=True,
    allow_delegation=True,
    llm=llm
    )
    
    quiz_creator = Agent(
    role="Costruttore di Quiz",
    goal=f"Creare {y} domande per ognuno degli {x} temi individuati, con 4 opzioni di risposta: una corretta, una parzialmente corretta, una errata, una errata e dannosa. Rispondi sempre e solo in italiano.",
    backstory="Esperto nella creazione di quiz in base a temi specifici e contenuti testuali.",
    verbose=True,
    allow_delegation=True,
    llm=llm
    )

    answer_evaluator = Agent(
    role="Valutatore Risposte",
    goal="Assegnare un punteggio tra quelli disponibili a ciascuna opzione di risposta in base al grado di correttezza: 5 per risposte corrette, 2 per risposte parzialmente corrette, 0 per risposte errate, -5 per risposte errate e dannose.",
    backstory="Esperto nella valutazione di domande a scelta multipla e nell'assegnazione di punteggi.",
    verbose=True,
    allow_delegation=True,
    llm=llm
    )

    copy_editor = Agent(
        role="Copy-Editor",
        goal="Rivedere il testo delle domande e delle risposte senza modificarne il significato, ma garantendo che: "
             "- Ogni domanda e opzione di risposta abbia una lunghezza inferiore a 250 caratteri."
             "- Le opzioni di risposta abbiano lunghezze simili per evitare squilibri visivi.",
        backstory="Esperto in revisione di contenuti didattici e uniformità linguistica.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return analyst, content_organizer, quiz_creator, answer_evaluator, copy_editor

def create_crew(use_quiz,x,y, pdf_text=None):
    """Crea il CrewAI e definisce i task in base alla scelta dell'utente."""
    
    analyst, content_organizer, quiz_creator, answer_evaluator, copy_editor = create_agents(use_quiz,x,y, pdf_text)

    if use_quiz:
        extract_themes_task = Task(
        description=f"Analizza il contenuto del PDF {pdf_text} e identifica i {x} temi più importanti.",
        agent=analyst,
        expected_output=f"Elenco di {x} temi in italiano"
        )

        organize_themes_task = Task(
        description=f"""Per ognuno dei {x} temi individuati, estrapola tutte le parti del testo relative ogni tema""",
        agent=content_organizer,
        depends_on=[extract_themes_task],
        expected_output=f"Testo riorganizzato in {x} temi"
        )

        generate_questions_task = Task(
        description=f"""Per difficoltà di una domanda si intende il livello di somiglianza tra le risposte. Per ognuno dei {x} temi individuati, genera esattamente {y} domande a difficoltà crescente in italiano in base alle informazioni contenute nel testo estrapolato per ogni tema.  
    Ogni domanda deve avere **esattamente** 4 opzioni di risposta, in italiano:
    - ✅ Una corretta (5 punti)
    - ⚠️ Una parzialmente corretta (2 punti)
    - ❌ Una errata (0 punti)
    - ❌❌ Una errata e potenzialmente dannosa (-5 punti)   
    **NON generare meno di {y} domande per tema** 
    Restituisci il risultato in formato CSV con queste colonne:  
    - **Tema**  
    - **Domanda**  
    - **Risposta 1**, **Punteggio 1**  
    - **Risposta 2**, **Punteggio 2**  
    - **Risposta 3**, **Punteggio 3**  
    - **Risposta 4**, **Punteggio 4**""",
        agent=quiz_creator,
        depends_on=[organize_themes_task] , # Dipende dall'estrazione dei temi
        expected_output=f"Elenco di {x} temi con **{y} domande per ogni tema** e **4 opzioni di risposta**."
        )

        score_answers_task = Task(
        description="""Assegna un punteggio a ogni opzione di risposta secondo le seguenti regole:
    - Ogni risposta deve avere uno dei seguenti punteggi:  
        - **5 punti se è corretta**  
        - **2 punti se è parzialmente corretta**  
        - **0 punti se è errata**  
        - **-5 punti se è errata e potenzialmente dannosa**  
    - Per una stessa domanda, non ci possono essere opzioni di risposta con lo stesso punteggio
    Restituisci il risultato in formato CSV con queste colonne:  
    - **Tema**  
    - **Domanda**  
    - **Risposta 1**, **Punteggio 1**  
    - **Risposta 2**, **Punteggio 2**  
    - **Risposta 3**, **Punteggio 3**  
    - **Risposta 4**, **Punteggio 4**""",
        agent=answer_evaluator,
        depends_on=[generate_questions_task] , # Dipende dalla generazione delle domande
        expected_output=f"Elenco in italiano di {x} temi e per ogni tema un elenco di {y} domande, 4 opzioni di risposta e un punteggio per ogni opzione."
        )

        review_texts_task = Task(
        description="""Rivedi i testi delle domande e delle risposte generate, garantendo che:  
        - Ogni testo di domanda e opzione di risposta abbia una lunghezza inferiore a 400 caratteri.  
        - Le opzioni di risposta abbiano lunghezze simili tra loro.  
        **Non modificare il significato delle domande, il significato delle risposte e i punteggi**.
        Restituisci il risultato in formato CSV con queste colonne:  
        - **Tema**  
        - **Domanda**  
        - **Risposta 1**, **Punteggio 1**  
        - **Risposta 2**, **Punteggio 2**  
        - **Risposta 3**, **Punteggio 3**  
        - **Risposta 4**, **Punteggio 4**
        **Assicurati di avere esattamente {x} temi, {y} domande, 4 opzioni di risposte per domande e un punteggio per ogni opzione di risposta""",
        agent=copy_editor,
        depends_on=[score_answers_task],
        expected_output=f"Elenco ottimizzato in lunghezza in italiano di {x} temi e per ogni tema un elenco di {y} domande, 4 opzioni di risposta e un punteggio per ogni opzione."
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

        review_texts_task = Task(
        description="TBD per storie",
        agent=analyst,
        expected_output="TBD per storie"
        ) 
        
    crew = Crew(
        agents=[analyst, content_organizer, quiz_creator, answer_evaluator,copy_editor],
        tasks=[extract_themes_task ,organize_themes_task, generate_questions_task, score_answers_task,review_texts_task]
    )

    return crew

# --- INTERFACCIA STREAMLIT ---
st.title("🤖 AI Collaborativa con CrewAI e Streamlit")

# Selettore per la modalità di ricerca
option = st.radio("Scegli tra:", ["Quiz", "Storie"])

x= st.select_slider("Seleziona il numero di temi", options=[1,2,3,4,5])
y=st.select_slider("Seleziona il numero di domande per tema",options=[1,2,3,4,5,6,7,8,9,10])

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
        crew = create_crew(use_quiz,x,y, pdf_text)
    
        st.write("### 🚀 Elaborazione della risposta...")
        result = crew.kickoff()
    
        st.subheader("📝 Risposta Generata")
        st.write(result)
