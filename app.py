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
def cerca_su_web(query: str) -> str:
    """Esegue una ricerca su DuckDuckGo e restituisce i primi risultati."""
    with DDGS() as ddgs:
        risultati = [f"{r['title']} - {r['href']}" for r in ddgs.text(query, max_results=3)]
    return "\n".join(risultati)

# Creiamo un Tool valido per CrewAI
search_tool = Tool(
    name="Ricerca Web",
    func=lambda query: cerca_su_web(query),
    description="Usa DuckDuckGo per trovare informazioni aggiornate su un argomento."
)

# Funzione per leggere il testo da un PDF
def estrai_testo_da_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pagine = loader.load()
    testo_completo = "\n".join([pagina.page_content for pagina in pagine])
    st.write(testo_completo)
    return testo_completo

def create_agents(use_web, pdf_text=None):
    """Crea agenti CrewAI per Ricerca Web o Analisi PDF"""
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Puoi usare "gpt-4"
        openai_api_key=OPENAI_API_KEY
    )

    researcher = Agent(
        role="Ricercatore AI",
        goal="Trovare informazioni aggiornate utilizzando il web" if use_web else "Leggere e analizzare il documento PDF",
        backstory="Un esperto di ricerca su internet" if use_web else "Un esperto nella lettura e comprensione di documenti complessi",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool] if use_web else [],  # Se è web usa il Tool, altrimenti no
        llm=llm
    )

    writer = Agent(
        role="Redattore AI",
        goal="Scrivere una risposta chiara e strutturata basata sulla ricerca" if use_web else "Riassumere e spiegare i contenuti del documento PDF",
        backstory="Un esperto in scrittura tecnica e comunicazione chiara.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    reviewer = Agent(
        role="Revisore AI",
        goal="Verificare la qualità e la chiarezza della risposta finale.",
        backstory="Un editor attento ai dettagli che migliora la leggibilità del testo.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return researcher, writer, reviewer

def create_crew(use_web, user_question, pdf_text=None):
    """Crea il CrewAI e definisce i task in base alla scelta dell'utente."""
    
    researcher, writer, reviewer = create_agents(use_web, pdf_text)

    if use_web:
        research_task = Task(
            description=f"Ricerca informazioni affidabili su: {user_question}",
            agent=researcher,
            expected_output="Ricerca strutturata e dettagliata"
        )
    else:
        research_task = Task(
            description="Analizza il contenuto del PDF e rispondi alla domanda dell'utente basandoti su di esso.",
            agent=researcher,
            expected_output="Descrizione dettagliata del PDF"
        )

    writing_task = Task(
        description="Scrivi una risposta dettagliata e ben strutturata basata sulla ricerca." if use_web else "Crea un riassunto basato sul contenuto del PDF.",
        agent=writer,
        depends_on=[research_task],
        expected_output="Testo chiaro ed autoesplicativo"
    )

    review_task = Task(
        description="Migliora la leggibilità e correggi eventuali errori nella risposta.",
        agent=reviewer,
        depends_on=[writing_task],
        expected_output="Testo chiaro ed esplicativo"
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task]
    )

    return crew

# --- INTERFACCIA STREAMLIT ---
st.title("🤖 AI Collaborativa con CrewAI e Streamlit")

# Selettore per la modalità di ricerca
option = st.radio("Scegli la fonte delle informazioni:", ["🔍 Ricerca Web", "📄 Analisi PDF"])

user_question = st.text_input("Inserisci una domanda:")

pdf_text = None  # Inizializza la variabile per il testo del PDF

if option == "📄 Analisi PDF":
    uploaded_file = st.file_uploader("Carica un file PDF", type="pdf")
    
    if uploaded_file is not None:
        pdf_path = os.path.join("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pdf_text = estrai_testo_da_pdf(pdf_path)  # Estrai il testo dal PDF
        os.remove(pdf_path)  # Cancella il file temporaneo dopo la lettura

if user_question and (option == "🔍 Ricerca Web" or (option == "📄 Analisi PDF" and pdf_text)):
    use_web = option == "🔍 Ricerca Web"
    
    crew = create_crew(use_web, user_question, pdf_text)
    
    st.write("### 🚀 Elaborazione della risposta...")
    result = crew.kickoff()
    
    st.subheader("📝 Risposta Generata")
    st.write(result)
