__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import OpenAI

# Configura la chiave API di OpenAI
OPENAI_API_KEY = "your-api-key-here"

def create_agents():
    """Crea agenti con ruoli specifici."""
    researcher = Agent(
        role="Ricercatore AI",
        goal="Trovare informazioni affidabili per rispondere a una domanda.",
        backstory="Un esperto di ricerca su internet e database accademici.",
        verbose=True,
        allow_delegation=True,
        llm=OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    )

    writer = Agent(
        role="Redattore AI",
        goal="Scrivere una risposta chiara e strutturata basata sulla ricerca.",
        backstory="Un esperto in scrittura tecnica e comunicazione chiara.",
        verbose=True,
        allow_delegation=False,
        llm=OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    )

    reviewer = Agent(
        role="Revisore AI",
        goal="Verificare la qualità e la chiarezza della risposta finale.",
        backstory="Un editor attento ai dettagli che migliora la leggibilità del testo.",
        verbose=True,
        allow_delegation=False,
        llm=OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    )

    return researcher, writer, reviewer

def create_crew(researcher, writer, reviewer, user_question):
    """Crea il CrewAI e definisce i task."""
    research_task = Task(
        description=f"Ricerca informazioni affidabili su: {user_question}",
        agent=researcher
    )
    
    writing_task = Task(
        description="Scrivi una risposta ben strutturata basata sulla ricerca effettuata.",
        agent=writer,
        depends_on=[research_task]
    )
    
    review_task = Task(
        description="Migliora la leggibilità e correggi eventuali errori nella risposta.",
        agent=reviewer,
        depends_on=[writing_task]
    )
    
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task]
    )
    
    return crew

# Streamlit UI
st.title("🤖 AI Collaborativa con CrewAI e Streamlit")
user_question = st.text_input("Inserisci una domanda:")

if user_question:
    researcher, writer, reviewer = create_agents()
    crew = create_crew(researcher, writer, reviewer, user_question)
    
    st.write("### 🚀 Elaborazione della risposta...")
    result = crew.kickoff()
    
    st.subheader("📝 Risposta Generata")
    st.write(result)
