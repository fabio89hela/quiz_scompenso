__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import pdfplumber
import openai
import pandas as pd
from crewai import Crew, Agent, Task
from io import BytesIO

# ✅ Usa la chiave OpenAI dai secrets di Streamlit
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Streamlit UI
st.title("📚 Generatore di Quiz da PDF")
st.write("Carica i documenti PDF e genera un quiz con tematiche estratte automaticamente.")

# 🚀 Upload PDF
uploaded_files = st.file_uploader("Carica i PDF", type=["pdf"], accept_multiple_files=True)

# 🔢 Selezione numero di temi e domande
x_temi = st.slider("Numero di temi", 1, 20, 10)
y_domande = st.slider("Numero di domande", 1, 20, 10)

# 🎯 Selezione della difficoltà
difficolta = st.selectbox("Scegli la difficoltà", ["Facile", "Intermedio", "Difficile"])

# 🤖 Scelta del modello OpenAI
modello_openai = st.selectbox("Modello AI", ["gpt-3.5-turbo", "gpt-4-turbo"], index=0)

# 🚀 Bottone per generare il quiz
if st.button("Genera Quiz"):
    with st.spinner("Analizzando i documenti e generando il quiz..."):

        # 📝 Estrazione testo dai PDF
        def extract_text_from_pdfs(pdf_files):
            text = ""
            for pdf in pdf_files:
                with pdfplumber.open(pdf) as pdf_reader:
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            return text

        testo_completo = extract_text_from_pdfs(uploaded_files)

        # ✅ Agente 1: Identificazione Temi
        theme_agent = Agent(
            name="Theme Extractor",
            role="Identifica i temi principali dai documenti PDF.",
            goal=f"Identificare {x_temi} temi principali basandosi sul contenuto dei documenti.",
            model=modello_openai,
            memory=False,  
            backstory="Esperto analista di documenti con capacità avanzate di identificazione dei temi principali."
        )

        extract_themes_task = Task(
            description=f"Analizza il testo e individua i {x_temi} temi più rilevanti.",
            agent=theme_agent,
            expected_output=f"Un elenco di {x_temi} temi principali estratti dai documenti."
        )

        # ✅ Agente 2: Generazione Domande
        question_agent = Agent(
            name="Question Generator",
            role="Genera domande su ogni tema con risposte bilanciate.",
            goal=f"Creare {y_domande} domande con risposte e punteggi bilanciati.",
            model=modello_openai,
            memory=False,  
            backstory="Specialista nella creazione di quiz educativi, con particolare attenzione alla validità scientifica delle risposte."
        )

        generate_questions_task = Task(
            description=(
                f"Per ogni tema, genera {y_domande} domande con 4 opzioni:"
                " - Una risposta deve essere completamente corretta (5 punti)."
                " - Una risposta deve essere parzialmente corretta (2 punti)."
                " - Una risposta deve essere errata ma non dannosa (0 punti)."
                " - Una risposta deve essere errata e completamente controproducente (-5 punti)."
            ),
            agent=question_agent,
            context=extract_themes_task,
            expected_output=f"{y_domande} domande con 4 risposte ciascuna e punteggi correttamente assegnati."
        )

        # ✅ CrewAI: Esecuzione senza memoria (senza ChromaDB)
        crew = Crew(
            agents=[theme_agent, question_agent],
            tasks=[extract_themes_task, generate_questions_task],
            memory=False  
        )

        # 🚨 Verifica del formato del risultato
        result = crew.kickoff()

        # 🔍 DEBUG: Mostra il tipo di `result`
        st.write(f"🔍 Debug: tipo di result = {type(result)}")

        # 🚨 Se il risultato è una tupla, prendiamo solo il primo elemento
        if isinstance(result, tuple):
            result = result[0]
            st.write("📌 Il risultato era una tupla, è stato convertito.")

        # 🚨 Se il risultato non è un dizionario, mostriamo errore
        if not isinstance(result, dict):
            st.error("❌ Errore: il formato del risultato non è valido. Output ricevuto:")
            st.write(result)  # Stampa l'output ricevuto per debug
            st.stop()

        # 📊 Creazione DataFrame per output
        quiz_data = []
        for tema, domande in result.items():
            for domanda in domande:
                quiz_data.append([
                    tema,
                    domanda.get("testo", ""),
                    domanda.get("opzioni", [{}])[0].get("testo", ""), domanda.get("opzioni", [{}])[0].get("punteggio", ""),
                    domanda.get("opzioni", [{}])[1].get("testo", ""), domanda.get("opzioni", [{}])[1].get("punteggio", ""),
                    domanda.get("opzioni", [{}])[2].get("testo", ""), domanda.get("opzioni", [{}])[2].get("punteggio", ""),
                    domanda.get("opzioni", [{}])[3].get("testo", ""), domanda.get("opzioni", [{}])[3].get("punteggio", ""),
                ])

        df = pd.DataFrame(quiz_data, columns=[
            "Tematica", "Domanda",
            "Risposta 1", "Punteggio 1",
            "Risposta 2", "Punteggio 2",
            "Risposta 3", "Punteggio 3",
            "Risposta 4", "Punteggio 4"
        ])

        # 🏁 Download del file Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        output.seek(0)

        st.success("✅ Quiz generato con successo!")
        st.download_button(
            label="📥 Scarica il Quiz in Excel",
            data=output,
            file_name="quiz_scompenso_cardiaco.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
