import os
import streamlit as st
import pdfplumber
import openai
import pandas as pd
from crewai import Crew, Agent, Task
from io import BytesIO

# 🚨 Forza CrewAI a NON usare ChromaDB o SQLite
os.environ["CREWAI_MEMORY_BACKEND"] = "no_memory"

# ✅ Configura OpenAI tramite le Secrets di Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Interfaccia Streamlit
st.title("📚 Generatore di Quiz da PDF")
st.write("Carica i documenti PDF e genera un quiz con tematiche estratte automaticamente.")

# 🚀 Upload dei file PDF
uploaded_files = st.file_uploader("Carica i PDF", type=["pdf"], accept_multiple_files=True)

# 🔢 Selezione X (temi) e Y (domande)
x_temi = st.slider("Numero di temi", 1, 20, 10)
y_domande = st.slider("Numero di domande", 1, 20, 10)

# 🎯 Scelta della difficoltà
difficolta = st.selectbox("Scegli la difficoltà", ["Facile", "Intermedio", "Difficile"])

# 🤖 Modello OpenAI
modello_openai = st.selectbox("Modello AI", ["gpt-3.5-turbo", "gpt-4-turbo"], index=0)

# 🚀 Bottone per avviare il processo
if st.button("Genera Quiz"):
    with st.spinner("Analizzando i documenti e generando il quiz..."):

        # 📝 Funzione per estrarre testo dai PDF
        def extract_text_from_pdfs(pdf_files):
            text = ""
            for pdf in pdf_files:
                with pdfplumber.open(pdf) as pdf_reader:
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            return text

        # 📝 Estrazione testo
        testo_completo = extract_text_from_pdfs(uploaded_files)

        # ✅ Agente 1: Identificazione Temi
        theme_agent = Agent(
            name="Theme Extractor",
            role="Identifica i temi principali dai documenti PDF.",
            goal=f"Identificare {x_temi} temi principali basandosi sul contenuto.",
            model=modello_openai
        )

        extract_themes_task = Task(
            description=f"Analizza il testo e individua i {x_temi} temi più rilevanti.",
            agent=theme_agent
        )

        # ✅ Agente 2: Generazione Domande
        question_agent = Agent(
            name="Question Generator",
            role="Genera domande su ogni tema con risposte bilanciate.",
            goal=f"Creare {y_domande} domande con 4 risposte e punteggi bilanciati.",
            model=modello_openai
        )

        generate_questions_task = Task(
            description=(
                f"Per ogni tema, genera {y_domande} domande con 4 opzioni:"
                " - Una risposta deve essere corretta (5 punti)."
                " - Una risposta deve essere parzialmente corretta (2 punti)."
                " - Una risposta deve essere errata ma non dannosa (0 punti)."
                " - Una risposta deve essere errata e completamente controproducente (-5 punti)."
            ),
            agent=question_agent,
            context=extract_themes_task
        )

        # ✅ CrewAI: Avvio
        crew = Crew(
            agents=[theme_agent, question_agent],
            tasks=[extract_themes_task, generate_questions_task]
        )

        result = crew.kickoff()

        # 📊 Creazione DataFrame
        quiz_data = []
        for tema, domande in result.items():
            for domanda in domande:
                quiz_data.append([
                    tema,
                    domanda["testo"],
                    domanda["opzioni"][0]["testo"], domanda["opzioni"][0]["punteggio"],
                    domanda["opzioni"][1]["testo"], domanda["opzioni"][1]["punteggio"],
                    domanda["opzioni"][2]["testo"], domanda["opzioni"][2]["punteggio"],
                    domanda["opzioni"][3]["testo"], domanda["opzioni"][3]["punteggio"],
                ])

        df = pd.DataFrame(quiz_data, columns=[
            "Tematica", "Domanda",
            "Risposta 1", "Punteggio 1",
            "Risposta 2", "Punteggio 2",
            "Risposta 3", "Punteggio 3",
            "Risposta 4", "Punteggio 4"
        ])

        # 🏁 Download Excel
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
