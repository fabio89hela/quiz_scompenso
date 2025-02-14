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
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            return text

        # ✅ Otteniamo il testo reale dai PDF
        testo_completo = extract_text_from_pdfs(uploaded_files)
        st.write(testo_completo)

        if not testo_completo.strip():
            st.error("Errore: Il testo estratto dai PDF è vuoto. Assicurati che i PDF contengano testo selezionabile.")
            st.stop()

        # ✅ Agente 1: Identificazione Temi basati sul PDF
        theme_agent = Agent(
            name="Theme Extractor",
            role="Analizza i PDF e identifica i temi principali basandosi esclusivamente sul loro contenuto.",
            goal=f"Identificare {x_temi} temi principali presenti nei documenti caricati.",
            model=modello_openai,
            memory=False,
            backstory="Esperto nell'analisi testuale e nella sintesi delle informazioni. Lavora solo sul contenuto fornito.",
            instructions=(
                f"Estrarre i {x_temi} temi principali presenti nei documenti forniti. "
                "NON inventare temi, usa solo le informazioni effettivamente contenute nei PDF. "
                "Fornisci l'elenco dei temi in un formato chiaro e conciso."
            ),
            context=testo_completo  # 📌 Passiamo il testo dei PDF come contesto
        )

        extract_themes_task = Task(
            description=f"Analizza il testo dei PDF e individua i {x_temi} temi più rilevanti.",
            agent=theme_agent,
            expected_output=f"Un elenco di {x_temi} temi estratti dai documenti PDF."
        )

        # ✅ Agente 2: Generazione Domande basate sui PDF
        question_agent = Agent(
            name="Question Generator",
            role="Genera domande e risposte basate esclusivamente sul contenuto dei documenti PDF.",
            goal=f"Creare {y_domande} domande con risposte e punteggi bilanciati.",
            model=modello_openai,
            memory=False,
            backstory="Esperto nella creazione di quiz educativi. Utilizza esclusivamente il materiale fornito.",
            instructions=(
                f"Per ogni tema fornito, crea {y_domande} domande con 4 opzioni di risposta. "
                "Le domande devono essere basate esclusivamente sul testo dei documenti PDF e NON devono essere inventate. "
                "Assegna i punteggi come segue: "
                " - Una risposta deve essere completamente corretta (5 punti). "
                " - Una risposta deve essere parzialmente corretta (2 punti). "
                " - Una risposta deve essere errata ma non dannosa (0 punti). "
                " - Una risposta deve essere errata e completamente controproducente (-5 punti). "
                "Fornisci l'output in formato strutturato."
            ),
            context=testo_completo  # 📌 Passiamo il testo dei PDF come contesto
        )

        generate_questions_task = Task(
            description=f"Genera {y_domande} domande per ogni tema, con 4 risposte e punteggi bilanciati.",
            agent=question_agent,
            context=[extract_themes_task],
            expected_output=f"{y_domande} domande basate sui PDF con risposte strutturate e punteggi corretti."
        )

        # ✅ CrewAI: Esecuzione senza memoria (senza ChromaDB)
        crew = Crew(
            agents=[theme_agent, question_agent],
            tasks=[extract_themes_task, generate_questions_task],
            memory=False
        )

        result = crew.kickoff()
        st.write(result)

        # 📊 Creazione DataFrame per output
        quiz_data = []
        if isinstance(result, list):
            for domanda in result:
                quiz_data.append([
                    domanda.get("tema", "Tema sconosciuto"),
                    domanda.get("testo", "Domanda non disponibile"),
                    domanda["opzioni"][0]["testo"], domanda["opzioni"][0]["punteggio"],
                    domanda["opzioni"][1]["testo"], domanda["opzioni"][1]["punteggio"],
                    domanda["opzioni"][2]["testo"], domanda["opzioni"][2]["punteggio"],
                    domanda["opzioni"][3]["testo"], domanda["opzioni"][3]["punteggio"],
                ])
        else:
            st.error("Errore: il formato dell'output non è valido.")

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
