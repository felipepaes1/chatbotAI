import os
import streamlit as st
from io import BytesIO
from fpdf import FPDF
from environs import Env
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from groq import Groq   # usado apenas para validar a key


env = Env()
env.read_env()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY não encontrado no .env")

# Valida a key (opcional, mas útil para erro rápido)
_ = Groq(api_key=GROQ_API_KEY)

chat = ChatGroq(temperature=0, model_name="llama3-70b-8192")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um expert em gestão de produção e usinagem."),
    ("human", "{text}")
])


chain = prompt | chat


def generate_pdf(messages) -> BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for msg in messages:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        text = f"{role}: {msg['content']}\n"
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.ln(2)

    buffer = BytesIO()
    # FPDF retorna string, convertemos p/ bytes
    buffer.write(pdf.output(dest="S").encode("latin-1"))
    buffer.seek(0)
    return buffer

def container_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe histórico apenas para visualização
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada do usuário
    if user_input := st.chat_input("Pergunte algo ao assistente…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Streaming de resposta (sem histórico enviado ao modelo)
        response_container = st.chat_message("assistant")
        response_text      = response_container.empty()

        full_response = ""
        for chunk in chain.stream({"text": user_input}):
            full_response += chunk.content
            response_text.markdown(full_response + "▌")

        response_text.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def streamlit_visual():
    st.set_page_config(page_title="Assistente de Produção", layout="wide")

    st.header("🤖 Assistente de Produção — Log Z Tech")
    container_chat()

    st.sidebar.title("⚙️ Opções")

    if st.sidebar.button("🗑️ Limpar chat"):
        st.session_state.messages = []

    if st.session_state.get("messages"):
        pdf_bytes = generate_pdf(st.session_state.messages)
        st.sidebar.download_button(
            label="📄 Exportar chat (PDF)",
            data=pdf_bytes,
            file_name="conversa_chatbot_logz.pdf",
            mime="application/pdf",
        )

def main():
    streamlit_visual()

if __name__ == "__main__":
    main()