import os
import streamlit as st
from io import BytesIO
from fpdf import FPDF
from environs import Env
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from groq import Groq   # usado apenas para validar a key
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics, ttfonts
from io import BytesIO
import re, textwrap, html


env = Env()
env.read_env()


GROQ_API_KEY = env.str("GROQ_API_KEY", default=None)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY não encontrado no .env")
_ = Groq(api_key=GROQ_API_KEY)

chat = ChatGroq(temperature=0, model_name="llama3-70b-8192")


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é a UziBot. Especialista em gestão de produção e usinagem. "
     "Você é um assistente de uma plataforma de gestão de ferramentas de corte"
     "Sempre responda em português técnico, usando dados numéricos em mm ou %, "
     "sem jargões de TI. Se a pergunta for ambígua, peça esclarecimentos. "
     "Se não souber, responda 'Não encontrado'."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{text}")
])

memory = ConversationBufferWindowMemory(
    k=10,                  # últimas 10 mensagens (ajuste se quiser)
    return_messages=True   # precisa ser True por causa do placeholder
)

base_chain = prompt | chat

# ── 2. Fábrica de memórias por sessão ───────────────────────
memories: dict[str, ConversationBufferWindowMemory] = {}

def get_session_history(session_id: str):
    """Usa st.session_state para persistir o histórico durante toda a conversa."""
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferWindowMemory(
            k=10, return_messages=True
        ).chat_memory
    return st.session_state.chat_memory

chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="text",
    history_messages_key="history",
)


if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(ttfonts.TTFont("DejaVu", "DejaVuSans.ttf"))

# ── estilos base ─────────────────────────────────────────────
styles = getSampleStyleSheet()
base = styles["Normal"]
base.fontName = "DejaVu"
base.fontSize = 11
base.leading = 15

user_style = ParagraphStyle(
    "user", parent=base, textColor=colors.HexColor("#0B5394"), spaceAfter=4
)
assistant_style = ParagraphStyle(
    "assistant", parent=base, textColor=colors.HexColor("#38761D"), spaceAfter=4
)
mono_style = ParagraphStyle(
    "mono", parent=base, fontName="Courier", backColor="#F3F3F3",
    borderColor="#DDDDDD", borderWidth=0.5, borderPadding=4, spaceAfter=6
)

# ── helpers ──────────────────────────────────────────────────
BOLD_RX = re.compile(r"\*\*(.+?)\*\*", flags=re.S)

def to_html(txt: str) -> str:
    """Converte '**negrito**' e preserva quebras de linha."""
    txt = html.escape(txt)               # &, <, >
    txt = BOLD_RX.sub(r"<b>\1</b>", txt)
    return txt.replace("\n", "<br/>")

LIST_RX = re.compile(r"^\s*[\*\-]\s+", flags=re.M)

def split_bullets(chunk: str):
    """Separa texto normal e itens de lista (* item)"""
    parts = LIST_RX.split(chunk)
    heads = LIST_RX.findall(chunk)
    if not heads:     # nenhum bullet
        return None
    items = [p.strip() for p in parts[1:]]  # primeira parte antes do 1º bullet é ''
    return items

# ── função principal ─────────────────────────────────────────
def generate_pdf(messages) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=35, rightMargin=35, topMargin=35, bottomMargin=35
    )

    flow = []
    for msg in messages:
        role_lbl = "Usuário:" if msg["role"] == "user" else "Assistente:"
        role_style = user_style if msg["role"] == "user" else assistant_style
        flow.append(Paragraph(role_lbl, role_style))

        content = msg["content"]

        # — trata blocos de código ```
        for i, part in enumerate(re.split(r"```", content)):
            if i % 2:   # bloco de código
                flow.append(Preformatted(part.strip(), mono_style))
                continue

            # — trata listas com '*' na parte "normal"
            bullets = split_bullets(part)
            if bullets:
                for it in bullets:
                    bullet_par = "&#8226;&nbsp;" + to_html(it)  # &#8226; = •
                    flow.append(Paragraph(bullet_par, base))
            else:
                plaintext = part.strip()
                if plaintext:
                    flow.append(Paragraph(to_html(plaintext), base))
        flow.append(Spacer(1, 6))

    doc.build(flow)
    buffer.seek(0)
    return buffer

session_id = "streamlit-session"
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
        for chunk in chat_chain.stream(
                {"text": user_input},
                config={"configurable": {"session_id": session_id}}
        ):
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