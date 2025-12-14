import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import io
import os

load_dotenv()

st.set_page_config(page_title="ИИ-агент", layout="wide", page_icon="🤖")

st.title("🤖 ИИ-агент для документов (OpenRouter)")

# OpenRouter клиент (правильный способ)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Создайте `.env`:\n`OPENAI_API_KEY=sk-or-v1-...`")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

st.success("✅ OpenRouter подключён!")

# Sidebar модели
with st.sidebar:
    model = st.selectbox("Модель:",
                        ["openai/gpt-4o-mini",
                         "meta-llama/llama-3.3-70b-instruct:free"])

# Загрузка файлов
uploaded_files = st.file_uploader("📤 PDF/TXT",
                                  type=['pdf','txt'],
                                  accept_multiple_files=True)

if uploaded_files:
    docs_text = ""
    for file in uploaded_files:
        st.success(f"✅ {file.name}")
        if file.name.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages[:10]:
                text += page.extract_text() + "\n"
            docs_text += f"\n\n=== {file.name} ===\n{text}"
        file.seek(0)  # Reset для повторного чтения
        docs_text += f"\n\n=== {file.name} ===\n{file.read().decode('utf-8', errors='ignore')}"

    col1, col2 = st.columns([3,1])
    with col1:
        question = st.text_area("❓ Вопрос:",
                               placeholder="Составь родословную Колобковых?")
    with col2:
        st.empty()

    if st.button("🚀 Спросить ИИ", type="primary") and question.strip():
        with st.spinner("🤖 Анализирует..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content":
                         "Ты эксперт по документам. Отвечай КРАТКО и ТОЧНО по тексту."},
                        {"role": "user", "content": f"ДОКУМЕНТ:\n{docs_text[:8000]}\n\nВОПРОС: {question}"}
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"❌ {str(e)}"

            st.markdown("---")
            st.subheader("📄 Ответ ИИ:")
            st.markdown(answer)
            st.balloons()

else:
    st.info("📤 Загрузите PDF → задайте вопрос → ИИ ответит!")
