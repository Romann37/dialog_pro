import os
import json
import asyncio
import logging
import shutil
import aiofiles

from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from functools import lru_cache

import streamlit as st
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI, APIError
from docx import Document
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS  # пока не используется, можно добавить веб-поиск
import easyocr
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CONFIG = {
    "app_name": "AI Document Assistant Pro",
    "max_context_tokens": 4000,  # здесь это лимит по символам
    "chunk_size": 500,
    "top_k": 3,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "openrouter_model": "meta-llama/llama-3.2-3b-instruct:free",
    "documents_dir": "documents",
    "answers_dir": "Ответы ИИ-агента",
    "db_path": "knowledge_base_pro.db",
    "poppler_path": shutil.which("pdftoppm"),  # автоопределение poppler
    "retry_attempts": 3,
    "ocr_cache": True,
}


def load_config() -> dict:
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        CONFIG.update(user_config)

    os.makedirs(CONFIG["documents_dir"], exist_ok=True)
    os.makedirs(CONFIG["answers_dir"], exist_ok=True)
    return CONFIG


CONFIG = load_config()


class OpenRouterClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_id = os.getenv("ID_MODEL", CONFIG["openrouter_model"])
        if not self.api_key:
            logging.warning("OPENAI_API_KEY не задан, запросы к OpenRouter не будут работать.")
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    async def ask(
        self,
        question: str,
        context: str,
        max_retries: int = CONFIG["retry_attempts"],
    ) -> str:
        if not self.api_key:
            return "❌ Не задан ключ OPENAI_API_KEY для OpenRouter."

        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Ты профессиональный ассистент для анализа документов. "
                            "Отвечай кратко и точно на русском языке."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Контекст документов:\n{context}\n\nВопрос: {question}",
                    },
                ]

                response = await asyncio.to_thread(
                    lambda: self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                    )
                )
                return response.choices[0].message.content
            except APIError as e:
                logging.error(f"Ошибка OpenRouter API: {e}")
                if attempt == max_retries - 1:
                    return f"❌ Ошибка API (попытка {attempt + 1}): {str(e)}"
                await asyncio.sleep(2 ** attempt)

        return "❌ Не удалось получить ответ"


client = OpenRouterClient()


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(CONFIG["embedding_model"])


class VectorKnowledgeBase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.model = load_embedding_model()
        self.init_db()
        self.chunks, self.embeddings = self._load_chunks()

    def init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                chunk_text TEXT,
                chunk_index INTEGER,
                doc_type TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def _chunk_text(self, text: str, chunk_size: int = CONFIG["chunk_size"]) -> List[str]:
        words = text.split()
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def add_document(self, filename: str, content: str, doc_type: str) -> None:
        chunks = self._chunk_text(content)
        if not chunks:
            logging.warning(f"Документ {filename} пустой, не добавлен в базу.")
            return

        embeddings = self.model.encode(chunks)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks (filename, chunk_text, chunk_index, doc_type, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (filename, chunk, i, doc_type, emb.tobytes()),
            )

        conn.commit()
        conn.close()
        self.chunks, self.embeddings = self._load_chunks()
        logging.info(f"Документ {filename} добавлен в базу: {len(chunks)} чанков.")

    def _load_chunks(self) -> Tuple[List[str], np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_text, embedding FROM chunks")
        chunks: List[str] = []
        embeddings: List[np.ndarray] = []

        for chunk_text, emb_bytes in cursor.fetchall():
            chunks.append(chunk_text)
            embeddings.append(np.frombuffer(emb_bytes, dtype=np.float32))

        conn.close()

        if embeddings:
            return chunks, np.array(embeddings)
        return [], np.array([])

    def search(self, query: str, top_k: int = CONFIG["top_k"]) -> str:
        if not self.chunks or self.embeddings.size == 0:
            return "База знаний пуста"

        query_emb = self.model.encode([query])
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        context = "\n\n".join([self.chunks[i] for i in top_indices])

        # Здесь max_context_tokens используется как лимит по символам
        return context[: CONFIG["max_context_tokens"]]


kb = VectorKnowledgeBase(CONFIG["db_path"])


@lru_cache(maxsize=128)
def get_ocr_reader():
    return easyocr.Reader(["ru", "en"], gpu=False)


async def process_document(filepath: str) -> Tuple[str, str]:
    """Возвращает (content, doc_type)"""
    filename = Path(filepath).name
    cache_key = f"ocr_{filename}"

    if hasattr(process_document, "cache") and cache_key in process_document.cache:
        return process_document.cache[cache_key]

    if filepath.lower().endswith(".txt"):
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
        doc_type = "txt"

    elif filepath.lower().endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            content = " ".join(page.extract_text() or "" for page in reader.pages)
            doc_type = "pdf"
        except Exception:
            ocr_reader = get_ocr_reader()
            images = convert_from_path(filepath, poppler_path=CONFIG.get("poppler_path"))
            content = " ".join(" ".join(d[1] for d in ocr_reader.readtext(img)) for img in images)
            doc_type = "pdf"

    else:
        ocr_reader = get_ocr_reader()
        result = ocr_reader.readtext(filepath)
        content = " ".join(detection[1] for detection in result)
        doc_type = "image"

    if CONFIG["ocr_cache"]:
        process_document.cache = getattr(process_document, "cache", {})
        process_document.cache[cache_key] = (content, doc_type)

    return content, doc_type


def save_answer_docx(question: str, answer: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(CONFIG["answers_dir"]) / f"answer_{timestamp}.docx"

    doc = Document()
    doc.add_heading("🤖 AI Document Assistant Pro", level=0)
    doc.add_heading(f"❓ Вопрос: {question}", level=1)
    doc.add_paragraph(answer)
    doc.save(filename)

    return str(filename)


async def ask_ai(question: str) -> str:
    with st.spinner("🤖 ИИ анализирует документы..."):
        context = kb.search(question)
        answer = await client.ask(question, context)
    return answer


def main() -> None:
    st.set_page_config(
        page_title=CONFIG["app_name"],
        page_icon="🤖",
        layout="wide",
    )

    st.title(f"🤖 {CONFIG['app_name']}")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 Документы")
        uploaded_files = st.file_uploader(
            "Загрузить документы",
            accept_multiple_files=True,
            type=["txt", "pdf", "png", "jpg", "jpeg"],
        )

        if uploaded_files:
            for file in uploaded_files:
                filepath = Path(CONFIG["documents_dir"]) / file.name
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    content, doc_type = asyncio.run(process_document(str(filepath)))
                    kb.add_document(file.name, content, doc_type)
                    st.success(f"✅ {file.name} загружен и добавлен в базу ({doc_type})")
                except Exception as e:
                    logging.exception(e)
                    st.error(f"❌ Ошибка обработки {file.name}: {e}")

        st.header("⚙️ Настройки")
        st.info(f"Чанков в базе: {len(kb.chunks)}")

        if st.button("🗑️ Очистить базу знаний"):
            conn = sqlite3.connect(CONFIG["db_path"])
            conn.execute("DELETE FROM chunks")
            conn.commit()
            conn.close()
            kb.chunks, kb.embeddings = [], np.array([])
            st.success("База очищена!")

    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_area("❓ Задайте вопрос по документам:", height=100, key="question")

    with col2:
        if st.button("🚀 Спросить ИИ", type="primary", use_container_width=True):
            if question:
                answer = asyncio.run(ask_ai(question))
                st.session_state.last_answer = answer
                st.session_state.last_question = question
                st.write("### ✅ Ответ ИИ:")
                st.write(answer)

                if st.button("💾 Сохранить ответ в DOCX"):
                    path = save_answer_docx(question, answer)
                    st.success(f"Ответ сохранён: {path}")
            else:
                st.warning("Введите вопрос.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        print("Консольный режим (запуск Web UI: streamlit run dialog_pro.py)")
    else:
        main()
