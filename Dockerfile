FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    poppler-utils tesseract-ocr tesseract-ocr-rus \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libfontconfig1 libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run dialog_pro.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false