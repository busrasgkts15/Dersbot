# Hugging Face build ortamı için Python 3.11 kullan
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Gerekli sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

 RUN mkdir -p /data && chmod -R 777 /data
ENV TRANSFORMERS_CACHE=/data/cache
ENV HF_HOME=/data/cache


# Gereken dosyaları kopyala
COPY requirements.txt .
COPY src/ ./src/

# Python paketlerini yükle
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit portu
EXPOSE 8501

# Sağlık kontrolü
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Uygulama çalıştırma komutu
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
