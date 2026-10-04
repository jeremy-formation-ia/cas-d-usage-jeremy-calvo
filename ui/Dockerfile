# Dockerfile — UI Streamlit Marketing bancaire
# Contexte de build : racine du dépôt (voir docker-compose.yml)

FROM python:3.11-slim

WORKDIR /app

COPY ui/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ui/ ./ui/

EXPOSE 8501

HEALTHCHECK --start-period=10s --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
