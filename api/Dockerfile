# Dockerfile — API de prédiction Marketing bancaire
# Contexte de build : racine du dépôt (voir docker-compose.yml)

FROM python:3.11-slim

RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -m appuser

WORKDIR /home/appuser/app

COPY --chown=appuser:appuser api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser models/ ./models/

RUN mkdir -p logs && chown appuser:appuser logs

USER appuser

EXPOSE 8000

HEALTHCHECK --start-period=15s --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
