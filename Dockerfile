FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

# CPU-only PyTorch first. The default build downloads GPU files you don't need (about 2 GB more).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a normal user, not root. /data holds chats, PDFs and indexes.
RUN useradd -m appuser && mkdir /data && chown appuser:appuser /data
USER appuser

# Download the embedding model now, so the first PDF upload doesn't wait for it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=appuser:appuser agentic_chatbot_hitl_backend.py app_hitl.py ./

VOLUME /data
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app_hitl.py", "--server.address=0.0.0.0", "--server.port=8501"]
