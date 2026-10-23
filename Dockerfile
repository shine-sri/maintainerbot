FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir streamlit pandas altair scikit-learn transformers sentence-transformers huggingface_hub requests ollama
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('shine-sri/maintainerbot-distilbert', local_dir='models/maintainerbot-distilbert')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
