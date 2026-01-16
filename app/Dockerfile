# Dockerfile
FROM python:3.10-slim

# set the llama model version to pull
ARG MODEL=llama3.2

# system‑level dependencies for python-docx, Milvus client, etc.
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       libgl1 libsm6 libxext6 && \
#     rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# Copy source code into container directory
# COPY ["tuiBeta.py", "."]
# COPY ["Rag_Code.py", "."]
# COPY ["entrypoint.sh","."]

# Expose ports if we switch to a web UI; for TUI it’s not needed
# EXPOSE 8000

# Set defaults for connecting to Milvus and Ollama in compose
ENV MILVUS_HOST=milvus \
    MILVUS_PORT=19530 \
    OLLAMA_HOST=ollama \
    OLLAMA_PORT=11434

# Default command: launch TUI
# CMD ["python", "tuiBeta.py"]
