FROM vllm/vllm-openai:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/models/adapters /app/models/cache/huggingface /app/data/uploads /app/artifacts

ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    ASR_CONDA_ENV= \
    EMBEDDING_CONDA_ENV= \
    SUMMARY_SERVER_CONDA_ENV= \
    QUIZ_SERVER_CONDA_ENV= \
    HF_HOME=/app/models/cache/huggingface \
    HF_HUB_CACHE=/app/models/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/models/cache/huggingface/transformers

EXPOSE 7860

ENTRYPOINT ["/app/docker/entrypoint.sh"]
