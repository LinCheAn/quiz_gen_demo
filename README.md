# Video Quiz Generation Gradio Demo

本專案提供一個可轉交的本地 demo，用來展示：

`Video -> ASR -> Transcript -> Summary Keywords + Chunking -> Embedding Retrieval -> Quiz Generation`

目前交付方式改成：

- project-relative 路徑，不再依賴 `/home/...` 這類個人機器絕對路徑
- Docker-first，單一 runtime 內執行 app、ASR、embedding、vLLM
- repo 內固定 `models/`、`data/`、`artifacts/` 結構
- 以 NVIDIA GPU 為前提，不提供 CPU fallback

## 專案結構

```bash
quiz_gen_demo/
  app.py
  model_info.json
  Dockerfile
  docker-compose.yml
  docker/
    entrypoint.sh
  models/
    adapters/
    cache/
  data/
    uploads/
  artifacts/
    runs/
    server_logs/
  services/
  utils/
  tests/
```

## 模型與資料目錄

這個 repo 預設把模型相關路徑都收斂到 `models/`：

- `models/adapters/`
  - 放自訂 LoRA adapter
  - `model_info.json` 內的 `lora_path` 一律是相對於 repo 的路徑
- `models/cache/huggingface/`
  - Hugging Face cache
  - container 與本地執行都會預設把 `HF_HOME` 指到這裡
- `data/uploads/`
  - 上傳的影片 / 字幕
- `artifacts/runs/`
  - 每次 pipeline 的中繼結果
- `artifacts/server_logs/`
  - auto-start vLLM server 的 log

目前內建的 optional adapter preset：

- `models/adapters/grpo_v4.2/`
- `models/adapters/dpo_v9.3_ocw/`

如果這兩個資料夾不存在，app 仍可啟動，但只有在你真的選到對應 preset 時才會報錯。預設 quiz model 已改成 base model `llama-3.1-8b-instruct`，不依賴額外 adapter。

## Docker 使用方式

### 前置條件

- Docker
- Docker Compose v2
- NVIDIA GPU
- NVIDIA Container Toolkit

### 1. 準備模型目錄

最少只需要讓 repo 內有這些目錄：

```bash
mkdir -p models/adapters models/cache/huggingface data/uploads artifacts
```

如果你要使用 GRPO / DPO preset，再把 adapter 放進：

```bash
models/adapters/grpo_v4.2/
models/adapters/dpo_v9.3_ocw/
```

base model 仍使用 Hugging Face model id，第一次啟動時會下載到 `models/cache/huggingface/`。

### 2. 啟動

主要入口：

```bash
docker compose up --build
```

啟動後預設可在：

```bash
http://127.0.0.1:7860
```

### 3. 常用操作

背景執行：

```bash
docker compose up --build -d
```

看 log：

```bash
docker compose logs -f demo
```

停止：

```bash
docker compose down
```

### 4. `docker run` 備用指令

```bash
docker build -t quiz-gen-demo .
docker run --rm -it \
  --gpus all \
  -p 7860:7860 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e APP_HOST=0.0.0.0 \
  -e APP_PORT=7860 \
  quiz-gen-demo
```

## 執行行為

預設情況下：

- `AUTO_START_MODEL_SERVERS=1`
- `MODEL_SERVER_START_STRATEGY=sequential`
- ASR / embedding worker 直接使用 current runtime
- summary / quiz server 也直接使用 current runtime 啟動 `vllm serve`

如果你要自己管理 summary / quiz endpoint，可關掉 auto-start：

```bash
AUTO_START_MODEL_SERVERS=0 docker compose up
```

## `model_info.json`

`Summary Model` 與 `Quiz Model` 下拉選單都從 `model_info.json` 載入。

格式重點：

- `defaults.summary_model_id`
- `defaults.quiz_model_id`
- `models[]`

每個 preset 可提供：

- `id`
- `label`
- `model_name`
- `base_url`
- `server_model`
- `server_conda_env`
- `lora_path`
- `gpu_memory_utilization`
- `max_model_len`
- `tensor_parallel_size`
- `dtype`
- `quantization`

說明：

- `server_conda_env` 現在可留空字串，表示直接使用 current runtime
- `lora_path` 現在應指向 repo 內相對路徑，例如 `models/adapters/grpo_v4.2`
- 若 `lora_path` 省略，代表直接使用 base model

## 測試

```bash
python -m unittest discover -s tests
```

## 本地開發附錄

如果你不想先用 Docker，本地也只需要單一環境，不再需要 `demo` / `inference` / `vllm` 三套 conda 切換。

範例：

```bash
conda create -n demo python=3.10 -y
conda activate demo
pip install -r requirements.txt
python app.py
```

本地模式同樣會把 Hugging Face cache 寫到 repo 內的 `models/cache/huggingface/`。

## 備註

- 這個專案是 demo workflow，不是 production service。
- GPU 是必要條件；目前不支援 CPU-only 執行。
- 不建議把實際模型權重提交到 Git；repo 已預設忽略 `models/adapters/` 與 `models/cache/` 內容。
