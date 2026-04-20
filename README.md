# Video Quiz Generation Gradio Demo

本專案提供一個可轉交的本地 demo，用來展示：

`Video -> ASR -> Transcript -> Summary Keywords + Chunking -> Embedding Retrieval -> Quiz Generation`


- Docker-first，單一 runtime 內執行 app、ASR、embedding、vLLM
- 鎖定 `demo` runtime 的 Python 套件版本與 vLLM base image tag
- repo 內固定 `models/`、`data/`、`artifacts/` 結構
- 以 NVIDIA GPU 為前提，不提供 CPU fallback

## Project Structure

```bash
quiz_gen_demo/
  app.py
  model_info.json
  Dockerfile
  docker-compose.yml
  docker-compose.deploy.yml
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

## Models and Data Directories

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

## Dependency Management

- `requirements.txt`
  - 維護者用的 top-level 依賴清單
  - 只在你要調整或升級依賴時使用
- `requirements.lock.txt`
  - 目前 `demo` env 驗證過的完整鎖版結果
  - 一般本地安裝、部署、Docker build 都應安裝這份檔案
- `Dockerfile`
  - 預設使用 `vllm/vllm-openai:v0.19.1`
  - 若你需要切換到其他已驗證的 base image，可在 build 時覆蓋 `VLLM_OPENAI_IMAGE`

一般使用方式：

```bash
pip install -r requirements.lock.txt
```

只有在你要更新依賴時，才使用以下流程：

```bash
conda activate demo
pip install -r requirements.txt
python -m unittest discover -s tests
python -m pip freeze > requirements.lock.txt
```

也就是：

- `requirements.txt` 用來產生新的 lock file
- `requirements.lock.txt` 用來重現已驗證可用的環境

只有在確認新環境可用後，才更新 `requirements.lock.txt`。

## Three Ways to Run


- 方法 A: 自己 build Docker image，再用 `docker-compose.yml` 啟動
- 方法 B: 直接使用已經 build 好的 image，用 `docker-compose.deploy.yml` 啟動
- 方法 C: 不用 Docker，直接在本機安裝 Python 環境後執行 `python app.py`

三種方式共通前提：

- NVIDIA GPU
- base model 第一次啟動時，會下載到 `models/cache/huggingface/`
- 若要使用 GRPO / DPO preset，需自行準備對應 adapter 目錄

### Shared Preparation

先建立必要目錄：

```bash
mkdir -p models/adapters models/cache/huggingface data/uploads artifacts
```

如果你要使用 GRPO / DPO preset，再把 adapter 放進：

```bash
models/adapters/grpo_v4.2/
models/adapters/dpo_v9.3_ocw/
```

### Method A: Build the Docker Image and Run

適用情境：

- 你要從目前 repo 原始碼自行 build
- 你有修改 `app.py`、`services/`、`requirements.lock.txt` 或 `Dockerfile`


主要入口：

```bash
docker compose up --build
```

若要背景執行：

```bash
docker compose up --build -d
```

如果你要覆蓋 base image：

```bash
VLLM_OPENAI_IMAGE=vllm/vllm-openai:v0.19.1 docker compose up --build
```

如果你要先手動 build：

```bash
docker build \
  --build-arg VLLM_OPENAI_IMAGE=vllm/vllm-openai:v0.19.1 \
  -t quiz-gen-demo:latest .
```

常用操作：

```bash
docker compose logs -f demo
docker compose down
```

### Method B: Use a Prebuilt Image Directly

適用情境：

- 你不想在目標機器重新 build
- 你已經有可直接拉取或本機已存在的 image tag


指定要使用的 image：

```bash
export QUIZ_GEN_DEMO_IMAGE=linchean/quiz-gen-demo:latest
```

直接啟動：

```bash
docker compose -f docker-compose.deploy.yml up -d
```

如果你想先明確 pull 再啟動：

```bash
docker compose -f docker-compose.deploy.yml pull
docker compose -f docker-compose.deploy.yml up -d
```

更新到其他 image tag：

```bash
export QUIZ_GEN_DEMO_IMAGE=your-image-repo/quiz-gen-demo:v2026-04-20
docker compose -f docker-compose.deploy.yml pull
docker compose -f docker-compose.deploy.yml up -d
```

常用操作：

```bash
docker compose -f docker-compose.deploy.yml logs -f demo
docker compose -f docker-compose.deploy.yml down
```

### Method C: Install the Environment and Run Locally

```bash
conda create -n demo python=3.10 -y
conda activate demo
pip install -r requirements.lock.txt
python app.py
```

如果你只想做本機 UI 開發，不自動啟動 vLLM 相關服務：

```bash
AUTO_START_MODEL_SERVERS=0 python app.py
```

本地模式同樣會把 Hugging Face cache 寫到 repo 內的 `models/cache/huggingface/`。

### `docker run` Alternative

```bash
docker build -t quiz-gen-demo:latest .
docker run --rm -it \
  --gpus all \
  -p 7860:7860 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -e APP_HOST=0.0.0.0 \
  -e APP_PORT=7860 \
  quiz-gen-demo:latest
```

## Runtime Behavior

預設情況下：

- `AUTO_START_MODEL_SERVERS=1`
- `MODEL_SERVER_START_STRATEGY=sequential`
- ASR / embedding worker 直接使用 current runtime
- summary / quiz server 也直接使用 current runtime 啟動 `vllm serve`

如果你要自己管理 summary / quiz endpoint，可關掉 auto-start：

```bash
AUTO_START_MODEL_SERVERS=0 docker compose up
```

使用 `docker-compose.deploy.yml` 時也一樣可以覆蓋：

```bash
AUTO_START_MODEL_SERVERS=0 docker compose -f docker-compose.deploy.yml up -d
```

補充：

- 這套流程可以避免每台機器重新 build image
- 但如果目標機器的 `models/cache/huggingface/` 是空的，第一次啟動時仍會下載 base model
- 如果你需要完全離線啟動，必須另外預載模型 cache 或把模型一併打包；這不在目前這份 repo 的預設流程內

## `model_info.json`

`Summary Model` 與 `Quiz Model` 下拉選單都從 `model_info.json` 載入。

目前預設行為：

- `Summary Model` 預設不啟用 LoRA，直接使用 base model
- `Quiz Model` 可選擇 base model 或帶有 `lora_path` 的 preset
- 如果某個 preset 沒有設定 `lora_path`，就表示該 preset 也直接使用 base model



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
- `Summary Model` 的預設設定就是不提供 `lora_path`，因此 summary 服務啟動時不會額外載入 adapter

## Tests

```bash
python -m unittest discover -s tests
```

## Notes

- 這個專案是 demo workflow，不是 production service。
- GPU 是必要條件；目前不支援 CPU-only 執行。
