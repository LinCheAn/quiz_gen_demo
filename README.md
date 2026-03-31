# Video Quiz Generation Gradio Demo

這個專案是一個本地可執行的 Gradio WebUI demo，用來展示影片轉測驗題的 inference workflow：

`Video -> ASR -> Transcript -> Summary Keywords + Chunking -> Embedding Retrieval -> Quiz Generation`

重點不是 production，而是讓流程可 demo、可觀察、可替換成既有模型程式。

## 專案結構

```bash
quiz_gen_demo/
  app.py
  model_info.json
  services/
    asr_service.py
    summary_service.py
    chunk_service.py
    embedding_service.py
    quiz_service.py
    pipeline_service.py
  utils/
    config.py
    model_registry.py
    schemas.py
    storage.py
  data/uploads/
  artifacts/runs/
  tests/
  requirements.txt
  README.md
```

## 模組責任

- `app.py`
  - Gradio UI
  - Pipeline status / progress 顯示
  - 重新生成 quiz / 只重生 options 的按鈕事件
- `services/asr_service.py`
  - 影片抽音訊
  - Breeze-ASR-25 wrapper
  - 直接執行真實 ASR；失敗時不回退
- `services/summary_service.py`
  - transcript 關鍵字抽取
  - 預設對接 OpenAI-compatible summary endpoint
- `services/chunk_service.py`
  - transcript chunking
- `services/embedding_service.py`
  - BGE-M3 retrieval
  - 透過指定 conda env 執行 embedding worker
- `services/quiz_service.py`
  - quiz generation
  - full regenerate / options-only regenerate
- `services/pipeline_service.py`
  - 串接整條 workflow
  - step status、artifact 寫入、錯誤處理
- `utils/config.py`
  - 所有預設參數與環境變數
- `utils/schemas.py`
  - Pipeline state 與輸出 schema
- `utils/storage.py`
  - run artifact 路徑與 JSON/TXT 存取

## Artifact 與中繼檔

所有過程檔案都只會寫在本專案內：

- `data/uploads/`
- `artifacts/runs/{run_id}/inputs/`
- `artifacts/runs/{run_id}/audio/`
- `artifacts/runs/{run_id}/outputs/`

每次點 `Run Pipeline` 都會建立新的 `run_id`，中繼資料包含：

- `transcript.txt`
- `keywords.json`
- `chunks.json`
- `retrieval.json`
- `quiz_v{n}.json`
- `state.json`
- `error.json`（若失敗）

## 安裝

### 方案 A：跑 UI 與完整 live pipeline

建議建立新的 UI conda environment，不污染既有模型環境：

```bash
conda create -n quiz-demo python=3.10 -y
conda activate quiz-demo
pip install -r requirements.txt
```

### 方案 B：接既有模型推論環境

此專案現在只支援 `live` 模式：

- ASR 需要 `transformers`, `torch`, `numpy`
- 可用 `ASR_CONDA_ENV` 指定 ASR worker 執行的 conda env
- retrieval 會在 `EMBEDDING_CONDA_ENV` 指定的環境內執行 `FlagEmbedding`
- summary / quiz 預設走 OpenAI-compatible endpoint

建議做法：

- UI 跑在新環境 `quiz-demo`
- 模型推論沿用你既有的 conda environment 或既有 vLLM service

## 執行

```bash
conda run -n demo python app.py
```

啟動後，`Run Pipeline` 旁邊會提供 `Summary Model` 與 `Quiz Model` 兩個下拉選單。兩者都從 repo 內的 `model_info.json` 載入，並在你按下 `Run Pipeline` 的當下鎖定到該次 run；後續的 `RAG` / `Regenerate` 會沿用同一份 run snapshot，不會因為 UI 下拉選單後來變動而改掉。

預設會啟在：

```bash
http://0.0.0.0:7860
```

你也可以用環境變數覆蓋：

```bash
APP_HOST=0.0.0.0 APP_PORT=7860 python app.py
```

預設情況下，`app.py` 使用 `MODEL_SERVER_START_STRATEGY=sequential`：

- 進入 summary step 前啟動 summary server: `http://127.0.0.1:8001/v1`
- summary 完成後釋放 summary server
- 進入 quiz step 前啟動 quiz server: `http://127.0.0.1:8000/v1`
- quiz 完成後釋放 quiz server

若你想保留舊行為，在 app 啟動時一次預載兩個 server，可改成：

```bash
MODEL_SERVER_START_STRATEGY=preload conda run -n demo python app.py
```

自動啟動的服務預設會透過 `conda run -n vllm vllm serve ...` 啟動，因此需要：

- 本機可用的 CUDA
- `vllm` conda env 存在
- 可以載入對應模型

如果你要手動管理 summary / quiz server，不要自動起模型服務：

```bash
AUTO_START_MODEL_SERVERS=0 conda run -n demo python app.py
```

## 模型對接位置

### model_info.json

檔案：`model_info.json`

`Summary Model` 與 `Quiz Model` 下拉選單都從這個檔案讀取。格式分成：

- `defaults.summary_model_id`
- `defaults.quiz_model_id`
- `summary_models[]`
- `quiz_models[]`

其中：

- `summary_models[]` 需要提供 model id、label、API `base_url`、對應的 served `model_name`，以及啟動 summary server 所需的 conda env / base model / vLLM 參數。
- `quiz_models[]` 除了上述欄位外，還需要提供 `lora_path`，讓 quiz server 能用該 adapter 啟動。

如果你要新增可選模型，優先改這個檔案，而不是直接改 `app.py`。

### 1. ASR

檔案：`services/asr_service.py`

目前直接在這個 demo 內封裝 Breeze-ASR-25，參考：

- `/home/r13922145/cool-course/all_courses/ocw_source/transcribe_breeze.py`

如果你之後要改成：

- import 既有函式
- subprocess 呼叫既有腳本
- HTTP service

只需要改 `ASRService._transcribe_live()`，不用動 UI 與 pipeline。

### 2. Summary

檔案：`services/summary_service.py`

目前預設：

- OpenAI-compatible client
- model 預設 `llama3.1-8b-instruct`
- prompt 參考 `/home/r13922145/local-genai-edu/Pipeline.py`

可用以下環境變數調整：

```bash
SUMMARY_BASE_URL=http://127.0.0.1:8001/v1
SUMMARY_API_KEY=0
SUMMARY_MODEL_NAME=llama3.1-8b-instruct
```

### 3. Quiz Model

檔案：`services/quiz_service.py`

目前預設：

- OpenAI-compatible / vLLM endpoint
- model name 預設 `grpo_v4.2`
- prompt 參考：
  - `/home/r13922145/local-genai-edu/2task_pipeline/full_question_generate/generate_full_questions.py`
  - `/home/r13922145/local-genai-edu/2task_pipeline/continuation_generate/generate_continuation_options.py`

你指定的 quiz adapter 路徑為：

```bash
/home/r13922145/rl_model/grpo_v4.2
```

此 adapter 的 base model 為：

```bash
unsloth/Llama-3.1-8B-Instruct
```

本 demo 不會直接修改或啟動你的既有模型程式；它只保留清楚對接點。

可用以下環境變數調整：

```bash
QUIZ_BASE_URL=http://127.0.0.1:8000/v1
QUIZ_API_KEY=0
QUIZ_MODEL_NAME=grpo_v4.2
QUIZ_MODEL_PATH=/home/r13922145/rl_model/grpo_v4.2
QUIZ_BASE_MODEL_NAME=unsloth/Llama-3.1-8B-Instruct
```

自動啟動 server 相關的設定也可以透過環境變數覆蓋，例如：

```bash
AUTO_START_MODEL_SERVERS=1
MODEL_SERVER_START_STRATEGY=sequential
SUMMARY_SERVER_CONDA_ENV=vllm
QUIZ_SERVER_CONDA_ENV=vllm
SUMMARY_SERVER_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
QUIZ_SERVER_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
SUMMARY_SERVER_GPU_MEMORY_UTILIZATION=0.45
QUIZ_SERVER_GPU_MEMORY_UTILIZATION=0.45
```

### 4. Embedding Retrieval

檔案：`services/embedding_service.py`

embedding 不再依賴啟動 `app.py` 的當前環境，而是固定透過子程序在指定 conda env 內執行：

```bash
ASR_CONDA_ENV=inference
EMBEDDING_CONDA_ENV=inference
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_USE_FP16=1
```

如果你的 `FlagEmbedding` 安裝在 `cool`，就改成：

```bash
ASR_CONDA_ENV=cool
EMBEDDING_CONDA_ENV=cool
```

## 使用方式

### Input 優先順序

1. 直接貼上的 transcript
2. 上傳字幕檔 `.txt` / `.srt`
3. 上傳影片檔，交由 ASR

### Regenerate 行為

- `Regenerate Quiz`
  - 只重跑 quiz step
  - 不重跑 ASR / summary / chunk / retrieval
- `Regenerate Options Only`
  - 保留現有題幹
  - 只重生選項與答案

## 測試

目前附的是 fail-fast 測試與 fake service 測試：

```bash
python -m unittest discover -s tests
```

## 備註

- 此專案只支援 `live` 模式。
- 任一步驟失敗都會直接停在該 step，不會輸出 mock 或本地替代結果。
- summary / quiz server 是否自動啟動，取決於 `AUTO_START_MODEL_SERVERS` 與 `MODEL_SERVER_START_STRATEGY`。
- 本專案不會修改 `/home/r13922145/local-genai-edu` 或其他既有程式碼。
