from __future__ import annotations

import socket
from typing import Any

import gradio as gr
import pandas as pd

from services.pipeline_service import PipelineService
from utils.config import load_config
from utils.server_manager import ModelServerManager
from utils.schemas import PipelineParameters, PipelineRunState, QuizResult
from utils.ui_helpers import (
    append_custom_question,
    format_question_markdown,
    parse_custom_keywords,
    parse_custom_question_lines,
)


APP_CONFIG = load_config()
APP_SERVER_MANAGER = ModelServerManager(APP_CONFIG)
DEFAULT_QUIZ_MARKDOWN = "尚未產生題目。"
INPUT_MODE_VIDEO = "video"
INPUT_MODE_TRANSCRIPT = "manual_transcript"
INPUT_MODE_SUBTITLE = "subtitle_file"
STATUS_TABLE_HEADERS = ["Step", "Status", "Message", "Artifact"]
STATUS_ROW_STYLES = {
    "pending": "background-color: #f3f4f6; color: #374151;",
    "running": "background-color: #dbeafe; color: #1d4ed8;",
    "completed": "background-color: #dcfce7; color: #166534;",
    "failed": "background-color: #fee2e2; color: #b91c1c;",
    "skipped": "background-color: #e5e7eb; color: #4b5563;",
}
PROGRESS_IDLE_HTML = """
<div style="padding: 12px 14px; border: 1px solid #d1d5db; border-radius: 10px; background: #f9fafb;">
  <div style="font-weight: 600; color: #111827;">Pipeline Progress</div>
  <div style="margin-top: 6px; color: #4b5563;">尚未開始執行。</div>
</div>
""".strip()
QUIZ_OUTPUT_CSS = """
.quiz-output-question {
  font-size: 1.12rem;
  line-height: 1.7;
}

.quiz-output-question h3 {
  font-size: 1.35rem;
  line-height: 1.45;
  margin-bottom: 0.85rem;
}

.quiz-output-question p,
.quiz-output-question li {
  font-size: 1.12rem;
}

.quiz-output-question li {
  margin-bottom: 0.35rem;
}
""".strip()


def format_status_rows(state: PipelineRunState):
    rows: list[list[str]] = []
    for key, label in PipelineService.STEP_ORDER:
        step = state.steps[key]
        message = step.message
        if step.error:
            message = f"{message} | Error: {step.error}" if message else f"Error: {step.error}"
        rows.append(
            [
                label,
                step.status,
                message,
                step.artifact_path or "",
            ]
        )
    status_frame = pd.DataFrame(rows, columns=STATUS_TABLE_HEADERS)
    return status_frame.style.apply(
        lambda row: [STATUS_ROW_STYLES.get(str(row["Status"]), "")] * len(row),
        axis=1,
    )


def format_keywords(state: PipelineRunState) -> dict[str, Any]:
    return {"auto_keywords": state.keywords}


def format_chunks(state: PipelineRunState) -> list[dict[str, Any]]:
    return [chunk.model_dump(mode="json") for chunk in state.chunks]


def format_retrieval(state: PipelineRunState) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in state.retrieved_chunks]


def resolve_quiz_results(state: PipelineRunState) -> list[QuizResult]:
    if state.quiz_results:
        return state.quiz_results
    if state.quiz_result:
        return [state.quiz_result]
    return []


def format_run_info(state: PipelineRunState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "mode": state.mode,
        "input_source": state.input_source,
        "input_filename": state.input_filename,
        "parameters": state.parameters.model_dump(mode="json"),
        "quiz_generation_count": state.quiz_generation_count,
        "quiz_versions_available": len(resolve_quiz_results(state)),
        "asr_model_name": APP_CONFIG.asr_model_name,
        "asr_conda_env": APP_CONFIG.asr_conda_env,
        "summary_model_name": APP_CONFIG.summary_model_name,
        "summary_base_url": APP_CONFIG.summary_base_url,
        "embedding_model_name": APP_CONFIG.embedding_model_name,
        "embedding_conda_env": APP_CONFIG.embedding_conda_env,
        "quiz_model_name": APP_CONFIG.quiz_model_name,
        "quiz_base_url": APP_CONFIG.quiz_base_url,
        "auto_start_model_servers": APP_CONFIG.auto_start_model_servers,
        "model_server_start_strategy": APP_CONFIG.model_server_start_strategy,
        "keep_model_servers_warm": APP_CONFIG.keep_model_servers_warm,
        "errors": state.errors,
        "overview": state.overview,
    }


def format_progress_html(state: PipelineRunState) -> str:
    progress_percent = 0
    accent = "#2563eb"
    title = "等待執行"
    description = "尚未開始執行。"

    for step_key, step_label in PipelineService.STEP_ORDER:
        step = state.steps[step_key]
        start, end = PipelineService.STEP_PROGRESS[step_key]

        if step.status in {"completed", "skipped"}:
            progress_percent = int(end * 100)
            title = f"{step_label} 完成"
            description = step.message or f"{step_label} completed"
            accent = "#16a34a" if step.status == "completed" else "#6b7280"
            continue

        if step.status == "running":
            progress_percent = int(((start + end) / 2) * 100)
            title = f"{step_label} 進行中"
            description = step.message or f"{step_label} running"
            accent = "#2563eb"
            break

        if step.status == "failed":
            progress_percent = int(start * 100)
            title = f"{step_label} 失敗"
            description = step.error or step.message or f"{step_label} failed"
            accent = "#dc2626"
            break

        break

    return f"""
<div style="padding: 12px 14px; border: 1px solid #d1d5db; border-radius: 10px; background: #f9fafb;">
  <div style="display: flex; justify-content: space-between; gap: 12px; align-items: baseline;">
    <div style="font-weight: 600; color: #111827;">Pipeline Progress</div>
    <div style="font-size: 13px; color: {accent};">{progress_percent}%</div>
  </div>
  <div style="margin-top: 8px; height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
    <div style="height: 100%; width: {progress_percent}%; background: {accent};"></div>
  </div>
  <div style="margin-top: 8px; font-weight: 600; color: #1f2937;">{title}</div>
  <div style="margin-top: 4px; color: #4b5563;">{description}</div>
</div>
""".strip()


def normalize_selected_inputs(
    input_mode: str,
    video_path: str | None,
    transcript_text: str,
    subtitle_path: str | None,
) -> tuple[str | None, str, str | None]:
    if input_mode == INPUT_MODE_TRANSCRIPT:
        return None, transcript_text, None
    if input_mode == INPUT_MODE_SUBTITLE:
        return None, "", subtitle_path
    return video_path, "", None


def _build_incremental_result_updates(
    state: PipelineRunState,
    *,
    clear_transcript: bool = False,
    clear_keywords: bool = False,
    clear_chunks: bool = False,
    clear_retrieval: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
    transcript_output: Any = "" if clear_transcript else gr.skip()
    current_keywords_output: Any = None if clear_keywords else gr.skip()
    keywords_output: Any = None if clear_keywords else gr.skip()
    chunks_output: Any = None if clear_chunks else gr.skip()
    retrieval_output: Any = None if clear_retrieval else gr.skip()

    if state.steps["asr"].status in {"completed", "skipped"} and state.transcript:
        transcript_output = state.transcript

    if state.steps["summary"].status == "completed":
        current_keywords_output = format_keywords(state)
        keywords_output = format_keywords(state)

    if state.steps["chunking"].status == "completed":
        chunks_output = format_chunks(state)

    if state.steps["retrieval"].status == "completed":
        retrieval_output = format_retrieval(state)

    return (
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    )


def render_pipeline_outputs(
    state: PipelineRunState,
    *,
    reset_unfinished: bool = False,
) -> tuple:
    (
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    ) = _build_incremental_result_updates(
        state,
        clear_transcript=reset_unfinished,
        clear_keywords=reset_unfinished,
        clear_chunks=reset_unfinished,
        clear_retrieval=reset_unfinished,
    )

    return (
        state.model_dump(mode="json"),
        format_progress_html(state),
        format_run_info(state),
        format_status_rows(state),
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    )


def render_rag_outputs(
    state: PipelineRunState,
    *,
    reset_unfinished: bool = False,
) -> tuple:
    (
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    ) = _build_incremental_result_updates(
        state,
        clear_retrieval=reset_unfinished,
    )

    return (
        state.model_dump(mode="json"),
        format_progress_html(state),
        format_run_info(state),
        format_status_rows(state),
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    )


def render_regeneration_outputs(
    state: PipelineRunState,
    *,
    reset_unfinished: bool = False,
) -> tuple:
    (
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    ) = _build_incremental_result_updates(
        state,
    )

    return (
        state.model_dump(mode="json"),
        format_progress_html(state),
        format_run_info(state),
        format_status_rows(state),
        transcript_output,
        current_keywords_output,
        keywords_output,
        chunks_output,
        retrieval_output,
    )


def build_service() -> PipelineService:
    return PipelineService(APP_CONFIG, APP_SERVER_MANAGER)


def run_pipeline_ui(
    input_mode: str,
    video_path: str | None,
    transcript_text: str,
    subtitle_path: str | None,
    n_keywords: int,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    quiz_question_count: int,
    quiz_variant_count: int,
):
    selected_video_path, selected_transcript_text, selected_subtitle_path = normalize_selected_inputs(
        input_mode,
        video_path,
        transcript_text,
        subtitle_path,
    )
    parameters = PipelineParameters(
        n_keywords=n_keywords,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        quiz_question_count=quiz_question_count,
        quiz_variant_count=quiz_variant_count,
    )
    service = build_service()

    is_first_yield = True
    for state in service.stream_pipeline(
        mode="live",
        parameters=parameters,
        video_path=selected_video_path,
        transcript_text=selected_transcript_text,
        subtitle_path=selected_subtitle_path,
    ):
        yield render_pipeline_outputs(state, reset_unfinished=is_first_yield)
        is_first_yield = False


def regenerate_ui(
    state_payload: dict | None,
    custom_question_text: str,
    *,
    options_only: bool,
):
    if not state_payload:
        raise gr.Error("請先執行 Run Pipeline。")

    service = build_service()

    custom_questions = parse_custom_question_lines(custom_question_text) if options_only else None

    is_first_yield = True
    for state in service.stream_regenerate_quiz(
        run_state_payload=state_payload,
        options_only=options_only,
        custom_questions=custom_questions,
    ):
        yield render_regeneration_outputs(state, reset_unfinished=is_first_yield)
        is_first_yield = False


def run_rag_ui(
    state_payload: dict | None,
    custom_keywords_text: str,
):
    if not state_payload:
        raise gr.Error("請先執行 Run Pipeline。")

    service = build_service()
    custom_keywords = parse_custom_keywords(custom_keywords_text)

    is_first_yield = True
    for state in service.stream_rag_retrieval(
        run_state_payload=state_payload,
        custom_keywords=custom_keywords,
    ):
        yield render_rag_outputs(state, reset_unfinished=is_first_yield)
        is_first_yield = False


def regenerate_quiz_ui(
    state_payload: dict | None,
    custom_question_text: str,
):
    yield from regenerate_ui(
        state_payload,
        custom_question_text,
        options_only=False,
    )


def regenerate_options_only_ui(
    state_payload: dict | None,
    custom_question_text: str,
):
    yield from regenerate_ui(
        state_payload,
        custom_question_text,
        options_only=True,
    )


def build_demo() -> gr.Blocks:
    config = APP_CONFIG

    with gr.Blocks(title="Video Quiz Generation Demo") as demo:
        gr.HTML(f"<style>{QUIZ_OUTPUT_CSS}</style>")
        gr.Markdown(
            """
            # Video to Quiz Demo
            `Video -> ASR -> Transcript -> Summary Keywords + Chunking -> Embedding Retrieval -> Quiz Generation`
            """
        )
        gr.Markdown(
            f"""
            此 UI 僅支援 `live` 模式，任何模型錯誤都會直接在對應 step 失敗，不會回退到 mock 或本地替代輸出。
            `AUTO_START_MODEL_SERVERS={int(config.auto_start_model_servers)}`，
            啟動策略為 `{config.model_server_start_strategy}`，
            ASR 會在 conda env `{config.asr_conda_env or 'current'}` 執行，
            embedding 會在 conda env `{config.embedding_conda_env}` 執行。
            """
        )

        state_store = gr.State(value=None)
        input_mode_state = gr.State(value=INPUT_MODE_VIDEO)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Video") as video_tab:
                        video_input = gr.File(
                            label="Upload Video",
                            file_types=[".mp4", ".mov", ".mkv"],
                            type="filepath",
                        )
                    with gr.Tab("輸入字幕") as transcript_tab:
                        transcript_input = gr.Textbox(
                            label="Paste Transcript",
                            lines=8,
                            placeholder="貼上 transcript 後會直接略過 ASR",
                        )
                    with gr.Tab("輸入字幕檔") as subtitle_tab:
                        subtitle_input = gr.File(
                            label="Upload Subtitle File",
                            file_types=[".txt", ".srt"],
                            type="filepath",
                        )
                run_button = gr.Button("Run Pipeline", variant="primary")

            with gr.Column(scale=1):
                keyword_input = gr.Slider(1, 10, value=5, step=1, label="Number of Keywords")
                top_k_input = gr.Slider(1, 10, value=5, step=1, label="Retrieval Top-K")
                chunk_size_input = gr.Slider(128, 2048, value=512, step=64, label="Chunk Size")
                chunk_overlap_input = gr.Slider(0, 512, value=64, step=16, label="Chunk Overlap")
                quiz_question_count_input = gr.Slider(
                    1,
                    10,
                    value=config.quiz_question_count,
                    step=1,
                    label="Number of Questions",
                )
                quiz_variant_count_input = gr.Slider(
                    1,
                    5,
                    value=1,
                    step=1,
                    label="Number of Quiz Variants",
                )

        video_tab.select(fn=lambda: INPUT_MODE_VIDEO, outputs=[input_mode_state])
        transcript_tab.select(fn=lambda: INPUT_MODE_TRANSCRIPT, outputs=[input_mode_state])
        subtitle_tab.select(fn=lambda: INPUT_MODE_SUBTITLE, outputs=[input_mode_state])

        gr.Markdown("## Pipeline Status")
        progress_output = gr.HTML(value=PROGRESS_IDLE_HTML, label="Pipeline Progress")
        with gr.Tabs():
            with gr.Tab("Step 表格"):
                status_table = gr.Dataframe(
                    headers=STATUS_TABLE_HEADERS,
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Intermediate Results"):
                transcript_output = gr.Textbox(label="Transcript", lines=10)
                keywords_output = gr.JSON(label="Extracted Keywords")
                chunks_output = gr.JSON(label="Chunks")
                retrieval_output = gr.JSON(label="Retrieved Relevant Chunks")
            with gr.Tab("Run Metadata"):
                run_info = gr.JSON(label="Run Metadata", show_label=False)

        with gr.Column():
            custom_question_input = gr.Textbox(
                label="Custom Questions for Options-Only Regeneration",
                lines=4,
                placeholder="一行一題。若有填寫，Regenerate Options Only 會使用這些題幹續寫選項與答案。",
            )
            with gr.Row():
                regenerate_button = gr.Button("Regenerate Quiz")
                regenerate_options_button = gr.Button("Regenerate Options Only")
        with gr.Column():
            custom_keywords_input = gr.Textbox(
                label="Custom Keywords for RAG",
                lines=3,
                placeholder="可用逗號或換行分隔。留空則使用自動關鍵字。",
            )
            current_keywords_output = gr.JSON(label="Current Auto Keywords")
            run_rag_button = gr.Button("進行RAG", variant="secondary")

        gr.Markdown("## Final Output")

        @gr.render(inputs=[state_store])
        def render_quiz_output(state_payload: dict | None):
            if not state_payload:
                gr.Markdown(DEFAULT_QUIZ_MARKDOWN)
                return

            state = PipelineRunState.model_validate(state_payload)
            quiz_results = resolve_quiz_results(state)
            if not quiz_results:
                gr.Markdown(DEFAULT_QUIZ_MARKDOWN)
                return

            with gr.Tabs():
                for version_index, quiz_result in enumerate(quiz_results, start=1):
                    with gr.Tab(f"Quiz v{version_index}"):
                        gr.Markdown(f"### Quiz Output v{version_index}")
                        for index, question in enumerate(quiz_result.questions, start=1):
                            with gr.Row():
                                with gr.Column(scale=6):
                                    gr.Markdown(
                                        format_question_markdown(question, index=index),
                                        elem_classes=["quiz-output-question"],
                                    )
                                with gr.Column(scale=1, min_width=180):
                                    copy_button = gr.Button("帶入自訂題目")
                                    copy_button.click(
                                        fn=lambda existing_text, stem=question.question: append_custom_question(
                                            existing_text,
                                            stem,
                                        ),
                                        inputs=[custom_question_input],
                                        outputs=[custom_question_input],
                                        show_progress="hidden",
                                    )
                        with gr.Accordion("Quiz JSON", open=False):
                            gr.JSON(value=quiz_result.model_dump(mode="json"), show_label=False)

        output_components = [
            state_store,
            progress_output,
            run_info,
            status_table,
            transcript_output,
            current_keywords_output,
            keywords_output,
            chunks_output,
            retrieval_output,
        ]

        run_button.click(
            fn=run_pipeline_ui,
            inputs=[
                input_mode_state,
                video_input,
                transcript_input,
                subtitle_input,
                keyword_input,
                top_k_input,
                chunk_size_input,
                chunk_overlap_input,
                quiz_question_count_input,
                quiz_variant_count_input,
            ],
            outputs=output_components,
            show_progress="hidden",
        )

        run_rag_button.click(
            fn=run_rag_ui,
            inputs=[state_store, custom_keywords_input],
            outputs=output_components,
            show_progress="hidden",
        )

        regenerate_button.click(
            fn=regenerate_quiz_ui,
            inputs=[state_store, custom_question_input],
            outputs=output_components,
            show_progress="hidden",
        )

        regenerate_options_button.click(
            fn=regenerate_options_only_ui,
            inputs=[state_store, custom_question_input],
            outputs=output_components,
            show_progress="hidden",
        )

    return demo


def find_available_port(host: str, start_port: int, max_attempts: int = 20) -> int:
    bind_host = host or "0.0.0.0"
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                continue
        return port
    raise OSError(
        f"Cannot find empty port in range: {start_port}-{start_port + max_attempts - 1}."
    )


if __name__ == "__main__":
    config = APP_CONFIG
    if config.auto_start_model_servers and config.model_server_start_strategy == "preload":
        APP_SERVER_MANAGER.ensure_servers_ready()
    launch_port = find_available_port(config.app_host, config.app_port)
    if launch_port != config.app_port:
        print(
            f"[gradio] Port {config.app_port} is occupied, falling back to available port {launch_port}."
        )
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=config.app_host,
        server_port=launch_port,
    )
