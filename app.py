from __future__ import annotations

from typing import Any

import gradio as gr

from services.pipeline_service import PipelineService
from utils.config import load_config
from utils.server_manager import ModelServerManager
from utils.schemas import PipelineParameters, PipelineRunState


APP_CONFIG = load_config()
APP_SERVER_MANAGER = ModelServerManager(APP_CONFIG)
DEFAULT_QUIZ_MARKDOWN = "尚未產生題目。"
INPUT_MODE_VIDEO = "video"
INPUT_MODE_TRANSCRIPT = "manual_transcript"
INPUT_MODE_SUBTITLE = "subtitle_file"


def format_status_rows(state: PipelineRunState) -> list[list[str]]:
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
    return rows


def format_chunks(state: PipelineRunState) -> list[dict[str, Any]]:
    return [chunk.model_dump(mode="json") for chunk in state.chunks]


def format_retrieval(state: PipelineRunState) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in state.retrieved_chunks]


def format_quiz_markdown(state: PipelineRunState) -> str:
    if not state.quiz_result:
        return "尚未產生題目。"
    lines = [f"### Quiz Output v{state.quiz_generation_count}"]
    for index, question in enumerate(state.quiz_result.questions, start=1):
        lines.append(f"**Q{index}. {question.question}**")
        for option_key in ("A", "B", "C", "D"):
            lines.append(f"- {option_key}. {question.options[option_key]}")
        lines.append(f"Answer: `{question.answer}`")
        if question.explanation:
            lines.append(f"Explanation: {question.explanation}")
        lines.append("")
    return "\n".join(lines)


def format_run_info(state: PipelineRunState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "mode": state.mode,
        "input_source": state.input_source,
        "input_filename": state.input_filename,
        "quiz_generation_count": state.quiz_generation_count,
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
    reset_unfinished: bool,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    transcript_output: Any = "" if reset_unfinished else gr.skip()
    keywords_output: Any = None if reset_unfinished else gr.skip()
    chunks_output: Any = None if reset_unfinished else gr.skip()
    retrieval_output: Any = None if reset_unfinished else gr.skip()
    quiz_markdown_output: Any = DEFAULT_QUIZ_MARKDOWN if reset_unfinished else gr.skip()
    quiz_json_output: Any = None if reset_unfinished else gr.skip()

    if state.steps["asr"].status in {"completed", "skipped"} and state.transcript:
        transcript_output = state.transcript

    if state.steps["summary"].status == "completed":
        keywords_output = {"keywords": state.keywords}

    if state.steps["chunking"].status == "completed":
        chunks_output = format_chunks(state)

    if state.steps["retrieval"].status == "completed":
        retrieval_output = format_retrieval(state)

    if state.steps["quiz"].status == "completed" and state.quiz_result:
        quiz_markdown_output = format_quiz_markdown(state)
        quiz_json_output = state.quiz_result.model_dump(mode="json")

    return (
        transcript_output,
        keywords_output,
        chunks_output,
        retrieval_output,
        quiz_markdown_output,
        quiz_json_output,
    )


def render_pipeline_outputs(
    state: PipelineRunState,
    *,
    reset_unfinished: bool = False,
) -> tuple:
    (
        transcript_output,
        keywords_output,
        chunks_output,
        retrieval_output,
        quiz_markdown_output,
        quiz_json_output,
    ) = _build_incremental_result_updates(state, reset_unfinished=reset_unfinished)

    return (
        state.model_dump(mode="json"),
        format_run_info(state),
        format_status_rows(state),
        transcript_output,
        keywords_output,
        chunks_output,
        retrieval_output,
        quiz_markdown_output,
        quiz_json_output,
    )


def render_regeneration_outputs(state: PipelineRunState) -> tuple:
    quiz_markdown_output: Any = gr.skip()
    quiz_json_output: Any = gr.skip()

    if state.steps["quiz"].status == "completed" and state.quiz_result:
        quiz_markdown_output = format_quiz_markdown(state)
        quiz_json_output = state.quiz_result.model_dump(mode="json")

    return (
        state.model_dump(mode="json"),
        format_run_info(state),
        format_status_rows(state),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        quiz_markdown_output,
        quiz_json_output,
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
    progress: gr.Progress = gr.Progress(track_tqdm=False),
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
    )
    service = build_service()

    def progress_callback(value: float, desc: str) -> None:
        progress(value, desc=desc)

    is_first_yield = True
    for state in service.stream_pipeline(
        mode="live",
        parameters=parameters,
        video_path=selected_video_path,
        transcript_text=selected_transcript_text,
        subtitle_path=selected_subtitle_path,
        progress_callback=progress_callback,
    ):
        yield render_pipeline_outputs(state, reset_unfinished=is_first_yield)
        is_first_yield = False


def regenerate_ui(
    state_payload: dict | None,
    *,
    options_only: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if not state_payload:
        raise gr.Error("請先執行 Run Pipeline。")

    service = build_service()

    def progress_callback(value: float, desc: str) -> None:
        progress(value, desc=desc)

    for state in service.stream_regenerate_quiz(
        run_state_payload=state_payload,
        options_only=options_only,
        progress_callback=progress_callback,
    ):
        yield render_regeneration_outputs(state)


def regenerate_quiz_ui(
    state_payload: dict | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    yield from regenerate_ui(state_payload, options_only=False, progress=progress)


def regenerate_options_only_ui(
    state_payload: dict | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    yield from regenerate_ui(state_payload, options_only=True, progress=progress)


def build_demo() -> gr.Blocks:
    config = APP_CONFIG

    with gr.Blocks(title="Video Quiz Generation Demo") as demo:
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

        video_tab.select(fn=lambda: INPUT_MODE_VIDEO, outputs=[input_mode_state])
        transcript_tab.select(fn=lambda: INPUT_MODE_TRANSCRIPT, outputs=[input_mode_state])
        subtitle_tab.select(fn=lambda: INPUT_MODE_SUBTITLE, outputs=[input_mode_state])

        gr.Markdown("## Pipeline Status")
        run_info = gr.JSON(label="Run Metadata")
        status_table = gr.Dataframe(
            headers=["Step", "Status", "Message", "Artifact"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        with gr.Row():
            regenerate_button = gr.Button("Regenerate Quiz")
            regenerate_options_button = gr.Button("Regenerate Options Only")

        gr.Markdown("## Intermediate Results")
        transcript_output = gr.Textbox(label="Transcript", lines=10)
        keywords_output = gr.JSON(label="Extracted Keywords")
        chunks_output = gr.JSON(label="Chunks")
        retrieval_output = gr.JSON(label="Retrieved Relevant Chunks")

        gr.Markdown("## Final Output")
        quiz_markdown = gr.Markdown(DEFAULT_QUIZ_MARKDOWN)
        quiz_json = gr.JSON(label="Quiz JSON")

        output_components = [
            state_store,
            run_info,
            status_table,
            transcript_output,
            keywords_output,
            chunks_output,
            retrieval_output,
            quiz_markdown,
            quiz_json,
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
            ],
            outputs=output_components,
            show_progress="minimal",
        )

        regenerate_button.click(
            fn=regenerate_quiz_ui,
            inputs=[state_store],
            outputs=output_components,
            show_progress="minimal",
        )

        regenerate_options_button.click(
            fn=regenerate_options_only_ui,
            inputs=[state_store],
            outputs=output_components,
            show_progress="minimal",
        )

    return demo


if __name__ == "__main__":
    config = APP_CONFIG
    if config.auto_start_model_servers and config.model_server_start_strategy == "preload":
        APP_SERVER_MANAGER.ensure_servers_ready()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=config.app_host,
        server_port=config.app_port,
    )
