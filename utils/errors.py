from __future__ import annotations


class ModelResponseFormatError(RuntimeError):
    def __init__(
        self,
        *,
        step: str,
        message: str,
        raw_response: str,
        model_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.raw_response = raw_response
        self.model_name = model_name
