from __future__ import annotations

import os
import time
from typing import Optional
from urllib.parse import quote

import requests


class ComfyExecutionError(RuntimeError):
    """Raised when ComfyUI cannot execute a submitted workflow."""


class ComfyClient:
    """Small client for ComfyUI's local prompt and history API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        poll_interval: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188")
        ).rstrip("/")
        self._poll_interval = (
            float(os.environ.get("COMFY_POLL_INTERVAL_SECONDS", "2"))
            if poll_interval is None
            else poll_interval
        )
        self._request_timeout = (
            float(os.environ.get("COMFY_REQUEST_TIMEOUT_SECONDS", "30"))
            if request_timeout is None
            else request_timeout
        )
        self._session = requests.Session()

    def execute(self, workflow: dict, client_id: str) -> dict:
        """Submit a workflow and return its completed ComfyUI history entry."""
        prompt_response = self._request(
            "post", "/prompt", json={"prompt": workflow, "client_id": client_id}
        )
        prompt_payload = self._json(prompt_response)
        prompt_id = (
            prompt_payload.get("prompt_id") if isinstance(prompt_payload, dict) else None
        )
        if not isinstance(prompt_id, str) or not prompt_id:
            validation_message = (
                _validation_message(prompt_payload.get("node_errors"))
                if isinstance(prompt_payload, dict)
                else None
            )
            if validation_message:
                raise ComfyExecutionError(f"ComfyUI validation failed: {validation_message}")
            raise ComfyExecutionError("ComfyUI response did not include a prompt_id")

        history_path = f"/history/{quote(prompt_id, safe='')}"
        while True:
            history_payload = self._json(self._request("get", history_path))
            if not isinstance(history_payload, dict):
                raise ComfyExecutionError("ComfyUI returned invalid history")
            history = history_payload.get(prompt_id)
            if isinstance(history, dict):
                status = history.get("status")
                if not isinstance(status, dict):
                    raise ComfyExecutionError("ComfyUI history has invalid status")
                execution_message = _execution_message(status.get("messages"))
                if execution_message:
                    raise ComfyExecutionError(execution_message)
                if status.get("completed") is True:
                    return history
            time.sleep(self._poll_interval)

    def _request(self, method: str, path: str, **kwargs: object) -> requests.Response:
        try:
            response = getattr(self._session, method)(
                f"{self._base_url}{path}", timeout=self._request_timeout, **kwargs
            )
        except requests.Timeout as exc:
            raise ComfyExecutionError("ComfyUI request timed out") from exc
        except requests.RequestException as exc:
            raise ComfyExecutionError("ComfyUI request failed") from exc

        if 200 <= response.status_code < 300:
            return response

        validation_message = _validation_message(_safe_json(response).get("node_errors"))
        if validation_message:
            raise ComfyExecutionError(f"ComfyUI validation failed: {validation_message}")
        raise ComfyExecutionError(f"ComfyUI request failed with HTTP {response.status_code}")

    @staticmethod
    def _json(response: requests.Response) -> object:
        try:
            return response.json()
        except ValueError as exc:
            raise ComfyExecutionError("ComfyUI returned invalid JSON") from exc


def _safe_json(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _execution_message(messages: object) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, (list, tuple)) and message and message[0] in {
            "execution_error",
            "execution_interrupted",
        }:
            detail = _message_from(message[1] if len(message) > 1 else None)
            return detail or f"ComfyUI {message[0]}"
    return None


def _message_from(value: object) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in ("message", "error", "exception_message"):
        message = value.get(key)
        if isinstance(message, str) and message:
            return message
    return None


def _validation_message(node_errors: object) -> Optional[str]:
    if not isinstance(node_errors, dict):
        return None
    for node_error in node_errors.values():
        if not isinstance(node_error, dict):
            continue
        message = _message_from(node_error)
        if message:
            return message
        errors = node_error.get("errors")
        if not isinstance(errors, list):
            continue
        for error in errors:
            message = _message_from(error)
            if message:
                return message
    return None
