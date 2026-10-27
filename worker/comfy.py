from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Optional
from urllib.parse import quote

import requests

from worker.diagnostics import log_hang_diagnostics

logger = logging.getLogger(__name__)

# DBee submits its jobs with a 20 minute RunPod execution cap. Stop earlier so the
# worker can log why it stalled; RunPod's own kill leaves no trace.
DBEE_EXECUTION_LIMIT_SECONDS = 1080.0
# Other job types, and DBee product showcases (a cold worker can spend most of 20
# minutes loading weights), run under the endpoint's own 3600 second cap.
DEFAULT_EXECUTION_LIMIT_SECONDS = 3300.0
DBEE_FULL_CAP_JOB_TYPES = frozenset({"dbee-product-showcase"})


class ComfyExecutionError(RuntimeError):
    """Raised when ComfyUI cannot execute a submitted workflow."""


class ComfyExecutionTimeout(ComfyExecutionError):
    """Raised when ComfyUI does not finish a workflow within its time limit."""


def execution_limit_seconds(job_type: str) -> float:
    """Seconds a job may run: COMFY_MAX_EXECUTION_SECONDS, else the job type default."""
    configured = os.environ.get("COMFY_MAX_EXECUTION_SECONDS")
    if configured is not None:
        try:
            value = float(configured)
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value > 0:
            return value
        logger.warning("COMFY_MAX_EXECUTION_SECONDS is invalid; using the job type default")
    if job_type.startswith("dbee-") and job_type not in DBEE_FULL_CAP_JOB_TYPES:
        return DBEE_EXECUTION_LIMIT_SECONDS
    return DEFAULT_EXECUTION_LIMIT_SECONDS


class ComfyClient:
    """Small client for ComfyUI's local prompt and history API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        poll_interval: Optional[float] = None,
        request_timeout: Optional[float] = None,
        max_execution_seconds: Optional[float] = None,
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
        self._max_execution_seconds = (
            execution_limit_seconds("")
            if max_execution_seconds is None
            else max_execution_seconds
        )
        self._session = requests.Session()

    def execute(
        self, workflow: dict, client_id: str, max_seconds: Optional[float] = None
    ) -> dict:
        """Submit a workflow and return its completed ComfyUI history entry.

        Raises ComfyExecutionTimeout if it has not completed within max_seconds
        (default: the client's limit), after logging diagnostics and interrupting it.
        """
        limit = self._max_execution_seconds if max_seconds is None else max_seconds
        deadline = time.monotonic() + limit
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
            if time.monotonic() >= deadline:
                self._abandon(client_id, prompt_id, limit)
            time.sleep(self._poll_interval)

    def _abandon(self, run_id: str, prompt_id: str, limit: float) -> None:
        # Capture the stalled state first: interrupting changes what ComfyUI reports.
        log_hang_diagnostics(run_id, prompt_id, self._get_json)
        try:
            self._request("post", "/interrupt")
        except ComfyExecutionError:
            pass  # Best effort; the handler asks RunPod to replace this worker next.
        raise ComfyExecutionTimeout(f"ComfyUI job exceeded {limit:g} seconds without completing")

    def _get_json(self, path: str) -> object:
        return self._json(self._request("get", path))

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
            payload = message[1] if len(message) > 1 else None
            detail = _message_from(payload) or f"ComfyUI {message[0]}"
            node_id = _safe_node_id(payload.get("node_id")) if isinstance(payload, dict) else None
            return f"ComfyUI node {node_id}: {detail}" if node_id is not None else detail
    return None


def _safe_node_id(value: object) -> Optional[str]:
    """Accept only bounded scalar identifiers, never nested error payload data."""
    if type(value) is int and 0 <= value < 10**128:
        value = str(value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        return value
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
