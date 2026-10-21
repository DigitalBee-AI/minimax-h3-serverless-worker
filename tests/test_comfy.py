from __future__ import annotations

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time
from types import SimpleNamespace
from typing import Iterator

import pytest

import logging

import worker.diagnostics as diagnostics
from worker.comfy import (
    ComfyClient,
    ComfyExecutionError,
    ComfyExecutionTimeout,
    execution_limit_seconds,
)


class ComfyResponses:
    def __init__(self) -> None:
        self.prompt_status = 200
        self.prompt_response: object = {"prompt_id": "prompt-1"}
        self.history_responses: list[object] = [
            {},
            {
                "prompt-1": {
                    "status": {"completed": True, "messages": []},
                    "outputs": {"42": {}},
                }
            },
        ]
        self.history_delay = 0.0
        self.history_delays: list[float] = []
        self.requests: list[tuple[str, str, object | None]] = []
        self.diagnostics_status = 200
        self.queue_response: object = {"queue_running": [], "queue_pending": []}
        self.logs_response: object = {"entries": [], "size": {"cols": 80, "rows": 24}}


@contextmanager
def comfy_server(responses: ComfyResponses) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            responses.requests.append(("POST", self.path, json.loads(body) if body else None))
            self._send(responses.prompt_status, responses.prompt_response)

        def do_GET(self) -> None:  # noqa: N802
            responses.requests.append(("GET", self.path, None))
            if self.path == "/queue":
                self._send(responses.diagnostics_status, responses.queue_response)
                return
            if self.path == "/internal/logs/raw":
                self._send(responses.diagnostics_status, responses.logs_response)
                return
            delay = (
                responses.history_delays.pop(0)
                if responses.history_delays
                else responses.history_delay
            )
            if delay:
                time.sleep(delay)
            response = responses.history_responses.pop(0)
            self._send(200, response)

        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, status: int, response: object) -> None:
            encoded = response if isinstance(response, bytes) else json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_execute_submits_the_workflow_and_returns_completed_history() -> None:
    responses = ComfyResponses()
    workflow = {"42": {"class_type": "SaveImage", "inputs": {"filename_prefix": "run"}}}

    with comfy_server(responses) as base_url:
        result = ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
            workflow, "client-7"
        )

    assert responses.requests == [
        ("POST", "/prompt", {"prompt": workflow, "client_id": "client-7"}),
        ("GET", "/history/prompt-1", None),
        ("GET", "/history/prompt-1", None),
    ]
    assert result == {"status": {"completed": True, "messages": []}, "outputs": {"42": {}}}


def test_execute_reports_comfy_validation_errors() -> None:
    responses = ComfyResponses()
    responses.prompt_status = 400
    responses.prompt_response = {
        "node_errors": {
            "unrelated": "do not report this",
            "12": {"errors": [{"message": "invalid image"}]},
        }
    }

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="invalid image"):
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute({}, "client-7")


def test_execute_reports_comfy_execution_errors() -> None:
    responses = ComfyResponses()
    responses.history_responses = [
        {
            "prompt-1": {
                "status": {
                    "completed": True,
                    "messages": [["execution_error", {"exception_message": "sampler failed"}]],
                }
            }
        }
    ]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="sampler failed"):
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute({}, "client-7")


@pytest.mark.parametrize("node_id", ["42", 42, 0, "sampler-12"])
@pytest.mark.parametrize("event", ["execution_error", "execution_interrupted"])
def test_execute_includes_safe_failing_node_id(node_id: object, event: str) -> None:
    responses = ComfyResponses()
    responses.history_responses = [{"prompt-1": {"status": {
        "completed": False,
        "messages": [[event, {
            "node_id": node_id,
            "exception_message": "sampler failed",
            "workflow": {"private": "never expose workflow"},
            "arbitrary": "never expose arbitrary value",
        }]],
    }}}]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError) as error:
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
                {"private": "never expose request"}, "client-7"
            )

    assert str(error.value) == f"ComfyUI node {node_id}: sampler failed"


@pytest.mark.parametrize("node_id", [
    {"node_id": "42", "private": "nested secret"}, ["42", "nested secret"],
    None, True, False, 42.5, -1, "", "42\nsecret", "42\x1b[31m", "../../secret",
    "node with spaces", "x" * 129,
])
def test_execute_excludes_unsafe_node_id_and_arbitrary_values(node_id: object) -> None:
    responses = ComfyResponses()
    responses.history_responses = [{"prompt-1": {"status": {
        "completed": True,
        "messages": [["execution_error", {
            "node_id": node_id,
            "exception_message": "sampler failed",
            "workflow": {"node_id": "42", "private": "workflow secret"},
            "arbitrary": {"message": "arbitrary secret"},
        }]],
    }}}]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError) as error:
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
                {"private": "request secret"}, "client-7"
            )

    assert str(error.value) == "sampler failed"


def test_execute_url_encodes_prompt_id_for_history_requests() -> None:
    responses = ComfyResponses()
    prompt_id = "prompt/id 1"
    responses.prompt_response = {"prompt_id": prompt_id}
    responses.history_responses = [
        {prompt_id: {"status": {"completed": True, "messages": []}, "outputs": {}}}
    ]

    with comfy_server(responses) as base_url:
        result = ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
            {}, "client-7"
        )

    assert responses.requests[-1] == ("GET", "/history/prompt%2Fid%201", None)
    assert result == {"status": {"completed": True, "messages": []}, "outputs": {}}


@pytest.mark.parametrize("prompt_response", [{}, {"prompt_id": ""}])
def test_execute_rejects_missing_or_empty_prompt_ids(prompt_response: object) -> None:
    responses = ComfyResponses()
    responses.prompt_response = prompt_response

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="did not include a prompt_id") as error:
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
                {"secret": "workflow must stay private"}, "client-7"
            )

    assert "workflow must stay private" not in str(error.value)


def test_execute_uses_safe_fallback_for_unrecognized_execution_error_details() -> None:
    responses = ComfyResponses()
    responses.history_responses = [
        {
            "prompt-1": {
                "status": {
                    "completed": True,
                    "messages": [["execution_error", {"unrelated": "sensitive value"}]],
                }
            }
        }
    ]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="ComfyUI execution_error") as error:
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
                {"secret": "workflow must stay private"}, "client-7"
            )

    assert "sensitive value" not in str(error.value)
    assert "workflow must stay private" not in str(error.value)


def test_execute_reports_generic_http_failures_without_workflow_content() -> None:
    responses = ComfyResponses()
    responses.prompt_status = 500
    responses.prompt_response = {"detail": "internal detail"}

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="HTTP 500") as error:
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute(
                {"secret": "workflow must stay private"}, "client-7"
            )

    assert "internal detail" not in str(error.value)
    assert "workflow must stay private" not in str(error.value)


def test_execute_reports_invalid_json() -> None:
    responses = ComfyResponses()
    responses.prompt_response = b"not valid json"

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="invalid JSON"):
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute({}, "client-7")


def test_execute_reports_comfy_interruption() -> None:
    responses = ComfyResponses()
    responses.history_responses = [
        {
            "prompt-1": {
                "status": {
                    "completed": False,
                    "messages": [["execution_interrupted", {"message": "cancelled"}]],
                }
            }
        }
    ]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="cancelled"):
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute({}, "client-7")


def test_execute_reports_history_request_timeouts() -> None:
    responses = ComfyResponses()
    responses.history_delays = [0, 0.1]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="request timed out"):
            ComfyClient(base_url, poll_interval=0.001, request_timeout=0.01).execute({}, "client-7")


def stalled_responses() -> ComfyResponses:
    responses = ComfyResponses()
    responses.history_responses = [{} for _ in range(5000)]
    return responses


def fake_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str | None) -> None:
    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        if stdout is None:
            raise FileNotFoundError("nvidia-smi")
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(diagnostics.subprocess, "run", run)


def hang_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if " hang_diagnostics " in r.getMessage()]


def hang_payload(caplog: pytest.LogCaptureFixture) -> dict:
    """Reassemble the short `run_id=... hang_diagnostics <section>=<json>` lines."""
    payload: dict = {"run_id": None, "gpu": None, "queue": "missing", "comfy_log_tail": []}
    messages = hang_messages(caplog)
    assert messages, "no hang_diagnostics lines were logged"
    for message in messages:
        head, rest = message.split(" hang_diagnostics ", 1)
        section, value = rest.split("=", 1)
        payload["run_id"] = head.removeprefix("run_id=")
        if section == "log":
            payload["comfy_log_tail"].append(json.loads(value))
        else:
            payload[section] = json.loads(value)
    assert payload["gpu"] is not None and payload["queue"] != "missing", "gpu and queue lines are always logged"
    return payload


def timing_out_client(base_url: str, limit: float = 0.05) -> ComfyClient:
    return ComfyClient(
        base_url, poll_interval=0.001, request_timeout=1, max_execution_seconds=limit
    )


def test_timeout_is_a_comfy_execution_error() -> None:
    assert issubclass(ComfyExecutionTimeout, ComfyExecutionError)


def test_execute_stops_a_stalled_job_at_the_limit_and_interrupts_comfyui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nvidia_smi(monkeypatch, None)
    responses = stalled_responses()
    started = time.monotonic()

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionTimeout, match="exceeded 0.05 seconds"):
            timing_out_client(base_url).execute({}, "client-7")

    assert time.monotonic() - started < 2
    assert ("POST", "/interrupt", None) in responses.requests


def test_execute_limit_can_be_set_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvidia_smi(monkeypatch, None)
    responses = stalled_responses()

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionTimeout, match="exceeded 0.05 seconds"):
            timing_out_client(base_url, limit=3600).execute({}, "client-7", max_seconds=0.05)


def test_job_finishing_before_the_limit_is_unaffected_by_it() -> None:
    responses = ComfyResponses()

    with comfy_server(responses) as base_url:
        result = timing_out_client(base_url, limit=30).execute({}, "client-7")

    assert result["status"]["completed"] is True
    assert not any(path == "/interrupt" for _, path, _ in responses.requests)


def test_timeout_logs_gpu_queue_and_log_tail_without_the_workflow(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_nvidia_smi(monkeypatch, "NVIDIA H100 NVL, 100, 90000, 95000, 61\n")
    responses = stalled_responses()
    responses.queue_response = {
        "queue_running": [[0, "prompt-1", {"secret": "workflow must stay private"}, {}, []]],
        "queue_pending": [[1, "other", {"secret": "another private workflow"}, {}, []]],
    }
    entries = [{"t": "2026-10-08 06:00:00", "m": f"line {index}\n"} for index in range(100)]
    entries.append({"t": "2026-10-08 06:00:01", "m": "\n"})
    entries.append({
        "t": "2026-10-08 06:00:02",
        "m": "\r  0%|          | 0/4\r 25%|##5       | 1/4 [01:59<05:59]\n",
    })
    responses.logs_response = {"entries": entries, "size": {"cols": 80, "rows": 24}}
    caplog.set_level(logging.INFO)

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionTimeout):
            timing_out_client(base_url).execute({}, "client-7")

    payload = hang_payload(caplog)
    assert payload["run_id"] == "client-7"
    assert payload["gpu"] == ["NVIDIA H100 NVL, 100, 90000, 95000, 61"]
    assert payload["queue"] == {"running": 1, "pending": 1, "prompt_running": True}
    tail = payload["comfy_log_tail"]
    assert len(tail) == 30
    assert tail[-1] == "25%|##5       | 1/4 [01:59<05:59]"
    assert tail[-2] == "line 99"
    assert "workflow must stay private" not in caplog.text
    assert "another private workflow" not in caplog.text


def test_timeout_log_lines_are_truncated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_nvidia_smi(monkeypatch, None)
    responses = stalled_responses()
    responses.logs_response = {"entries": [{"t": "t", "m": "x" * 5000 + "\n"}]}
    caplog.set_level(logging.INFO)

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionTimeout):
            timing_out_client(base_url).execute({}, "client-7")

    assert hang_payload(caplog)["comfy_log_tail"] == ["x" * 300]


def test_every_hang_diagnostics_line_stays_short_for_runpod_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # RunPod blanked a single ~7 KB log line; keep each line far below that.
    fake_nvidia_smi(monkeypatch, "NVIDIA H100 NVL, 100, 90000, 95000, 61\n")
    responses = stalled_responses()
    responses.logs_response = {"entries": [{"t": "t", "m": "y" * 5000 + "\n"} for _ in range(40)]}
    caplog.set_level(logging.INFO)

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionTimeout):
            timing_out_client(base_url).execute({}, "client-7")

    messages = hang_messages(caplog)
    assert len(messages) == 2 + 30  # gpu, queue and the last 30 ComfyUI log lines
    assert max(len(message) for message in messages) <= 500


def test_timeout_is_still_reported_when_diagnostics_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_nvidia_smi(monkeypatch, None)
    responses = stalled_responses()
    responses.diagnostics_status = 500
    caplog.set_level(logging.INFO)

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionTimeout, match="exceeded"):
            timing_out_client(base_url).execute({}, "client-7")

    payload = hang_payload(caplog)
    assert payload["gpu"] == []
    assert payload["queue"] is None
    assert payload["comfy_log_tail"] == []
    assert ("POST", "/interrupt", None) in responses.requests


def test_timeout_is_reported_when_comfyui_cannot_be_reached_after_the_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_nvidia_smi(monkeypatch, None)
    responses = stalled_responses()
    caplog.set_level(logging.INFO)

    with comfy_server(responses) as base_url:
        client = timing_out_client(base_url)
        original_request = client._request

        def request(method: str, path: str, **kwargs: object):
            if path in {"/interrupt", "/queue", "/internal/logs/raw"}:
                raise ComfyExecutionError("ComfyUI request failed")
            return original_request(method, path, **kwargs)

        client._request = request  # type: ignore[method-assign]
        with pytest.raises(ComfyExecutionTimeout, match="exceeded"):
            client.execute({}, "client-7")

    assert hang_payload(caplog)["queue"] is None


@pytest.mark.parametrize(
    ("job_type", "expected"),
    [
        ("dbee-ai-ugc", 1080.0),
        ("dbee-product-showcase", 1080.0),
        ("dbee-agent-text-to-video", 1080.0),
        ("dbee-agent-reference-to-video", 1080.0),
        ("dance", 3300.0),
        ("foodie", 3300.0),
    ],
)
def test_execution_limit_defaults_stay_below_the_runpod_caps(
    monkeypatch: pytest.MonkeyPatch, job_type: str, expected: float
) -> None:
    monkeypatch.delenv("COMFY_MAX_EXECUTION_SECONDS", raising=False)

    assert execution_limit_seconds(job_type) == expected


@pytest.mark.parametrize("job_type", ["dbee-ai-ugc", "dance"])
def test_execution_limit_can_be_overridden_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, job_type: str
) -> None:
    monkeypatch.setenv("COMFY_MAX_EXECUTION_SECONDS", "600")

    assert execution_limit_seconds(job_type) == 600.0


@pytest.mark.parametrize("configured", ["", "abc", "0", "-5", "nan", "inf"])
def test_invalid_execution_limit_falls_back_to_the_defaults(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    monkeypatch.setenv("COMFY_MAX_EXECUTION_SECONDS", configured)

    assert execution_limit_seconds("dbee-ai-ugc") == 1080.0
    assert execution_limit_seconds("dance") == 3300.0
