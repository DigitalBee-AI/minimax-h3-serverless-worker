from __future__ import annotations

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time
from typing import Iterator

import pytest

from worker.comfy import ComfyClient, ComfyExecutionError


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


@contextmanager
def comfy_server(responses: ComfyResponses) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            responses.requests.append(("POST", self.path, json.loads(body)))
            self._send(responses.prompt_status, responses.prompt_response)

        def do_GET(self) -> None:  # noqa: N802
            responses.requests.append(("GET", self.path, None))
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
