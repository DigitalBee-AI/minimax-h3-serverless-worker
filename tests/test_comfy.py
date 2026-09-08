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
            encoded = json.dumps(response).encode()
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
    responses.prompt_response = {"node_errors": {"12": {"errors": [{"message": "invalid image"}]}}}

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="invalid image"):
            ComfyClient(base_url, poll_interval=0.001, request_timeout=1).execute({}, "client-7")


def test_execute_reports_comfy_execution_errors() -> None:
    responses = ComfyResponses()
    responses.history_responses = [
        {
            "prompt-1": {
                "status": {
                    "completed": False,
                    "messages": [["execution_error", {"exception_message": "sampler failed"}]],
                }
            }
        }
    ]

    with comfy_server(responses) as base_url:
        with pytest.raises(ComfyExecutionError, match="sampler failed"):
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
