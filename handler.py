from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Protocol

import runpod

from worker.comfy import ComfyClient
from worker.contracts import parse_job_input
from worker.storage import find_output_video, publish_video, staged_assets


VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume"))
COMFY_INPUT_ROOT = Path(os.environ.get("COMFY_INPUT_ROOT", "/comfyui/input"))
COMFY_OUTPUT_ROOT = Path(os.environ.get("COMFY_OUTPUT_ROOT", "/comfyui/output"))

logger = logging.getLogger(__name__)


class WorkflowClient(Protocol):
    def execute(self, workflow: dict, client_id: str) -> dict: ...


def build_handler(
    client: WorkflowClient,
    volume_root: Path,
    comfy_input_root: Path,
    comfy_output_root: Path,
) -> Callable[[dict], dict]:
    """Compose the validated media lifecycle into a RunPod handler."""

    def handle(job: dict) -> dict:
        request = parse_job_input(job["input"], volume_root)
        _log_lifecycle(request.run_id, "validated")

        with staged_assets(request, volume_root, comfy_input_root):
            _log_lifecycle(request.run_id, "executing")
            history = client.execute(request.workflow, request.run_id)
            _log_lifecycle(request.run_id, "publishing")
            source = find_output_video(
                history, request.output_node_id, comfy_output_root
            )
            published = publish_video(source, volume_root, request.run_id)

        _log_lifecycle(request.run_id, "completed")
        return {
            "status": "success",
            "run_id": request.run_id,
            "video": {
                "volume_path": published.volume_path,
                "filename": published.filename,
                "size_bytes": published.size_bytes,
            },
        }

    return handle


def _log_lifecycle(run_id: str, state: str) -> None:
    logger.info("run_id=%s state=%s", run_id, state)


handler = build_handler(
    ComfyClient(), VOLUME_ROOT, COMFY_INPUT_ROOT, COMFY_OUTPUT_ROOT
)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
