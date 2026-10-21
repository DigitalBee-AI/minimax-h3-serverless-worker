"""Best-effort diagnostics, without media, prompts or storage paths."""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Callable

from worker.contracts import JobRequest

logger = logging.getLogger(__name__)

GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
]
LOG_TAIL_LINES = 30
LOG_LINE_LIMIT = 300

LINKS = {
    "11": ("ref_images.ref_image_0", "ref_videos.ref_video_0"),
    "46": ("image",),
    "42": ("images", "audio"),
}


def log_inputs(request: JobRequest, root: Path, *, staged: bool) -> None:
    if request.job_type != "dance":
        return
    phase = "staged" if staged else "source"
    try:
        assets = []
        for index, asset in enumerate(request.assets):
            path = root / (asset.comfy_name if staged else asset.volume_path)
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            assets.append({"index": index, "sha256": digest.hexdigest(), "bytes": size})
        links = {}
        for node_id, names in LINKS.items():
            node = request.workflow.get(node_id, {})
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            for name in names:
                value = inputs.get(name) if isinstance(inputs, dict) else None
                # Only numeric node IDs/socket indexes can reach logs. Never
                # serialize arbitrary workflow input values, even in errors.
                if (isinstance(value, list) and len(value) == 2
                        and isinstance(value[0], str)
                        and value[0].isascii() and value[0].isdigit()
                        and len(value[0]) <= 10
                        and type(value[1]) is int and 0 <= value[1] <= 100):
                    links[f"{node_id}.{name}"] = value
        workflow = json.dumps(request.workflow, sort_keys=True, separators=(",", ":"))
        payload = {"phase": phase, "assets": assets, "links": links,
                   "workflow_sha256": hashlib.sha256(workflow.encode()).hexdigest()}
        logger.info("run_id=%s diagnostics=%s", request.run_id, json.dumps(payload))
    except Exception:
        # Diagnostics must neither interrupt a render nor expose exception
        # messages that might contain a path, prompt or credential.
        logger.warning("run_id=%s diagnostics_unavailable phase=%s", request.run_id, phase)


def log_hang_diagnostics(
    run_id: str, prompt_id: str, get_json: Callable[[str], object]
) -> None:
    """Record what the GPU and ComfyUI were doing when a job hit its time limit.

    Never raises and never logs the workflow: only counts, GPU readings and the
    tail of ComfyUI's own log, which RunPod already captures. Each fact is its own
    short line: RunPod's log pipeline blanked a single ~7 KB line in a live test.
    """
    try:
        logger.error("run_id=%s hang_diagnostics gpu=%s", run_id, json.dumps(_gpu_state()))
        logger.error(
            "run_id=%s hang_diagnostics queue=%s",
            run_id,
            json.dumps(_queue_state(get_json, prompt_id)),
        )
        for line in _log_tail(get_json):
            logger.error("run_id=%s hang_diagnostics log=%s", run_id, json.dumps(line))
    except Exception:
        logger.warning("run_id=%s hang_diagnostics_unavailable", run_id)


def _gpu_state() -> list[str]:
    try:
        result = subprocess.run(
            GPU_QUERY, capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()][:8]
    except Exception:
        return []


def _queue_state(get_json: Callable[[str], object], prompt_id: str) -> dict | None:
    try:
        queue = get_json("/queue")
        if not isinstance(queue, dict):
            return None
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        running = running if isinstance(running, list) else []
        pending = pending if isinstance(pending, list) else []
        return {
            "running": len(running),
            "pending": len(pending),
            "prompt_running": any(
                isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id
                for item in running
            ),
        }
    except Exception:
        return None


def _log_tail(get_json: Callable[[str], object]) -> list[str]:
    try:
        payload = get_json("/internal/logs/raw")
        entries = payload.get("entries") if isinstance(payload, dict) else None
        lines: list[str] = []
        for entry in entries if isinstance(entries, list) else []:
            message = entry.get("m") if isinstance(entry, dict) else None
            if not isinstance(message, str):
                continue
            for raw_line in message.replace("\r\n", "\n").split("\n"):
                # Progress bars redraw with carriage returns; keep only the latest frame.
                line = raw_line.rsplit("\r", 1)[-1].strip()
                if line:
                    lines.append(line[:LOG_LINE_LIMIT])
        return lines[-LOG_TAIL_LINES:]
    except Exception:
        return []
