"""Best-effort Dance fingerprints, without media, prompts or storage paths."""

import hashlib
import json
import logging
from pathlib import Path

from worker.contracts import JobRequest

logger = logging.getLogger(__name__)

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
