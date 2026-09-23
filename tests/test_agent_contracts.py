from __future__ import annotations

from pathlib import Path

import pytest

from worker.contracts import RequestValidationError, parse_job_input


RUN_ID = "dbee-agent-test"


def payload(tmp_path: Path, count: int) -> tuple[dict, Path]:
    root = tmp_path / "volume"
    inputs = root / "jobs" / RUN_ID / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    workflow = {
        "11": {
            "class_type": "MiniMaxH3ReferenceToVideo" if count else "MiniMaxH3ImageToVideo",
            "inputs": {},
        },
        "42": {"class_type": "VHS_VideoCombine", "inputs": {}},
    }
    assets = []
    for index in range(count):
        name = f"picture-{index + 1}.png"
        role = f"picture-{index + 1}"
        node = str(100 + index)
        (inputs / name).write_bytes(b"image")
        workflow[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        workflow["11"]["inputs"][f"ref_images.ref_image_{index}"] = [node, 0]
        assets.append({
            "kind": "image", "role": role, "comfy_name": name,
            "volume_path": f"jobs/{RUN_ID}/input/{name}",
        })
    return {
        "job_type": "dbee-agent-reference-to-video" if count else "dbee-agent-text-to-video",
        "run_id": RUN_ID,
        "workflow": workflow,
        "assets": assets,
        "output_node_id": "42",
    }, root


@pytest.mark.parametrize("count", [0, 1, 9])
def test_accepts_distinct_agent_workflows(tmp_path: Path, count: int) -> None:
    job, root = payload(tmp_path, count)
    request = parse_job_input(job, root)
    assert len(request.assets) == count
    assert request.job_type == job["job_type"]


def test_text_workflow_rejects_references(tmp_path: Path) -> None:
    job, root = payload(tmp_path, 1)
    job["job_type"] = "dbee-agent-text-to-video"
    with pytest.raises(RequestValidationError, match="no assets"):
        parse_job_input(job, root)


def test_reference_workflow_rejects_empty_and_ten_images(tmp_path: Path) -> None:
    job, root = payload(tmp_path, 0)
    job["job_type"] = "dbee-agent-reference-to-video"
    with pytest.raises(RequestValidationError, match="1-9 ordered"):
        parse_job_input(job, root)
    job, root = payload(tmp_path, 10)
    with pytest.raises(RequestValidationError, match="1-9 ordered"):
        parse_job_input(job, root)


def test_text_workflow_rejects_reference_node(tmp_path: Path) -> None:
    job, root = payload(tmp_path, 0)
    job["workflow"]["11"]["class_type"] = "MiniMaxH3ReferenceToVideo"
    with pytest.raises(RequestValidationError, match="MiniMaxH3ImageToVideo"):
        parse_job_input(job, root)


def test_reference_workflow_rejects_wrong_order(tmp_path: Path) -> None:
    job, root = payload(tmp_path, 2)
    job["workflow"]["11"]["inputs"]["ref_images.ref_image_0"] = ["101", 0]
    with pytest.raises(RequestValidationError, match="reference 0"):
        parse_job_input(job, root)
