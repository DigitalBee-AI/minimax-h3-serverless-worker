from __future__ import annotations

import logging
from pathlib import Path

import pytest

from handler import build_handler
from worker.comfy import ComfyExecutionError
from worker.contracts import RequestValidationError
from worker.storage import PublishedVideo, StorageError


RUN_ID = "018f-example-id"
IMAGE_NAME = "018f-example-id-kol.png"
VIDEO_NAME = "018f-example-id-source.mp4"
STORYBOARD_NAME = "storyboard.png"
VOICE_NAME = "voice.wav"
VIDEO_BYTES = b"real mp4 fixture"


def make_job() -> dict:
    return {
        "input": {
            "run_id": RUN_ID,
            "workflow": {
                "9": {"class_type": "LoadImage", "inputs": {"image": IMAGE_NAME}},
                "42": {"class_type": "VHS_VideoCombine", "inputs": {}},
                "43": {"class_type": "VHS_LoadVideo", "inputs": {"video": VIDEO_NAME}},
                "secret_node": {"inputs": {"workflow_secret": "top-secret"}},
            },
            "assets": [
                {
                    "kind": "image",
                    "volume_path": f"jobs/{RUN_ID}/input/{IMAGE_NAME}",
                    "comfy_name": IMAGE_NAME,
                },
                {
                    "kind": "video",
                    "volume_path": f"jobs/{RUN_ID}/input/{VIDEO_NAME}",
                    "comfy_name": VIDEO_NAME,
                },
            ],
            "output_node_id": "42",
        }
    }


def make_foodie_job() -> dict:
    return {
        "input": {
            "job_type": "foodie",
            "run_id": RUN_ID,
            "workflow": {
                "9": {"class_type": "LoadImage", "inputs": {"image": IMAGE_NAME}},
                "42": {"class_type": "VHS_VideoCombine", "inputs": {}},
                "48": {
                    "class_type": "LoadImage",
                    "inputs": {"image": STORYBOARD_NAME},
                },
                "50": {
                    "class_type": "LoadAudio",
                    "inputs": {"audio": VOICE_NAME},
                },
            },
            "assets": [
                {
                    "kind": "image",
                    "role": "kol",
                    "volume_path": f"jobs/{RUN_ID}/input/{IMAGE_NAME}",
                    "comfy_name": IMAGE_NAME,
                },
                {
                    "kind": "image",
                    "role": "storyboard",
                    "volume_path": f"jobs/{RUN_ID}/input/{STORYBOARD_NAME}",
                    "comfy_name": STORYBOARD_NAME,
                },
                {
                    "kind": "audio",
                    "role": "voice",
                    "volume_path": f"jobs/{RUN_ID}/input/{VOICE_NAME}",
                    "comfy_name": VOICE_NAME,
                },
            ],
            "output_node_id": "42",
        }
    }


def make_volume(tmp_path: Path) -> Path:
    volume = tmp_path / "volume"
    input_root = volume / "jobs" / RUN_ID / "input"
    input_root.mkdir(parents=True)
    (input_root / IMAGE_NAME).write_bytes(b"image fixture")
    (input_root / VIDEO_NAME).write_bytes(b"video fixture")
    return volume


class FakeClient:
    def __init__(
        self,
        output_root: Path,
        staged_names: tuple[str, ...] = (IMAGE_NAME, VIDEO_NAME),
    ) -> None:
        self.output_root = output_root
        self.staged_names = staged_names
        self.workflow: dict | None = None
        self.client_id: str | None = None
        self.staged_inputs_present = False

    def execute(self, workflow: dict, client_id: str) -> dict:
        self.workflow = workflow
        self.client_id = client_id
        self.staged_inputs_present = all(
            (self.output_root.parent / "comfy-input" / name).is_file()
            for name in self.staged_names
        )
        video = self.output_root / "minimaxh3_00001.mp4"
        video.write_bytes(VIDEO_BYTES)
        return {
            "outputs": {
                "42": {
                    "gifs": [
                        {
                            "filename": video.name,
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }
        }


def make_handler(tmp_path: Path) -> tuple[object, FakeClient, Path, Path]:
    volume = make_volume(tmp_path)
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    comfy_output = tmp_path / "comfy-output"
    comfy_output.mkdir()
    client = FakeClient(comfy_output)
    return (
        build_handler(client, volume, comfy_input, comfy_output),
        client,
        comfy_input,
        volume,
    )


def test_returns_published_video_contract_after_running_the_workflow(tmp_path: Path) -> None:
    handler, _, _, _ = make_handler(tmp_path)

    assert handler(make_job()) == {
        "status": "success",
        "run_id": RUN_ID,
        "video": {
            "volume_path": f"jobs/{RUN_ID}/output/result.mp4",
            "filename": "result.mp4",
            "size_bytes": len(VIDEO_BYTES),
        },
    }


def test_passes_the_untouched_workflow_and_run_id_to_comfyui(tmp_path: Path) -> None:
    handler, client, _, _ = make_handler(tmp_path)
    job = make_job()

    handler(job)

    assert client.workflow is job["input"]["workflow"]
    assert client.client_id == RUN_ID


def test_stages_assets_for_execution_and_removes_them_afterward(tmp_path: Path) -> None:
    handler, client, comfy_input, _ = make_handler(tmp_path)

    handler(make_job())

    assert client.staged_inputs_present is True
    assert list(comfy_input.iterdir()) == []


def test_foodie_job_stages_three_assets_and_returns_the_standard_video_contract(
    tmp_path: Path,
) -> None:
    volume = make_volume(tmp_path)
    input_root = volume / "jobs" / RUN_ID / "input"
    (input_root / STORYBOARD_NAME).write_bytes(b"storyboard fixture")
    (input_root / VOICE_NAME).write_bytes(b"voice fixture")
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    comfy_output = tmp_path / "comfy-output"
    comfy_output.mkdir()
    client = FakeClient(
        comfy_output,
        (IMAGE_NAME, STORYBOARD_NAME, VOICE_NAME),
    )
    handler = build_handler(client, volume, comfy_input, comfy_output)

    result = handler(make_foodie_job())

    assert client.staged_inputs_present is True
    assert list(comfy_input.iterdir()) == []
    assert result == {
        "status": "success",
        "run_id": RUN_ID,
        "video": {
            "volume_path": f"jobs/{RUN_ID}/output/result.mp4",
            "filename": "result.mp4",
            "size_bytes": len(VIDEO_BYTES),
        },
    }


def test_dbee_ai_ugc_stages_five_assets_through_the_shared_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_foodie_job()
    job["input"]["job_type"] = "dbee-ai-ugc"
    job["input"]["assets"][0]["role"] = "character"
    input_root_names = [IMAGE_NAME, STORYBOARD_NAME, VOICE_NAME]
    for role, node_id in [("product", "51"), ("environment", "52")]:
        filename = f"{role}.png"
        input_root_names.append(filename)
        job["input"]["assets"].append({
            "kind": "image",
            "role": role,
            "volume_path": f"jobs/{RUN_ID}/input/{filename}",
            "comfy_name": filename,
        })
        job["input"]["workflow"][node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
        }
    # The contract is role-based; asset order does not alter the graph mapping.
    job["input"]["assets"] = [
        next(asset for asset in job["input"]["assets"] if asset["role"] == role)
        for role in ["character", "product", "environment", "storyboard", "voice"]
    ]
    volume = make_volume(tmp_path)
    input_root = volume / "jobs" / RUN_ID / "input"
    for filename in [STORYBOARD_NAME, VOICE_NAME, "product.png", "environment.png"]:
        (input_root / filename).write_bytes(b"fixture")
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    comfy_output = tmp_path / "comfy-output"
    comfy_output.mkdir()
    client = FakeClient(comfy_output, tuple(input_root_names))
    monkeypatch.setattr(
        "handler.publish_video",
        lambda source, volume_root, run_id: PublishedVideo(
            volume_path=f"jobs/{run_id}/output/result.mp4",
            filename="result.mp4",
            size_bytes=source.stat().st_size,
        ),
    )

    result = build_handler(client, volume, comfy_input, comfy_output)(job)

    assert client.staged_inputs_present is True
    assert result["video"]["volume_path"] == f"jobs/{RUN_ID}/output/result.mp4"


def test_dbee_product_showcase_stages_ordered_assets_through_the_shared_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    filenames = [f"picture-{index}.png" for index in range(1, 4)]
    workflow = {
        "11": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {}},
        "42": {"class_type": "VHS_VideoCombine", "inputs": {}},
    }
    assets = []
    volume = tmp_path / "volume"
    input_root = volume / "jobs" / RUN_ID / "input"
    input_root.mkdir(parents=True)
    for index, filename in enumerate(filenames):
        node_id = str(100 + index)
        (input_root / filename).write_bytes(b"product image")
        workflow[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
        }
        workflow["11"]["inputs"][f"ref_images.ref_image_{index}"] = [node_id, 0]
        assets.append({
            "kind": "image",
            "role": f"picture-{index + 1}",
            "volume_path": f"jobs/{RUN_ID}/input/{filename}",
            "comfy_name": filename,
        })
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    comfy_output = tmp_path / "comfy-output"
    comfy_output.mkdir()
    client = FakeClient(comfy_output, tuple(filenames))
    monkeypatch.setattr(
        "handler.publish_video",
        lambda source, volume_root, run_id: PublishedVideo(
            volume_path=f"jobs/{run_id}/output/result.mp4",
            filename="result.mp4",
            size_bytes=source.stat().st_size,
        ),
    )

    result = build_handler(client, volume, comfy_input, comfy_output)({
        "input": {
            "job_type": "dbee-product-showcase",
            "run_id": RUN_ID,
            "workflow": workflow,
            "assets": assets,
            "output_node_id": "42",
        }
    })

    assert client.staged_inputs_present is True
    assert list(comfy_input.iterdir()) == []
    assert result["video"]["filename"] == "result.mp4"


@pytest.mark.parametrize(
    ("job", "client_factory", "expected_error"),
    [
        ({"input": {}}, lambda output: FakeClient(output), RequestValidationError),
        (
            make_job(),
            lambda output: RaisingClient(ComfyExecutionError("ComfyUI unavailable")),
            ComfyExecutionError,
        ),
    ],
)
def test_contract_and_comfyui_errors_escape_for_runpod(
    tmp_path: Path, job: dict, client_factory: object, expected_error: type[Exception]
) -> None:
    volume = make_volume(tmp_path)
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    comfy_output = tmp_path / "comfy-output"
    comfy_output.mkdir()
    handler = build_handler(client_factory(comfy_output), volume, comfy_input, comfy_output)

    with pytest.raises(expected_error):
        handler(job)


def test_storage_errors_escape_for_runpod(tmp_path: Path) -> None:
    handler, _, comfy_input, _ = make_handler(tmp_path)
    (comfy_input / IMAGE_NAME).write_bytes(b"already here")

    with pytest.raises(StorageError, match="already exists"):
        handler(make_job())


def test_logs_only_validated_run_id_and_lifecycle_state(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    handler, _, _, _ = make_handler(tmp_path)
    caplog.set_level(logging.INFO)

    handler(make_job())

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert RUN_ID in messages
    assert "state=" in messages
    assert "workflow_secret" not in messages
    assert "top-secret" not in messages
    assert "jobs/" not in messages


class RaisingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self, workflow: dict, client_id: str) -> dict:
        raise self.error
