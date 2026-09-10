from __future__ import annotations

from pathlib import Path

import pytest

from worker.contracts import RequestValidationError, parse_job_input


RUN_ID = "018f-example-id"
IMAGE_NAME = "018f-example-id-kol.png"
VIDEO_NAME = "018f-example-id-source.mp4"
STORYBOARD_NAME = "storyboard.png"
VOICE_NAME = "voice.wav"


def make_volume(tmp_path: Path) -> Path:
    volume_root = tmp_path / "volume"
    input_root = volume_root / "jobs" / RUN_ID / "input"
    input_root.mkdir(parents=True)
    (input_root / IMAGE_NAME).write_bytes(b"image")
    (input_root / VIDEO_NAME).write_bytes(b"video")
    return volume_root


def make_payload() -> dict:
    return {
        "run_id": RUN_ID,
        "workflow": {
            "9": {"class_type": "LoadImage", "inputs": {"image": IMAGE_NAME}},
            "42": {"class_type": "VHS_VideoCombine", "inputs": {}},
            "43": {"class_type": "VHS_LoadVideo", "inputs": {"video": VIDEO_NAME}},
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


def make_foodie_volume(tmp_path: Path) -> Path:
    volume_root = tmp_path / "volume"
    input_root = volume_root / "jobs" / RUN_ID / "input"
    input_root.mkdir(parents=True)
    (input_root / IMAGE_NAME).write_bytes(b"kol image")
    (input_root / STORYBOARD_NAME).write_bytes(b"storyboard image")
    (input_root / VOICE_NAME).write_bytes(b"voice audio")
    return volume_root


def make_foodie_payload() -> dict:
    return {
        "job_type": "foodie",
        "run_id": RUN_ID,
        "workflow": {
            "9": {"class_type": "LoadImage", "inputs": {"image": IMAGE_NAME}},
            "42": {"class_type": "VHS_VideoCombine", "inputs": {}},
            "48": {
                "class_type": "LoadImage",
                "inputs": {"image": STORYBOARD_NAME},
            },
            "50": {"class_type": "LoadAudio", "inputs": {"audio": VOICE_NAME}},
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


def parse(payload: dict, tmp_path: Path):
    return parse_job_input(payload, make_volume(tmp_path))


def test_accepts_the_complete_matching_job_contract(tmp_path: Path) -> None:
    request = parse_job_input(make_payload(), make_volume(tmp_path))

    assert request.run_id == "018f-example-id"
    assert request.output_node_id == "42"
    assert [asset.kind for asset in request.assets] == ["image", "video"]
    assert request.workflow["9"]["inputs"]["image"] == "018f-example-id-kol.png"
    assert request.workflow["43"]["inputs"]["video"] == "018f-example-id-source.mp4"


def test_missing_job_type_remains_a_dance_job(tmp_path: Path) -> None:
    request = parse_job_input(make_payload(), make_volume(tmp_path))

    assert request.job_type == "dance"


def test_accepts_an_explicit_dance_job_type(tmp_path: Path) -> None:
    payload = make_payload()
    payload["job_type"] = "dance"

    request = parse_job_input(payload, make_volume(tmp_path))

    assert request.job_type == "dance"


def test_accepts_the_complete_foodie_job_contract(tmp_path: Path) -> None:
    request = parse_job_input(make_foodie_payload(), make_foodie_volume(tmp_path))

    assert request.job_type == "foodie"
    assert [(asset.kind, asset.role) for asset in request.assets] == [
        ("image", "kol"),
        ("image", "storyboard"),
        ("audio", "voice"),
    ]
    assert request.workflow["9"]["inputs"]["image"] == IMAGE_NAME
    assert request.workflow["48"]["inputs"]["image"] == STORYBOARD_NAME
    assert request.workflow["50"]["inputs"]["audio"] == VOICE_NAME


@pytest.mark.parametrize("job_type", [None, "fashion", 42])
def test_rejects_an_unknown_or_invalid_job_type(
    job_type: object, tmp_path: Path
) -> None:
    payload = make_payload()
    payload["job_type"] = job_type

    with pytest.raises(RequestValidationError, match="job_type is invalid"):
        parse_job_input(payload, make_volume(tmp_path))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda assets: assets.pop(),
        lambda assets: assets[1].update(role="kol"),
        lambda assets: assets[2].update(kind="video"),
        lambda assets: assets[0].pop("role"),
    ],
)
def test_rejects_incomplete_or_mislabeled_foodie_assets(
    mutate: object, tmp_path: Path
) -> None:
    payload = make_foodie_payload()
    mutate(payload["assets"])

    with pytest.raises(
        RequestValidationError,
        match="exactly one Foodie KOL image, storyboard image, and voice audio are required",
    ):
        parse_job_input(payload, make_foodie_volume(tmp_path))


def test_rejects_unsupported_foodie_audio_extension(tmp_path: Path) -> None:
    payload = make_foodie_payload()
    voice = payload["assets"][2]
    voice["volume_path"] = f"jobs/{RUN_ID}/input/voice.mp3"
    voice["comfy_name"] = "voice.mp3"
    payload["workflow"]["50"]["inputs"]["audio"] = "voice.mp3"
    volume_root = make_foodie_volume(tmp_path)
    (volume_root / "jobs" / RUN_ID / "input" / "voice.mp3").write_bytes(b"mp3")

    with pytest.raises(
        RequestValidationError, match="audio asset extension is unsupported"
    ):
        parse_job_input(payload, volume_root)


@pytest.mark.parametrize(
    ("node_id", "input_name", "value"),
    [
        ("9", "image", "different-kol.png"),
        ("48", "image", "different-storyboard.png"),
        ("50", "audio", "different-voice.wav"),
    ],
)
def test_rejects_foodie_workflow_filename_mismatch(
    node_id: str, input_name: str, value: str, tmp_path: Path
) -> None:
    payload = make_foodie_payload()
    payload["workflow"][node_id]["inputs"][input_name] = value

    with pytest.raises(
        RequestValidationError,
        match=f"workflow node {node_id} filename does not match asset",
    ):
        parse_job_input(payload, make_foodie_volume(tmp_path))


def test_rejects_missing_required_input(tmp_path: Path) -> None:
    payload = make_payload()
    del payload["workflow"]

    with pytest.raises(RequestValidationError, match="workflow is required"):
        parse(payload, tmp_path)


@pytest.mark.parametrize("run_id", ["", "-starts-with-dash", "contains space", "a" * 129])
def test_rejects_invalid_run_id(run_id: str, tmp_path: Path) -> None:
    payload = make_payload()
    payload["run_id"] = run_id

    with pytest.raises(RequestValidationError, match="run_id is invalid"):
        parse(payload, tmp_path)


@pytest.mark.parametrize("assets", [[], [make_payload()["assets"][0]], make_payload()["assets"] * 2])
def test_rejects_wrong_asset_count(assets: list[dict], tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"] = assets

    with pytest.raises(RequestValidationError, match="exactly one image and one video are required"):
        parse(payload, tmp_path)


def test_rejects_duplicate_asset_kind(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][1]["kind"] = "image"

    with pytest.raises(RequestValidationError, match="exactly one image and one video are required"):
        parse(payload, tmp_path)


def test_rejects_absolute_asset_path(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][0]["volume_path"] = "/runpod-volume/jobs/example/input/image.png"

    with pytest.raises(RequestValidationError, match="asset path must be relative"):
        parse(payload, tmp_path)


def test_rejects_asset_path_traversal(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][0]["volume_path"] = f"jobs/{RUN_ID}/input/../{IMAGE_NAME}"

    with pytest.raises(RequestValidationError, match="asset path escapes the job input directory"):
        parse(payload, tmp_path)


def test_rejects_asset_path_outside_job_input_directory(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][0]["volume_path"] = f"jobs/{RUN_ID}/output/{IMAGE_NAME}"

    with pytest.raises(RequestValidationError, match="asset path escapes the job input directory"):
        parse(payload, tmp_path)


def test_rejects_symlinked_asset(tmp_path: Path) -> None:
    volume_root = make_volume(tmp_path)
    input_root = volume_root / "jobs" / RUN_ID / "input"
    (input_root / IMAGE_NAME).unlink()
    target = tmp_path / "outside.png"
    target.write_bytes(b"outside")
    (input_root / IMAGE_NAME).symlink_to(target)

    with pytest.raises(RequestValidationError, match="asset source must not be a symlink"):
        parse_job_input(make_payload(), volume_root)


def test_rejects_symlinked_job_input_prefix(tmp_path: Path) -> None:
    volume_root = tmp_path / "volume"
    redirected_jobs = tmp_path / "outside" / "jobs"
    redirected_input = redirected_jobs / RUN_ID / "input"
    redirected_input.mkdir(parents=True)
    (redirected_input / IMAGE_NAME).write_bytes(b"image")
    (redirected_input / VIDEO_NAME).write_bytes(b"video")
    volume_root.mkdir()
    (volume_root / "jobs").symlink_to(redirected_jobs, target_is_directory=True)

    with pytest.raises(RequestValidationError, match="asset source must not be a symlink"):
        parse_job_input(make_payload(), volume_root)


def test_rejects_missing_asset_file(tmp_path: Path) -> None:
    volume_root = make_volume(tmp_path)
    (volume_root / "jobs" / RUN_ID / "input" / IMAGE_NAME).unlink()

    with pytest.raises(RequestValidationError, match="asset source file is missing"):
        parse_job_input(make_payload(), volume_root)


def test_rejects_unsupported_asset_extension(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][0]["volume_path"] = f"jobs/{RUN_ID}/input/photo.gif"
    payload["assets"][0]["comfy_name"] = "photo.gif"
    volume_root = make_volume(tmp_path)
    (volume_root / "jobs" / RUN_ID / "input" / "photo.gif").write_bytes(b"gif")

    with pytest.raises(RequestValidationError, match="image asset extension is unsupported"):
        parse_job_input(payload, volume_root)


def test_rejects_duplicate_comfy_names(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][1]["comfy_name"] = IMAGE_NAME

    with pytest.raises(RequestValidationError, match="asset comfy_name values must be unique"):
        parse(payload, tmp_path)


def test_rejects_non_basename_comfy_name(tmp_path: Path) -> None:
    payload = make_payload()
    payload["assets"][0]["comfy_name"] = "subdir/image.png"

    with pytest.raises(RequestValidationError, match="asset comfy_name is invalid"):
        parse(payload, tmp_path)


@pytest.mark.parametrize(
    ("node_id", "input_name", "value"),
    [("9", "image", "different.png"), ("43", "video", "different.mp4")],
)
def test_rejects_workflow_filename_mismatch(
    node_id: str, input_name: str, value: str, tmp_path: Path
) -> None:
    payload = make_payload()
    payload["workflow"][node_id]["inputs"][input_name] = value

    with pytest.raises(RequestValidationError, match=f"workflow node {node_id} filename does not match asset"):
        parse(payload, tmp_path)


def test_rejects_wrong_output_node_class(tmp_path: Path) -> None:
    payload = make_payload()
    payload["workflow"]["42"]["class_type"] = "SaveImage"

    with pytest.raises(RequestValidationError, match="workflow node 42 must be VHS_VideoCombine"):
        parse(payload, tmp_path)
