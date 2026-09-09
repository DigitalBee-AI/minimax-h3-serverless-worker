from __future__ import annotations

from pathlib import Path

import pytest

from worker.contracts import AssetSpec, JobRequest
from worker.storage import (
    PublishedVideo,
    StorageError,
    find_output_video,
    publish_video,
    staged_assets,
)


RUN_ID = "018f-example-id"


def make_request() -> JobRequest:
    return JobRequest(
        run_id=RUN_ID,
        workflow={},
        assets=(
            AssetSpec("image", f"jobs/{RUN_ID}/input/image.png", "image.png"),
            AssetSpec("video", f"jobs/{RUN_ID}/input/source.mp4", "source.mp4"),
        ),
        output_node_id="42",
    )


def make_volume(tmp_path: Path) -> Path:
    root = tmp_path / "volume"
    input_root = root / "jobs" / RUN_ID / "input"
    input_root.mkdir(parents=True)
    (input_root / "image.png").write_bytes(b"image bytes")
    (input_root / "source.mp4").write_bytes(b"video bytes")
    return root


def make_history(filename: str = "minimaxh3_00001.mp4", subfolder: str = "") -> dict:
    return {
        "outputs": {
            "42": {
                "gifs": [
                    {"filename": filename, "subfolder": subfolder, "type": "output"}
                ]
            }
        }
    }


def test_stages_both_assets_and_removes_them_on_exit(tmp_path: Path) -> None:
    volume = make_volume(tmp_path)
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()

    with staged_assets(make_request(), volume, comfy_input):
        assert (comfy_input / "image.png").read_bytes() == b"image bytes"
        assert (comfy_input / "source.mp4").read_bytes() == b"video bytes"

    assert not (comfy_input / "image.png").exists()
    assert not (comfy_input / "source.mp4").exists()


def test_staging_does_not_overwrite_an_existing_destination(tmp_path: Path) -> None:
    volume = make_volume(tmp_path)
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()
    destination = comfy_input / "image.png"
    destination.write_bytes(b"keep me")

    with pytest.raises(StorageError, match="already exists"):
        with staged_assets(make_request(), volume, comfy_input):
            pass

    assert destination.read_bytes() == b"keep me"
    assert not (comfy_input / "source.mp4").exists()


def test_staging_cleans_up_when_the_body_raises(tmp_path: Path) -> None:
    volume = make_volume(tmp_path)
    comfy_input = tmp_path / "comfy-input"
    comfy_input.mkdir()

    with pytest.raises(RuntimeError, match="workflow failed"):
        with staged_assets(make_request(), volume, comfy_input):
            raise RuntimeError("workflow failed")

    assert list(comfy_input.iterdir()) == []


def test_finds_videohelpersuite_mp4_in_nested_gifs(tmp_path: Path) -> None:
    output = tmp_path / "comfy-output"
    output.mkdir()
    video = output / "minimaxh3_00001.mp4"
    video.write_bytes(b"mp4")

    assert find_output_video(make_history(), "42", output) == video


@pytest.mark.parametrize(
    ("filename", "subfolder"),
    [
        ("../escape.mp4", ""),
        ("movie.mp4", "../escape"),
        ("/absolute.mp4", ""),
        ("movie.mov", ""),
    ],
)
def test_rejects_unsafe_or_non_mp4_output_candidate(
    filename: str, subfolder: str, tmp_path: Path
) -> None:
    output = tmp_path / "comfy-output"
    output.mkdir()

    with pytest.raises(StorageError):
        find_output_video(make_history(filename, subfolder), "42", output)


def test_rejects_symlinked_or_missing_output_candidate(tmp_path: Path) -> None:
    output = tmp_path / "comfy-output"
    output.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"mp4")
    (output / "minimaxh3_00001.mp4").symlink_to(outside)

    with pytest.raises(StorageError):
        find_output_video(make_history(), "42", output)

    (output / "minimaxh3_00001.mp4").unlink()
    with pytest.raises(StorageError):
        find_output_video(make_history(), "42", output)


def test_rejects_multiple_mp4_candidates(tmp_path: Path) -> None:
    output = tmp_path / "comfy-output"
    output.mkdir()
    (output / "one.mp4").write_bytes(b"one")
    (output / "two.mp4").write_bytes(b"two")
    history = {"outputs": {"42": {"gifs": [
        {"filename": "one.mp4", "type": "output"},
        {"filename": "two.mp4", "type": "output"},
    ]}}}

    with pytest.raises(StorageError, match="exactly one"):
        find_output_video(history, "42", output)


def test_publishes_result_with_exact_metadata_and_no_temporary_file(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"final video bytes")

    result = publish_video(source, volume, RUN_ID)
    destination = volume / "jobs" / RUN_ID / "output" / "result.mp4"

    assert result == PublishedVideo(
        volume_path=f"jobs/{RUN_ID}/output/result.mp4",
        filename="result.mp4",
        size_bytes=len(b"final video bytes"),
    )
    assert destination.read_bytes() == b"final video bytes"
    assert list(destination.parent.iterdir()) == [destination]


def test_republication_replaces_the_previous_result(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second and final")

    publish_video(first, volume, RUN_ID)
    result = publish_video(second, volume, RUN_ID)

    destination = volume / "jobs" / RUN_ID / "output" / "result.mp4"
    assert destination.read_bytes() == b"second and final"
    assert result.size_bytes == len(b"second and final")


def test_rejects_traversal_run_id_when_publishing(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    with pytest.raises(StorageError, match="run_id is invalid"):
        publish_video(source, volume, "..")

    assert not (volume / "output").exists()


@pytest.mark.parametrize("prefix", ["", "jobs", f"jobs/{RUN_ID}", f"jobs/{RUN_ID}/output"])
@pytest.mark.parametrize("existing_result", [False, True])
def test_publication_rejects_symlinked_directories_without_outside_writes(
    tmp_path: Path, prefix: str, existing_result: bool
) -> None:
    volume = tmp_path / "volume"
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_directory = volume / prefix
    linked_directory.parent.mkdir(parents=True, exist_ok=True)
    linked_directory.symlink_to(outside, target_is_directory=True)
    remaining_parts = Path(f"jobs/{RUN_ID}/output").parts[len(Path(prefix).parts):]
    outside_result = outside.joinpath(*remaining_parts, "result.mp4")
    if existing_result:
        outside_result.parent.mkdir(parents=True, exist_ok=True)
        outside_result.write_bytes(b"preserve outside result")
    before = sorted(path.relative_to(outside) for path in outside.rglob("*"))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"new result")

    with pytest.raises(StorageError, match="symlink"):
        publish_video(source, volume, RUN_ID)

    assert sorted(path.relative_to(outside) for path in outside.rglob("*")) == before
    if existing_result:
        assert outside_result.read_bytes() == b"preserve outside result"


def test_publication_rejects_a_symlink_even_when_its_target_is_inside_volume(
    tmp_path: Path,
) -> None:
    volume = make_volume(tmp_path)
    target = volume / "other-output"
    target.mkdir()
    (volume / "jobs" / RUN_ID / "output").symlink_to(target, target_is_directory=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"new result")

    with pytest.raises(StorageError, match="symlink"):
        publish_video(source, volume, RUN_ID)

    assert list(target.iterdir()) == []


def test_publication_rejects_symlinked_destination_without_overwriting_target(
    tmp_path: Path,
) -> None:
    volume = make_volume(tmp_path)
    output = volume / "jobs" / RUN_ID / "output"
    output.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"preserve outside")
    destination = output / "result.mp4"
    destination.symlink_to(outside)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"new result")

    with pytest.raises(StorageError, match="symlink"):
        publish_video(source, volume, RUN_ID)

    assert outside.read_bytes() == b"preserve outside"
    assert destination.is_symlink()
    assert list(output.iterdir()) == [destination]


def test_failed_retry_preserves_published_result_and_cleans_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = make_volume(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first result")
    publish_video(source, volume, RUN_ID)
    destination = volume / "jobs" / RUN_ID / "output" / "result.mp4"
    source.write_bytes(b"retry result")

    def failed_copy(source_path: Path, temporary_path: Path) -> None:
        temporary_path.write_bytes(b"partial")
        assert destination.read_bytes() == b"first result"
        raise OSError("copy failed")

    with monkeypatch.context() as patch:
        patch.setattr("worker.storage.shutil.copyfile", failed_copy)
        with pytest.raises(OSError, match="copy failed"):
            publish_video(source, volume, RUN_ID)

    assert destination.read_bytes() == b"first result"
    assert list(destination.parent.iterdir()) == [destination]
    result = publish_video(source, volume, RUN_ID)
    assert destination.read_bytes() == b"retry result"
    assert result.size_bytes == len(b"retry result")
    assert list(destination.parent.iterdir()) == [destination]
