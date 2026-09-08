from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import ContextManager, Iterator

from worker.contracts import JobRequest, RUN_ID_PATTERN


class StorageError(RuntimeError):
    """Raised when worker storage cannot be accessed safely."""


@dataclass(frozen=True)
class PublishedVideo:
    volume_path: str
    filename: str
    size_bytes: int


@contextmanager
def staged_assets(
    request: JobRequest, volume_root: Path, comfy_input_root: Path
) -> Iterator[None]:
    """Copy validated job assets into ComfyUI's input directory temporarily."""
    destinations = [comfy_input_root / asset.comfy_name for asset in request.assets]
    if any(destination.exists() or destination.is_symlink() for destination in destinations):
        raise StorageError("ComfyUI input destination already exists")

    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for asset, destination in zip(request.assets, destinations):
            source = volume_root / asset.volume_path
            if not source.is_file() or source.is_symlink():
                raise StorageError("staged asset source is not a regular file")
            try:
                with destination.open("xb"):
                    pass
            except FileExistsError as exc:
                raise StorageError("ComfyUI input destination already exists") from exc
            destination_stat = destination.stat()
            created.append((destination, (destination_stat.st_dev, destination_stat.st_ino)))
            shutil.copy2(source, destination)
        yield
    finally:
        for destination, identity in reversed(created):
            try:
                destination_stat = destination.lstat()
            except FileNotFoundError:
                continue
            if (
                stat.S_ISREG(destination_stat.st_mode)
                and (destination_stat.st_dev, destination_stat.st_ino) == identity
            ):
                destination.unlink()


def find_output_video(
    history: dict, output_node_id: str, comfy_output_root: Path
) -> Path:
    """Locate the sole regular MP4 reported by the requested ComfyUI node."""
    try:
        node_output = history["outputs"][output_node_id]
    except (KeyError, TypeError) as exc:
        raise StorageError("requested output node is missing from history") from exc

    root = comfy_output_root.resolve()
    candidates: list[Path] = []
    for value in _walk(node_output):
        if not (
            isinstance(value, dict)
            and value.get("type") == "output"
            and isinstance(value.get("filename"), str)
            and value["filename"].endswith(".mp4")
        ):
            continue
        candidates.append(_resolve_output_candidate(value, comfy_output_root, root))

    if len(candidates) != 1:
        raise StorageError("expected exactly one MP4 output candidate")
    return candidates[0]


def _walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


def _resolve_output_candidate(candidate: dict, lexical_root: Path, root: Path) -> Path:
    filename = candidate["filename"]
    subfolder = candidate.get("subfolder", "")
    if not isinstance(subfolder, str):
        raise StorageError("output subfolder is invalid")
    filename_path = Path(filename)
    subfolder_path = Path(subfolder)
    if (
        filename_path.is_absolute()
        or filename_path.name != filename
        or "\\" in filename
        or subfolder_path.is_absolute()
        or ".." in subfolder_path.parts
        or "\\" in subfolder
    ):
        raise StorageError("output path is unsafe")

    path = lexical_root / subfolder_path / filename_path
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise StorageError("output path escapes ComfyUI output directory") from exc

    relative_path = path.relative_to(lexical_root)
    component = lexical_root
    for part in relative_path.parts:
        component /= part
        if component.is_symlink():
            raise StorageError("output file must not be a symlink")
    if not path.is_file():
        raise StorageError("output file is missing or not a regular file")
    return path


def publish_video(source: Path, volume_root: Path, run_id: str) -> PublishedVideo:
    """Atomically publish a ComfyUI result into the job's private output area."""
    if not source.is_file() or source.is_symlink():
        raise StorageError("video source is not a regular file")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise StorageError("run_id is invalid")

    output_directory = volume_root / "jobs" / run_id / "output"
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "result.mp4"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".result-", suffix=".tmp", dir=output_directory
    )
    temporary = Path(temporary_name)
    try:
        os.close(file_descriptor)
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()

    return PublishedVideo(
        volume_path=f"jobs/{run_id}/output/result.mp4",
        filename="result.mp4",
        size_bytes=destination.stat().st_size,
    )
