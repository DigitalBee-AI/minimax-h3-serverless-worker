from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


class RequestValidationError(ValueError):
    """Raised when a RunPod job does not satisfy the worker contract."""


@dataclass(frozen=True)
class AssetSpec:
    kind: str
    volume_path: str
    comfy_name: str


@dataclass(frozen=True)
class JobRequest:
    run_id: str
    workflow: dict
    assets: tuple[AssetSpec, AssetSpec]
    output_node_id: str


def parse_job_input(value: object, volume_root: Path) -> JobRequest:
    if not isinstance(value, dict):
        raise RequestValidationError("input must be a dictionary")

    run_id = _required_string(value, "run_id")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RequestValidationError("run_id is invalid")

    workflow = value.get("workflow")
    if not isinstance(workflow, dict):
        raise RequestValidationError("workflow is required")

    output_node_id = _required_string(value, "output_node_id")
    if output_node_id != "42":
        raise RequestValidationError("output_node_id must be 42")

    raw_assets = value.get("assets")
    if not isinstance(raw_assets, (list, tuple)) or len(raw_assets) != 2:
        raise RequestValidationError("exactly one image and one video are required")

    assets = tuple(_parse_asset(asset) for asset in raw_assets)
    if {asset.kind for asset in assets} != {"image", "video"}:
        raise RequestValidationError("exactly one image and one video are required")
    if len({asset.comfy_name for asset in assets}) != len(assets):
        raise RequestValidationError("asset comfy_name values must be unique")

    input_root = _resolve_job_input_root(volume_root, run_id)
    for asset in assets:
        _validate_asset_source(asset, input_root)

    image = next(asset for asset in assets if asset.kind == "image")
    video = next(asset for asset in assets if asset.kind == "video")
    _validate_workflow(workflow, image, video)

    return JobRequest(
        run_id=run_id,
        workflow=workflow,
        assets=(assets[0], assets[1]),
        output_node_id=output_node_id,
    )


def _required_string(value: dict, name: str) -> str:
    field = value.get(name)
    if not isinstance(field, str):
        raise RequestValidationError(f"{name} is required")
    return field


def _parse_asset(value: object) -> AssetSpec:
    if not isinstance(value, dict):
        raise RequestValidationError("asset is invalid")
    kind = _required_string(value, "kind")
    volume_path = _required_string(value, "volume_path")
    comfy_name = _required_string(value, "comfy_name")
    if kind not in {"image", "video"}:
        raise RequestValidationError("exactly one image and one video are required")
    if (
        not NAME_PATTERN.fullmatch(comfy_name)
        or Path(comfy_name).name != comfy_name
        or "\\" in comfy_name
    ):
        raise RequestValidationError("asset comfy_name is invalid")
    return AssetSpec(kind=kind, volume_path=volume_path, comfy_name=comfy_name)


def _resolve_job_input_root(volume_root: Path, run_id: str) -> Path:
    lexical_components = (
        volume_root,
        volume_root / "jobs",
        volume_root / "jobs" / run_id,
        volume_root / "jobs" / run_id / "input",
    )
    if any(component.is_symlink() for component in lexical_components):
        raise RequestValidationError("asset source must not be a symlink")
    return lexical_components[-1].resolve()


def _validate_asset_source(asset: AssetSpec, input_root: Path) -> None:
    supplied_path = Path(asset.volume_path)
    if supplied_path.is_absolute():
        raise RequestValidationError("asset path must be relative")
    if ".." in supplied_path.parts:
        raise RequestValidationError("asset path escapes the job input directory")

    source = input_root.parent.parent.parent / supplied_path
    try:
        source.relative_to(input_root)
    except ValueError:
        raise RequestValidationError("asset path escapes the job input directory") from None

    relative_source = source.relative_to(input_root)
    component = input_root
    for part in relative_source.parts:
        component /= part
        if component.is_symlink():
            raise RequestValidationError("asset source must not be a symlink")

    resolved_source = source.resolve()
    try:
        resolved_source.relative_to(input_root)
    except ValueError:
        raise RequestValidationError("asset path escapes the job input directory") from None
    if not source.exists():
        raise RequestValidationError("asset source file is missing")
    if not source.is_file():
        raise RequestValidationError("asset source must be a regular file")

    extensions = IMAGE_EXTENSIONS if asset.kind == "image" else VIDEO_EXTENSIONS
    if source.suffix.lower() not in extensions:
        raise RequestValidationError(f"{asset.kind} asset extension is unsupported")


def _validate_workflow(workflow: dict, image: AssetSpec, video: AssetSpec) -> None:
    _require_workflow_node(workflow, "9", "LoadImage")
    _require_workflow_node(workflow, "43", "VHS_LoadVideo")
    _require_workflow_node(workflow, "42", "VHS_VideoCombine")

    if workflow["9"]["inputs"].get("image") != image.comfy_name:
        raise RequestValidationError("workflow node 9 filename does not match asset")
    if workflow["43"]["inputs"].get("video") != video.comfy_name:
        raise RequestValidationError("workflow node 43 filename does not match asset")


def _require_workflow_node(workflow: dict, node_id: str, class_type: str) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict) or node.get("class_type") != class_type:
        raise RequestValidationError(f"workflow node {node_id} must be {class_type}")
    if not isinstance(node.get("inputs"), dict):
        raise RequestValidationError(f"workflow node {node_id} inputs are invalid")
