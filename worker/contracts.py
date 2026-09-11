from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".wav"}
SUPPORTED_JOB_TYPES = {
    "dance",
    "foodie",
    "dbee-ai-ugc",
    "dbee-product-showcase",
}


class RequestValidationError(ValueError):
    """Raised when a RunPod job does not satisfy the worker contract."""


@dataclass(frozen=True)
class AssetSpec:
    kind: str
    volume_path: str
    comfy_name: str
    role: str | None = None


@dataclass(frozen=True)
class JobRequest:
    run_id: str
    workflow: dict
    assets: tuple[AssetSpec, ...]
    output_node_id: str
    job_type: str = "dance"


def parse_job_input(value: object, volume_root: Path) -> JobRequest:
    if not isinstance(value, dict):
        raise RequestValidationError("input must be a dictionary")

    job_type = value.get("job_type", "dance")
    if not isinstance(job_type, str) or job_type not in SUPPORTED_JOB_TYPES:
        raise RequestValidationError("job_type is invalid")

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
    if not isinstance(raw_assets, (list, tuple)):
        raise RequestValidationError("assets are required")

    assets = tuple(_parse_asset(asset) for asset in raw_assets)
    asset_validators = {
        "dance": _validate_dance_assets,
        "foodie": _validate_foodie_assets,
        "dbee-ai-ugc": _validate_dbee_ai_ugc_assets,
        "dbee-product-showcase": _validate_dbee_product_showcase_assets,
    }
    asset_validators[job_type](assets)
    if len({asset.comfy_name for asset in assets}) != len(assets):
        raise RequestValidationError("asset comfy_name values must be unique")

    input_root = _resolve_job_input_root(volume_root, run_id)
    for asset in assets:
        _validate_asset_source(asset, input_root)

    workflow_validators = {
        "dance": _validate_dance_workflow,
        "foodie": _validate_foodie_workflow,
        "dbee-ai-ugc": _validate_dbee_ai_ugc_workflow,
        "dbee-product-showcase": _validate_dbee_product_showcase_workflow,
    }
    workflow_validators[job_type](workflow, assets)

    return JobRequest(
        run_id=run_id,
        workflow=workflow,
        assets=assets,
        output_node_id=output_node_id,
        job_type=job_type,
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
    role = value.get("role")
    if kind not in {"image", "video", "audio"}:
        raise RequestValidationError("asset kind is unsupported")
    if role is not None and not isinstance(role, str):
        raise RequestValidationError("asset role is invalid")
    if (
        not NAME_PATTERN.fullmatch(comfy_name)
        or Path(comfy_name).name != comfy_name
        or "\\" in comfy_name
    ):
        raise RequestValidationError("asset comfy_name is invalid")
    return AssetSpec(
        kind=kind,
        volume_path=volume_path,
        comfy_name=comfy_name,
        role=role,
    )


def _validate_dance_assets(assets: tuple[AssetSpec, ...]) -> None:
    if len(assets) != 2 or {asset.kind for asset in assets} != {"image", "video"}:
        raise RequestValidationError("exactly one image and one video are required")


def _validate_foodie_assets(assets: tuple[AssetSpec, ...]) -> None:
    required = {
        ("image", "kol"),
        ("image", "storyboard"),
        ("audio", "voice"),
    }
    if len(assets) != 3 or {(asset.kind, asset.role) for asset in assets} != required:
        raise RequestValidationError(
            "exactly one Foodie KOL image, storyboard image, and voice audio are required"
        )


def _validate_dbee_ai_ugc_assets(assets: tuple[AssetSpec, ...]) -> None:
    required = {
        ("image", "character"),
        ("image", "product"),
        ("image", "environment"),
        ("image", "storyboard"),
        ("audio", "voice"),
    }
    if len(assets) != 5 or {(asset.kind, asset.role) for asset in assets} != required:
        raise RequestValidationError(
            "exactly five DBee AI UGC assets are required: character, product, "
            "environment, storyboard, and voice"
        )


def _validate_dbee_product_showcase_assets(
    assets: tuple[AssetSpec, ...]
) -> None:
    expected_roles = [f"picture-{index + 1}" for index in range(len(assets))]
    if (
        len(assets) < 1
        or len(assets) > 9
        or any(asset.kind != "image" for asset in assets)
        or [asset.role for asset in assets] != expected_roles
    ):
        raise RequestValidationError(
            "exactly 1-9 ordered DBee product images named picture-1 through "
            "picture-9 are required"
        )


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
    if supplied_path.is_absolute() or asset.volume_path.startswith(("/", "\\")):
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

    extensions = {
        "image": IMAGE_EXTENSIONS,
        "video": VIDEO_EXTENSIONS,
        "audio": AUDIO_EXTENSIONS,
    }[asset.kind]
    if source.suffix.lower() not in extensions:
        raise RequestValidationError(f"{asset.kind} asset extension is unsupported")


def _validate_dance_workflow(
    workflow: dict, assets: tuple[AssetSpec, ...]
) -> None:
    image = next(asset for asset in assets if asset.kind == "image")
    video = next(asset for asset in assets if asset.kind == "video")
    _require_workflow_node(workflow, "9", "LoadImage")
    _require_workflow_node(workflow, "43", "VHS_LoadVideo")
    _require_workflow_node(workflow, "42", "VHS_VideoCombine")

    _validate_workflow_filename(workflow, "9", "image", image)
    _validate_workflow_filename(workflow, "43", "video", video)


def _validate_foodie_workflow(
    workflow: dict, assets: tuple[AssetSpec, ...]
) -> None:
    kol = next(asset for asset in assets if asset.role == "kol")
    storyboard = next(asset for asset in assets if asset.role == "storyboard")
    voice = next(asset for asset in assets if asset.role == "voice")
    _require_workflow_node(workflow, "9", "LoadImage")
    _require_workflow_node(workflow, "48", "LoadImage")
    _require_workflow_node(workflow, "50", "LoadAudio")
    _require_workflow_node(workflow, "42", "VHS_VideoCombine")

    _validate_workflow_filename(workflow, "9", "image", kol)
    _validate_workflow_filename(workflow, "48", "image", storyboard)
    _validate_workflow_filename(workflow, "50", "audio", voice)


def _validate_dbee_ai_ugc_workflow(
    workflow: dict, assets: tuple[AssetSpec, ...]
) -> None:
    mapping = {
        "character": ("9", "LoadImage", "image"),
        "product": ("51", "LoadImage", "image"),
        "environment": ("52", "LoadImage", "image"),
        "storyboard": ("48", "LoadImage", "image"),
        "voice": ("50", "LoadAudio", "audio"),
    }
    _require_workflow_node(workflow, "42", "VHS_VideoCombine")
    for role, (node_id, class_type, input_name) in mapping.items():
        asset = next(asset for asset in assets if asset.role == role)
        _require_workflow_node(workflow, node_id, class_type)
        _validate_workflow_filename(workflow, node_id, input_name, asset)


def _validate_dbee_product_showcase_workflow(
    workflow: dict, assets: tuple[AssetSpec, ...]
) -> None:
    _require_workflow_node(workflow, "11", "MiniMaxH3ReferenceToVideo")
    _require_workflow_node(workflow, "42", "VHS_VideoCombine")
    for index, asset in enumerate(assets):
        node_id = str(100 + index)
        _require_workflow_node(workflow, node_id, "LoadImage")
        _validate_workflow_filename(workflow, node_id, "image", asset)
        reference = workflow["11"]["inputs"].get(
            f"ref_images.ref_image_{index}"
        )
        if reference != [node_id, 0]:
            raise RequestValidationError(
                f"workflow node 11 reference {index} does not match asset"
            )


def _validate_workflow_filename(
    workflow: dict, node_id: str, input_name: str, asset: AssetSpec
) -> None:
    if workflow[node_id]["inputs"].get(input_name) != asset.comfy_name:
        raise RequestValidationError(
            f"workflow node {node_id} filename does not match asset"
        )


def _require_workflow_node(workflow: dict, node_id: str, class_type: str) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict) or node.get("class_type") != class_type:
        raise RequestValidationError(f"workflow node {node_id} must be {class_type}")
    if not isinstance(node.get("inputs"), dict):
        raise RequestValidationError(f"workflow node {node_id} inputs are invalid")
