from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
START_SCRIPT = ROOT / "docker" / "start.sh"
MODEL_PATHS = ROOT / "docker" / "extra_model_paths.yaml"
VERIFY_NODES = ROOT / "docker" / "verify_nodes.py"
README = ROOT / "README.md"
JOB_INPUT_FIXTURE = ROOT / "tests" / "fixtures" / "job-input.json"

WORKFLOW_CLASSES = {
    "BasicGuider",
    "BasicScheduler",
    "CLIPLoader",
    "ComfyMathExpression",
    "ImageResizeKJv2",
    "KSamplerSelect",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LoadAudio",
    "LoadImage",
    "ManualSigmas",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
    "MinimaxH3LatentUpscaler3D",
    "ModelAttentionBackend",
    "ModelPreviewOverrideKJ",
    "PrimitiveFloat",
    "PrimitiveStringMultiline",
    "RandomNoise",
    "ResolutionSelector",
    "SamplerCustomAdvanced",
    "SplitSigmas",
    "UNETLoader",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAELoader",
    "VHS_LoadVideo",
    "VHS_VideoCombine",
}

MODEL_FILENAMES = {
    "10Eros_Max_h3_TURBO-hybrid_beta4.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_latent_upscaler_3d_bf16.safetensors",
}


def test_dockerfile_pins_the_approved_runtime_and_revisions() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04" in dockerfile
    assert "torch==2.10.0" in dockerfile
    assert "torchvision==0.25.0" in dockerfile
    assert "torchaudio==2.10.0" in dockerfile
    assert "https://download.pytorch.org/whl/cu130" in dockerfile
    assert "8a33128f2f8c5585c57486c07de481241e70a39c" in dockerfile
    assert "57105374f47d0fbb49c9c3926fb981702e0a4b5c" in dockerfile
    assert "115de7a9d9e34410cffb9ecfd268e993b11a50fb" in dockerfile
    assert "d7c01b9011f2e8439493f6c02c29995a27df276f" in dockerfile


def test_dockerfile_excludes_models_media_and_secret_build_arguments() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert all(filename not in dockerfile for filename in MODEL_FILENAMES)
    assert not re.search(
        r"(?im)^\s*ARG\s+[^\n]*(?:TOKEN|KEY|SECRET)", dockerfile
    )
    copy_lines = re.findall(r"(?im)^\s*COPY\s+.*$", dockerfile)
    forbidden_source = re.compile(
        r"(?i)(?:^|[\s\"'=./])(?:models?|media|assets|images?|videos?)(?:$|[/\s\"'])"
    )
    assert all(not forbidden_source.search(line) for line in copy_lines)


def test_custom_node_dependency_installation_stops_on_first_failure(
    tmp_path: Path,
) -> None:
    dockerfile = DOCKERFILE.read_text()
    loop_start = dockerfile.index("for requirements in")
    loop_end = dockerfile.index("done", loop_start) + len("done")
    install_loop = dockerfile[loop_start:loop_end].replace("\\\n", "\n")

    custom_nodes = tmp_path / "custom_nodes"
    first = custom_nodes / "a-first" / "requirements.txt"
    second = custom_nodes / "b-second" / "requirements.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first-dependency\n")
    second.write_text("second-dependency\n")
    install_loop = install_loop.replace("/comfyui/custom_nodes", str(custom_nodes))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip_log = tmp_path / "pip.log"
    fake_pip = bin_dir / "pip"
    fake_pip.write_text(
        """#!/bin/sh
printf '%s\\n' "$3" >> "$PIP_LOG"
case "$3" in
  */a-first/requirements.txt) exit 23 ;;
  *) exit 0 ;;
esac
"""
    )
    fake_pip.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PIP_LOG"] = str(pip_log)

    result = subprocess.run(["sh", "-c", install_loop], env=env)

    assert result.returncode != 0
    assert pip_log.read_text().splitlines() == [str(first)]


def test_model_paths_map_every_required_category_to_the_network_volume() -> None:
    values = {}
    for line in MODEL_PATHS.read_text().splitlines():
        if ":" in line:
            key, value = line.strip().split(":", 1)
            values[key] = value.strip()

    assert values["base_path"] == "/runpod-volume"
    assert values["is_default"] == "true"
    assert values["diffusion_models"] == "models/diffusion_models"
    assert values["text_encoders"] == "models/text_encoders"
    assert values["vae"] == "models/vae"
    assert values["latent_upscale_models"] == "models/latent_upscale_models"


def test_start_script_has_the_required_local_startup_sequence() -> None:
    script = START_SCRIPT.read_text()

    subprocess.run(["sh", "-n", START_SCRIPT], check=True)
    comfy_command = (
        "python /comfyui/main.py --listen 127.0.0.1 --port 8188 "
        "--disable-auto-launch --extra-model-paths-config /app/extra_model_paths.yaml"
    )
    assert comfy_command in script
    assert "http://127.0.0.1:8188/system_stats" in script
    assert "--timeout=1" in script
    assert "--tries=1" in script
    assert "kill -0 \"$COMFY_PID\"" in script
    assert "300" in script
    verify_position = script.index(
        "python /app/verify_nodes.py http://127.0.0.1:8188/object_info"
    )
    handler_position = script.index("python /app/handler.py")
    assert verify_position < handler_position
    assert re.search(r"python /app/handler\.py\s*&\s*\nHANDLER_PID=\$!", script)
    assert "wait \"$HANDLER_PID\"" in script
    assert "exec python /app/handler.py" not in script


def _write_fake_commands(bin_dir: Path) -> None:
    python = bin_dir / "python"
    python.write_text(
        """#!/bin/sh
case "$1" in
  /comfyui/main.py)
    printf '%s\\n' comfy-start >> "$EVENT_LOG"
    if [ "${COMFY_EXIT:-0}" = 1 ]; then exit 7; fi
    trap 'printf "%s\\n" comfy-term >> "$EVENT_LOG"; exit 0' TERM INT
    while :; do sleep 1; done
    ;;
  /app/verify_nodes.py)
    printf '%s\\n' verify >> "$EVENT_LOG"
    ;;
  /app/handler.py)
    printf '%s\\n' handler-start >> "$EVENT_LOG"
    trap 'printf "%s\\n" handler-term >> "$EVENT_LOG"; exit 0' TERM INT
    while :; do sleep 1; done
    ;;
  *) exit 64 ;;
esac
"""
    )
    python.chmod(0o755)
    wget = bin_dir / "wget"
    wget.write_text(
        """#!/bin/sh
[ "${COMFY_EXIT:-0}" = 0 ] && grep -q '^comfy-start$' "$EVENT_LOG"
"""
    )
    wget.chmod(0o755)


def _wait_for_event(event_log: Path, expected: str) -> None:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if event_log.exists() and expected in event_log.read_text().splitlines():
            return
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for {expected}")


def test_start_script_forwards_termination_to_both_children(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_commands(bin_dir)
    event_log = tmp_path / "events"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["EVENT_LOG"] = str(event_log)

    process = subprocess.Popen(
        ["sh", START_SCRIPT],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_event(event_log, "handler-start")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    events = event_log.read_text().splitlines()
    assert events[:3] == ["comfy-start", "verify", "handler-start"]
    assert "handler-term" in events
    assert "comfy-term" in events


def test_start_script_stops_if_comfyui_exits_before_readiness(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_commands(bin_dir)
    event_log = tmp_path / "events"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["EVENT_LOG"] = str(event_log)
    env["COMFY_EXIT"] = "1"

    result = subprocess.run(
        ["sh", START_SCRIPT],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )

    assert result.returncode != 0
    assert event_log.read_text().splitlines() == ["comfy-start"]


def _load_verify_nodes():
    spec = importlib.util.spec_from_file_location("verify_nodes", VERIFY_NODES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_nodes_declares_the_exact_approved_workflow_classes() -> None:
    tree = ast.parse(VERIFY_NODES.read_text())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value.args[0])
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "REQUIRED_NODE_CLASSES"
        and isinstance(node.value, ast.Call)
    }

    assert set(assignments["REQUIRED_NODE_CLASSES"]) == WORKFLOW_CLASSES


def test_verify_nodes_prints_only_sorted_missing_class_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_verify_nodes()

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                node_class: {} for node_class in WORKFLOW_CLASSES - {"VAELoader", "CLIPLoader"}
            }

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())

    assert module.main(["http://127.0.0.1:8188/object_info"]) == 1
    assert capsys.readouterr().out == "CLIPLoader\nVAELoader\n"


def test_verify_nodes_succeeds_silently_when_all_classes_are_registered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_verify_nodes()

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {node_class: {} for node_class in WORKFLOW_CLASSES}

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())

    assert module.main(["http://127.0.0.1:8188/object_info"]) == 0
    assert capsys.readouterr().out == ""


def test_dockerignore_excludes_repository_metadata_models_and_media() -> None:
    ignored = set((ROOT / ".dockerignore").read_text().splitlines())

    assert {
        ".git",
        ".github",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "docs",
        "tests",
        "*.safetensors",
        "*.ckpt",
        "*.pt",
        "*.pth",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp",
        "*.mp4",
        "*.mov",
        "*.webm",
        ".env",
        ".env.*",
    } <= ignored


def test_readme_documents_the_private_runpod_deployment_contract() -> None:
    readme = README.read_text()

    assert "j4ds1uajmj" in readme
    assert "US-KS-2" in readme
    assert "/runpod-volume/models" in readme
    assert "10Eros_Max_h3_TURBO-hybrid_beta4.safetensors" in readme
    assert "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in readme
    assert "minimax_h3_video_vae_fp16.safetensors" in readme
    assert "minimax_h3_audio_vae_fp32.safetensors" in readme
    assert "minimax_h3_latent_upscaler_3d_bf16.safetensors" in readme
    assert "queue-based" in readme.lower()
    assert "RTX PRO 6000" in readme
    assert "one GPU per worker" in readme
    assert "Max workers: 1" in readme
    assert "Active workers: 0" in readme
    assert "Queue delay: 1 second" in readme
    assert "FlashBoot: enabled" in readme
    assert "Execution timeout: 3600 seconds" in readme
    assert "CUDA 13.0 or newer" in readme
    assert "DigitalBee-AI/minimax-h3-serverless-worker" in readme
    assert "POST /run" in readme
    assert "GET /status/{job-id}" in readme
    assert "Idle timeout: 5 seconds" in readme
    assert '"volume_path": "jobs/018f-example-id/output/result.mp4"' in readme
    assert "RunPod and S3 credentials belong only in server-side secret stores" in readme
    assert "NETWORK_VOLUME_DEBUG=true" in readme
    assert "old Pod must remain until live verification passes" in readme
    assert "s3://j4ds1uajmj/jobs/018f-example-id/output/result.mp4" in readme
    assert "--region us-ks-2" in readme
    assert "https://s3api-us-ks-2.runpod.io" in readme
    assert "--profile runpod" in readme
    assert "requests==2.32.5" in readme
    assert "runpod==1.8.1" in readme
    assert "torch==2.10.0" in readme
    assert "torchvision==0.25.0" in readme
    assert "torchaudio==2.10.0" in readme
    assert "8a33128f2f8c5585c57486c07de481241e70a39c" in readme
    assert "c2a47f161bdcecc1e6baf3412f1d116febc26ce3" in readme
    assert "115de7a9d9e34410cffb9ecfd268e993b11a50fb" in readme
    assert "d7c01b9011f2e8439493f6c02c29995a27df276f" in readme


def test_job_input_fixture_is_a_non_secret_handler_request() -> None:
    payload = json.loads(JOB_INPUT_FIXTURE.read_text())
    job_input = payload["input"]

    assert job_input["run_id"] == "018f-example-id"
    assert job_input["output_node_id"] == "42"
    assert job_input["workflow"]["9"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "018f-example-id-kol.png"},
    }
    assert job_input["workflow"]["42"]["class_type"] == "VHS_VideoCombine"
    assert job_input["workflow"]["43"] == {
        "class_type": "VHS_LoadVideo",
        "inputs": {"video": "018f-example-id-source.mp4"},
    }
    assert job_input["assets"] == [
        {
            "kind": "image",
            "volume_path": "jobs/018f-example-id/input/018f-example-id-kol.png",
            "comfy_name": "018f-example-id-kol.png",
        },
        {
            "kind": "video",
            "volume_path": "jobs/018f-example-id/input/018f-example-id-source.mp4",
            "comfy_name": "018f-example-id-source.mp4",
        },
    ]


def test_readme_requires_full_workflow_before_submitting_contract_fixture() -> None:
    readme = README.read_text()
    contract_section = readme.split("## Private job contract", 1)[1]
    before_submit = " ".join(contract_section.split("curl --request POST", 1)[0].split())

    assert "contract-validation-only" in before_submit
    assert "not runnable" in before_submit
    assert "Replace `input.workflow` with the full approved API-format workflow" in before_submit
    assert "before `POST /run`" in before_submit
    assert "--data @tests/fixtures/job-input.json" not in contract_section
    assert "--data @job-request.json" in contract_section


def test_ci_runs_pytest_as_a_module_after_editable_install() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    commands = re.findall(r"(?m)^\s*- run: (.+)$", workflow)

    assert 'pip install -e ".[test]"' in commands
    assert "python -m pytest" in commands
    assert commands.index('pip install -e ".[test]"') < commands.index("python -m pytest")
    assert "pytest" not in commands


def test_project_installs_only_the_worker_package_in_an_isolated_venv(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "package-test-venv"
    subprocess.run([sys.executable, "-m", "venv", venv], check=True)
    subprocess.run(
        [venv / "bin" / "pip", "install", "--upgrade", "pip", "setuptools"],
        check=True,
        capture_output=True,
        text=True,
    )

    install_command = [
        venv / "bin" / "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "-e",
        str(ROOT),
    ]
    if sys.version_info < (3, 12):
        install_command.append("--ignore-requires-python")
    subprocess.run(install_command, check=True, capture_output=True, text=True)

    result = subprocess.run(
        [
            venv / "bin" / "python",
            "-c",
            "import importlib.util; import worker; "
            "assert importlib.util.find_spec('docker') is None",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
