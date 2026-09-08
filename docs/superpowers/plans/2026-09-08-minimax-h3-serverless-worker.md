# MiniMax H3 Serverless Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a queue-based RunPod worker that runs the approved MiniMax H3 Beta 4 ComfyUI workflow with private volume-backed inputs and MP4 output.

**Architecture:** A small Python handler validates an explicit job contract, stages one KOL image and one reference video from the attached RunPod volume into ComfyUI, submits the API-format workflow to a local ComfyUI process, safely discovers node 42's MP4, and atomically publishes it back to the private volume. A pinned CUDA 13 container installs the exact ComfyUI and custom-node revisions verified on the existing Pod.

**Tech Stack:** Python 3.12, pytest, requests, RunPod Python SDK, ComfyUI, CUDA 13.0, Docker, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-08-minimax-h3-serverless-worker-design.md`

## Global Constraints

- CUDA runtime is 13.0 and PyTorch is 2.10.0 with CUDA 13.0 wheels.
- ComfyUI is pinned to `8a33128f2f8c5585c57486c07de481241e70a39c`.
- ComfyUI-KJNodes is pinned to `c2a47f161bdcecc1e6baf3412f1d116febc26ce3`.
- ComfyUI-VideoHelperSuite is pinned to `115de7a9d9e34410cffb9ecfd268e993b11a50fb`.
- Comfyui_Minimax_h3_latent_Upscaler is pinned to `d7c01b9011f2e8439493f6c02c29995a27df276f`.
- The attached RunPod volume is mounted at `/runpod-volume`; no model or media data is committed to Git.
- Every accepted job has exactly one image, one video, and output node `42`.
- The response contains a private relative volume path and never contains base64 video data or credentials.
- All production Python behavior is implemented test-first.

---

### Task 1: Define and validate the worker request contract

**Files:**
- Create: `pyproject.toml`
- Create: `worker/__init__.py`
- Create: `worker/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Produces: `AssetSpec(kind: str, volume_path: str, comfy_name: str)`
- Produces: `JobRequest(run_id: str, workflow: dict, assets: tuple[AssetSpec, AssetSpec], output_node_id: str)`
- Produces: `parse_job_input(value: object, volume_root: Path) -> JobRequest`
- Produces: `RequestValidationError(ValueError)`

- [ ] **Step 1: Add the Python package and pytest configuration**

Create `pyproject.toml`:

```toml
[project]
name = "minimax-h3-serverless-worker"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "requests==2.32.5",
  "runpod==1.8.1",
]

[project.optional-dependencies]
test = ["pytest==8.4.2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Create an empty `worker/__init__.py`.

- [ ] **Step 2: Write failing contract tests**

Create `tests/test_contracts.py` with helpers that build a temporary volume tree and a minimal workflow containing nodes 9, 42, and 43. Cover a valid request and independent failures for missing input, invalid run IDs, wrong asset counts, duplicate kinds, absolute paths, `..` traversal, a path outside `jobs/<run-id>/input`, symlinks, missing files, unsupported extensions, duplicate `comfy_name` values, a non-basename `comfy_name`, mismatched node filenames, and a node 42 class other than `VHS_VideoCombine`.

The valid assertion is:

```python
request = parse_job_input(payload, volume_root)

assert request.run_id == "018f-example-id"
assert request.output_node_id == "42"
assert [asset.kind for asset in request.assets] == ["image", "video"]
assert request.workflow["9"]["inputs"]["image"] == "018f-example-id-kol.png"
assert request.workflow["43"]["inputs"]["video"] == "018f-example-id-source.mp4"
```

Each invalid case uses:

```python
with pytest.raises(RequestValidationError, match="specific stable message"):
    parse_job_input(payload, volume_root)
```

- [ ] **Step 3: Run the contract tests and confirm RED**

Run:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/test_contracts.py
```

Expected: collection fails because `worker.contracts` does not exist.

- [ ] **Step 4: Implement minimal contract validation**

Create `worker/contracts.py` with frozen dataclasses and these fixed rules:

```python
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
```

`parse_job_input` must:

1. Require a dictionary with `run_id`, `workflow`, `assets`, and `output_node_id`.
2. Require output node ID `"42"`.
3. Require exactly one `image` and one `video`.
4. Resolve every source beneath `volume_root / "jobs" / run_id / "input"`.
5. Reject absolute paths, traversal, symlinks, non-regular files, unsupported extensions, and target-name collisions.
6. Require node 9 to be `LoadImage`, node 43 to be `VHS_LoadVideo`, and node 42 to be `VHS_VideoCombine`.
7. Require nodes 9 and 43 to reference the matching `comfy_name`.

Use stable messages such as `"run_id is invalid"`, `"exactly one image and one video are required"`, `"asset path escapes the job input directory"`, and `"workflow node 42 must be VHS_VideoCombine"`.

- [ ] **Step 5: Run contract tests and confirm GREEN**

Run:

```bash
.venv/bin/pytest tests/test_contracts.py
```

Expected: all contract tests pass.

- [ ] **Step 6: Commit the contract layer**

```bash
git add pyproject.toml worker/__init__.py worker/contracts.py tests/test_contracts.py
git commit -m "feat: validate MiniMax H3 worker jobs"
```

---

### Task 2: Stage private inputs and publish MP4 output safely

**Files:**
- Create: `worker/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Consumes: `JobRequest` from Task 1.
- Produces: `staged_assets(request: JobRequest, volume_root: Path, comfy_input_root: Path) -> ContextManager[None]`
- Produces: `find_output_video(history: dict, output_node_id: str, comfy_output_root: Path) -> Path`
- Produces: `publish_video(source: Path, volume_root: Path, run_id: str) -> PublishedVideo`
- Produces: `PublishedVideo(volume_path: str, filename: str, size_bytes: int)`
- Produces: `StorageError(RuntimeError)`

- [ ] **Step 1: Write failing storage tests**

Create tests using `tmp_path` and real files. Cover:

- Both assets copy into the ComfyUI input root and are removed when the context exits.
- A pre-existing destination causes failure and is not overwritten.
- Cleanup occurs when the body raises.
- Node 42 output is found when VideoHelperSuite reports the MP4 under a nested `gifs` list.
- Output traversal, absolute output paths, symlinks, missing files, non-MP4 files, and multiple MP4 candidates fail.
- Publication writes `jobs/<run-id>/output/result.mp4`, preserves bytes, returns the exact metadata, and leaves no temporary file.
- Re-publication atomically replaces an earlier result for safe RunPod retries.

Use a representative history value:

```python
history = {
    "outputs": {
        "42": {
            "gifs": [
                {
                    "filename": "minimaxh3_00001.mp4",
                    "subfolder": "",
                    "type": "output",
                }
            ]
        }
    }
}
```

- [ ] **Step 2: Run storage tests and confirm RED**

Run:

```bash
.venv/bin/pytest tests/test_storage.py
```

Expected: collection fails because `worker.storage` does not exist.

- [ ] **Step 3: Implement staging, discovery, and publication**

Create `worker/storage.py`.

`staged_assets` copies with `shutil.copy2`, records only destinations it created, and unlinks those exact regular files in `finally`.

`find_output_video` recursively inspects dictionaries and lists beneath `history["outputs"][output_node_id]`. A candidate qualifies only when its dictionary has `type == "output"`, a string filename ending in `.mp4`, and an optional safe relative subfolder. Resolve the candidate and require it to remain beneath `comfy_output_root`, be a regular file, and not be a symlink. Require exactly one candidate.

`publish_video` creates `volume_root/jobs/<run-id>/output`, copies to a unique temporary sibling with `shutil.copyfile`, flushes and `os.fsync`s it, and calls `os.replace` to publish `result.mp4`. Return:

```python
PublishedVideo(
    volume_path=f"jobs/{run_id}/output/result.mp4",
    filename="result.mp4",
    size_bytes=destination.stat().st_size,
)
```

- [ ] **Step 4: Run storage tests and confirm GREEN**

Run:

```bash
.venv/bin/pytest tests/test_storage.py
```

Expected: all storage tests pass.

- [ ] **Step 5: Commit safe storage handling**

```bash
git add worker/storage.py tests/test_storage.py
git commit -m "feat: stage volume inputs and publish videos"
```

---

### Task 3: Execute workflows through the local ComfyUI API

**Files:**
- Create: `worker/comfy.py`
- Create: `tests/test_comfy.py`

**Interfaces:**
- Produces: `ComfyClient(base_url: str, poll_interval: float, request_timeout: float)`
- Produces: `ComfyClient.execute(workflow: dict, client_id: str) -> dict`
- Produces: `ComfyExecutionError(RuntimeError)`

- [ ] **Step 1: Write failing ComfyUI client tests**

Use a tiny local HTTP server fixture rather than mocking `requests`. The fixture must provide deterministic responses for:

- `POST /prompt` returning `{"prompt_id": "prompt-1"}`.
- `GET /history/prompt-1` returning an empty object once, then completed history.
- A 400 prompt response containing `node_errors`.
- Completed history with `execution_error`.
- A request timeout on repeated history requests.

Assert that `execute` sends the exact workflow and client ID, polls until completion, returns the single prompt history object, and raises stable errors containing ComfyUI's validation or execution message.

- [ ] **Step 2: Run client tests and confirm RED**

Run:

```bash
.venv/bin/pytest tests/test_comfy.py
```

Expected: collection fails because `worker.comfy` does not exist.

- [ ] **Step 3: Implement the minimal ComfyUI client**

Create `worker/comfy.py` using a private `requests.Session`.

`execute` must:

1. POST `{"prompt": workflow, "client_id": client_id}` to `/prompt`.
2. Parse and require a non-empty `prompt_id`.
3. Poll `/history/<urlencoded-prompt-id>` at `poll_interval`.
4. Return only when the prompt appears and reports completion.
5. Inspect `status.messages` for `execution_error` or `execution_interrupted`.
6. Raise `ComfyExecutionError` for HTTP, JSON, validation, execution, or timeout failures without logging the workflow.

The runtime defaults are:

```python
base_url = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188")
poll_interval = float(os.environ.get("COMFY_POLL_INTERVAL_SECONDS", "2"))
request_timeout = float(os.environ.get("COMFY_REQUEST_TIMEOUT_SECONDS", "30"))
```

The RunPod execution timeout remains the outer job-duration limit; the client does not impose a second render deadline.

- [ ] **Step 4: Run client tests and confirm GREEN**

Run:

```bash
.venv/bin/pytest tests/test_comfy.py
```

Expected: all ComfyUI client tests pass.

- [ ] **Step 5: Commit the ComfyUI client**

```bash
git add worker/comfy.py tests/test_comfy.py
git commit -m "feat: execute workflows through ComfyUI"
```

---

### Task 4: Integrate the RunPod handler

**Files:**
- Create: `handler.py`
- Create: `tests/test_handler.py`

**Interfaces:**
- Consumes: `parse_job_input`, `staged_assets`, `ComfyClient.execute`, `find_output_video`, and `publish_video`.
- Produces: `handler(job: dict) -> dict`.
- Produces success output matching the approved specification.

- [ ] **Step 1: Write failing handler tests**

Create tests with dependency injection through a `build_handler(client, volume_root, comfy_input_root, comfy_output_root)` factory. Use a fake client that writes a real MP4 fixture to the temporary ComfyUI output directory and returns VideoHelperSuite-shaped history.

Cover:

- A successful job returns exactly:

```python
{
    "status": "success",
    "run_id": "018f-example-id",
    "video": {
        "volume_path": "jobs/018f-example-id/output/result.mp4",
        "filename": "result.mp4",
        "size_bytes": len(video_bytes),
    },
}
```

- The factory passes the untouched workflow and run ID as ComfyUI client ID.
- Staged inputs exist during execution and are removed afterward.
- Contract, ComfyUI, and storage errors escape the handler so RunPod marks the job failed.
- No secret or workflow data appears in captured logs.

- [ ] **Step 2: Run handler tests and confirm RED**

Run:

```bash
.venv/bin/pytest tests/test_handler.py
```

Expected: collection fails because `handler` does not exist.

- [ ] **Step 3: Implement the handler composition**

Create `handler.py` with environment-derived roots:

```python
VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume"))
COMFY_INPUT_ROOT = Path(os.environ.get("COMFY_INPUT_ROOT", "/comfyui/input"))
COMFY_OUTPUT_ROOT = Path(os.environ.get("COMFY_OUTPUT_ROOT", "/comfyui/output"))
```

`build_handler` must validate `job["input"]`, stage inputs, execute ComfyUI, discover the MP4, publish it, and return the dataclass fields as JSON-compatible dictionaries. Log only the run ID and lifecycle state.

At module startup construct the production handler. Under `if __name__ == "__main__":`, call:

```python
runpod.serverless.start({"handler": handler})
```

- [ ] **Step 4: Run handler tests and the full unit suite**

Run:

```bash
.venv/bin/pytest
```

Expected: all tests pass.

- [ ] **Step 5: Commit handler integration**

```bash
git add handler.py tests/test_handler.py
git commit -m "feat: add RunPod MiniMax H3 handler"
```

---

### Task 5: Build the pinned CUDA 13 ComfyUI container

**Files:**
- Create: `Dockerfile`
- Create: `docker/start.sh`
- Create: `docker/extra_model_paths.yaml`
- Create: `docker/verify_nodes.py`
- Create: `.dockerignore`
- Create: `tests/test_container_files.py`

**Interfaces:**
- The container starts local ComfyUI on `127.0.0.1:8188`, then runs `handler.py`.
- ComfyUI reads all five model categories from `/runpod-volume/models`.
- `verify_nodes.py <object-info-url>` exits zero only when every class used by the approved workflow is present.

- [ ] **Step 1: Write failing static container tests**

Create `tests/test_container_files.py` to assert:

- Dockerfile pins the CUDA image, PyTorch packages, ComfyUI commit, and all three custom-node commits from Global Constraints.
- Dockerfile never contains the five model filenames, `ARG` names containing `TOKEN`, `KEY`, or `SECRET`, or `COPY` commands for model/media directories.
- `extra_model_paths.yaml` maps `diffusion_models`, `text_encoders`, `vae`, and `latent_upscale_models` beneath `/runpod-volume/models`.
- `start.sh` traps worker termination, starts ComfyUI only on loopback, waits for `/system_stats`, runs node verification, and then executes `handler.py`.
- `verify_nodes.py` declares every unique class from the approved workflow.

- [ ] **Step 2: Run static container tests and confirm RED**

Run:

```bash
.venv/bin/pytest tests/test_container_files.py
```

Expected: failures because the container files do not exist.

- [ ] **Step 3: Create the pinned Dockerfile**

Use this build structure:

```dockerfile
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 python3.12 python3.12-venv wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && pip install --upgrade pip \
    && pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
       --index-url https://download.pytorch.org/whl/cu130

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && git -C /comfyui checkout 8a33128f2f8c5585c57486c07de481241e70a39c \
    && pip install -r /comfyui/requirements.txt \
    && pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0"

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes \
    && git -C /comfyui/custom_nodes/ComfyUI-KJNodes checkout c2a47f161bdcecc1e6baf3412f1d116febc26ce3 \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
    && git -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite checkout 115de7a9d9e34410cffb9ecfd268e993b11a50fb \
    && git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git /comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler \
    && git -C /comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler checkout d7c01b9011f2e8439493f6c02c29995a27df276f \
    && for requirements in /comfyui/custom_nodes/*/requirements.txt; do \
         test ! -f "$requirements" || pip install -r "$requirements"; \
       done

WORKDIR /app
COPY pyproject.toml handler.py ./
COPY worker ./worker
COPY docker/start.sh /start.sh
COPY docker/extra_model_paths.yaml /app/extra_model_paths.yaml
COPY docker/verify_nodes.py /app/verify_nodes.py
RUN pip install . && chmod +x /start.sh

CMD ["/start.sh"]
```

During implementation, run a no-cache dependency resolution once and record the fully resolved direct runtime versions in `README.md`; do not add model weights to the image.

- [ ] **Step 4: Add model-path and startup files**

Create `docker/extra_model_paths.yaml`:

```yaml
runpod_volume:
  base_path: /runpod-volume
  is_default: true
  diffusion_models: models/diffusion_models
  text_encoders: models/text_encoders
  vae: models/vae
  latent_upscale_models: models/latent_upscale_models
```

Create `docker/start.sh` that:

1. Starts `python /comfyui/main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --extra-model-paths-config /app/extra_model_paths.yaml` in the background.
2. Saves its PID and installs an EXIT/TERM/INT trap that terminates only that PID.
3. Polls `http://127.0.0.1:8188/system_stats` for at most 300 seconds while failing immediately if the process exits.
4. Runs `python /app/verify_nodes.py http://127.0.0.1:8188/object_info`.
5. Runs `exec python /app/handler.py`.

Create `docker/verify_nodes.py` with the exact unique class set extracted from `assets/comfyui/minimax_h3_r2v_video_edit.json` in the KOL Dance repository and a requests-based check that prints only missing class names.

- [ ] **Step 5: Add a strict Docker build context**

Create `.dockerignore`:

```text
.git
.github
.venv
__pycache__
.pytest_cache
docs
tests
*.safetensors
*.ckpt
*.pt
*.pth
*.png
*.jpg
*.jpeg
*.webp
*.mp4
*.mov
*.webm
.env
.env.*
```

- [ ] **Step 6: Run static tests and local image build**

Run:

```bash
.venv/bin/pytest tests/test_container_files.py
docker build --platform linux/amd64 -t minimax-h3-serverless-worker:test .
```

Expected: tests pass and Docker build exits zero.

- [ ] **Step 7: Run the CPU container smoke test**

Run the image with a temporary empty directory mounted at `/runpod-volume`, override the command to start ComfyUI in CPU mode, and query `object_info` with `verify_nodes.py`. The smoke test must demonstrate that all workflow classes import without GPU access; it does not execute the video model.

- [ ] **Step 8: Commit the container**

```bash
git add Dockerfile docker .dockerignore tests/test_container_files.py
git commit -m "build: add pinned CUDA 13 ComfyUI image"
```

---

### Task 6: Add repository documentation and continuous verification

**Files:**
- Create: `README.md`
- Create: `.github/workflows/test.yml`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `tests/fixtures/job-input.json`

**Interfaces:**
- Documents the exact RunPod endpoint source and runtime configuration.
- Provides a non-secret example request matching the handler contract.
- CI runs the unit suite on every push and pull request.

- [ ] **Step 1: Add a representative request fixture**

Create `tests/fixtures/job-input.json` using run ID `018f-example-id`, both documented volume paths, nodes 9, 42, and 43 with their required classes and filenames, and no private data.

- [ ] **Step 2: Write README verification tests**

Extend `tests/test_container_files.py` to require the README to document:

- Network-volume ID `j4ds1uajmj` and region `US-KS-2`.
- Model directory tree.
- Endpoint values: queue-based, RTX PRO 6000, one GPU per worker, Max workers 1, Active workers 0, Queue delay 1 second, FlashBoot enabled, execution timeout 3600 seconds, and CUDA 13.0 or newer.
- GitHub source configuration.
- `POST /run`, `GET /status/{job-id}`, and the private success response.
- A warning that RunPod and S3 credentials belong only in server-side secret stores.
- A troubleshooting command that enables `NETWORK_VOLUME_DEBUG=true`.
- A note that the old Pod must remain until live verification passes.

Run:

```bash
.venv/bin/pytest tests/test_container_files.py
```

Expected: README assertions fail because the file does not exist.

- [ ] **Step 3: Create README, licensing, and notice files**

Write `README.md` with setup, architecture, exact endpoint fields, request/response examples, deployment, logs, and troubleshooting sections. Link to the approved design and plan.

Copy the AGPL-3.0 license text required by the upstream `worker-comfyui` derivative into `LICENSE`. Add `NOTICE` identifying RunPod's `worker-comfyui` 5.8.6 as the behavior baseline, linking its source, and describing the volume-backed input and MP4-output modifications.

- [ ] **Step 4: Add GitHub Actions**

Create `.github/workflows/test.yml` that checks out the repository, installs Python 3.12, runs `pip install -e ".[test]"`, and runs `pytest`. Do not place any secrets in CI and do not run paid RunPod jobs.

- [ ] **Step 5: Run the full local verification suite**

Run:

```bash
.venv/bin/pytest
docker build --platform linux/amd64 -t minimax-h3-serverless-worker:test .
```

Expected: all tests pass and the image builds.

- [ ] **Step 6: Commit documentation and CI**

```bash
git add README.md LICENSE NOTICE .github tests/fixtures
git commit -m "docs: add deployment and verification guide"
```

---

### Task 7: Deploy and verify on RunPod

**Files:**
- No repository changes unless deployment exposes a reproducible defect; defects require a failing regression test before a fix.

**Interfaces:**
- Consumes the GitHub `main` branch and network volume `j4ds1uajmj`.
- Produces a verified RunPod release and private MP4 object.

- [ ] **Step 1: Confirm model population before deployment**

From the existing Pod, run:

```bash
/workspace/tools/runpod-s3-cli/bin/aws s3 ls \
  s3://j4ds1uajmj/models \
  --recursive --human-readable --summarize \
  --profile runpod \
  --region us-ks-2 \
  --endpoint-url https://s3api-us-ks-2.runpod.io
```

Require all five exact model filenames and nonzero sizes. Do not terminate the old Pod.

- [ ] **Step 2: Configure the RunPod GitHub source**

Edit endpoint `minimaxh3_beta4` to build `locust08/minimax-h3-serverless-worker`, branch `main`, context path `/`, and Dockerfile path `Dockerfile`. Attach `j4ds1uajmj`.

Retain:

```text
Type: Queue based
GPU: RTX PRO 6000
GPU count: 1
Max workers: 1
Active workers: 0
Idle timeout: 5 seconds
FlashBoot: enabled
Execution timeout: 3600 seconds
Auto scaling: Queue delay
Queue delay: 1 second
Allowed CUDA: 13.0 and newer
```

- [ ] **Step 3: Verify worker startup**

Wait for the release build. Submit no generation until logs show:

- Local ComfyUI is reachable.
- Every required class passes node verification.
- All five model names appear in the relevant `object_info` loader lists.
- The RunPod handler is ready.

- [ ] **Step 4: Upload a short private smoke-test input**

Use a new run ID, upload one existing KOL image and a 2–3 second video excerpt to:

```text
s3://j4ds1uajmj/jobs/<run-id>/input/<run-id>-kol.png
s3://j4ds1uajmj/jobs/<run-id>/input/<run-id>-source.mp4
```

Submit the approved API workflow with duration and frame count adjusted to the excerpt.

- [ ] **Step 5: Submit and monitor the queue job**

Call `POST https://api.runpod.ai/v2/8vrjc9ecbvk8bl/run` with the server-side RunPod API key. Poll `GET /status/<job-id>` until `COMPLETED` or `FAILED`. Never print the API key or complete prompt in logs.

- [ ] **Step 6: Verify the private MP4**

Require the response path to equal `jobs/<run-id>/output/result.mp4`. Download it through the RunPod S3-compatible API and verify:

```bash
ffprobe -v error -show_entries format=duration -show_streams result.mp4
```

Require a decodable H.264 video stream, 24 fps, duration matching the smoke-test source within one frame, and the expected audio stream when the source has audio.

- [ ] **Step 7: Preserve evidence and decide whether the old Pod can be stopped**

Record the RunPod release ID, job ID, worker startup time, execution time, result size, and ffprobe summary in the deployment notes. Keep the old Pod until the KOL Dance website integration downloads and saves one Serverless-generated result successfully.
