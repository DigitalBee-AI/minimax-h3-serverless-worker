# Foodie Job Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shared MiniMax H3 worker to accept DBee Foodie jobs containing a KOL image, storyboard image, and voice WAV while preserving the existing Dance contract.

**Architecture:** `parse_job_input` dispatches to small Dance and Foodie validators after common request, filename, path, and source-file validation. The generic staging and handler pipeline consumes a variable-length immutable asset tuple, so both job types share ComfyUI execution and MP4 publication.

**Tech Stack:** Python 3.12, dataclasses, pathlib, pytest, RunPod Serverless, ComfyUI

**Spec:** `docs/superpowers/specs/2026-09-10-foodie-job-contract-design.md`

## Global Constraints

- Preserve the existing Dance request when `job_type` is missing.
- Accept `job_type: "dance"` as an explicit equivalent of the existing request.
- Foodie requires exactly one `image/kol`, one `image/storyboard`, and one `audio/voice` asset.
- Foodie workflow nodes are 9 `LoadImage`, 48 `LoadImage`, 50 `LoadAudio`, and 42 `VHS_VideoCombine`.
- Keep all existing path traversal, symlink, regular-file, extension, and duplicate-name protections.
- Keep the existing MP4 output contract and output node `42`.
- Do not push, deploy, submit a paid RunPod job, or modify Supabase or Cloudflare.

---

### Task 1: Add role-aware Foodie request validation

**Files:**
- Modify: `worker/contracts.py`
- Modify: `tests/test_contracts.py`

**Interfaces:**
- Consumes: DBee payload `{ job_type, run_id, workflow, assets, output_node_id }`.
- Produces: `AssetSpec(kind, volume_path, comfy_name, role)` and `JobRequest(job_type, run_id, workflow, assets, output_node_id)`.

- [x] **Step 1: Write failing Foodie contract tests**

Add a Foodie fixture with asset identities `("image", "kol")`, `("image", "storyboard")`, and `("audio", "voice")`. Its workflow must reference those filenames from nodes 9, 48, and 50 and declare node 42 as `VHS_VideoCombine`. Assert the valid request parses and parameterize failures for missing, duplicate, or incorrect roles/kinds, unsupported audio extensions, filename mismatches, and unknown `job_type`. Add an explicit Dance test and assert a missing `job_type` remains compatible.

- [x] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_contracts.py -q`

Expected: the Foodie valid request fails under the current exactly-one-image-and-one-video validator.

- [x] **Step 3: Implement the minimal contract dispatch**

Update the immutable models to support optional asset roles and variable asset counts:

```python
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
```

Normalize a missing `job_type` to `"dance"`, reject unknown types, allow `image`, `video`, and `audio` parsing, choose extensions by kind, and dispatch to `_validate_dance_contract` or `_validate_foodie_contract`. The Foodie validator must compare the exact `(kind, role)` multiset and bind nodes 9, 48, and 50 to the matching `comfy_name`.

- [x] **Step 4: Run focused contract tests and confirm GREEN**

Run: `python -m pytest tests/test_contracts.py -q`

Expected: all contract tests pass.

- [x] **Step 5: Commit the contract change**

```bash
git add worker/contracts.py tests/test_contracts.py docs/superpowers/specs/2026-09-10-foodie-job-contract-design.md docs/superpowers/plans/2026-09-10-foodie-job-contract.md
git commit -m "feat: accept role-aware Foodie worker jobs"
```

### Task 2: Verify Foodie staging through the shared handler

**Files:**
- Create: `.gitignore`
- Modify: `docker/verify_nodes.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_handler.py`
- Modify: `tests/test_container_files.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `JobRequest.assets: tuple[AssetSpec, ...]` from Task 1.
- Produces: the unchanged success response containing `jobs/<run-id>/output/result.mp4`.

- [x] **Step 1: Write failing shared-lifecycle tests**

Add a storage request with three Foodie assets and assert all three files are present during `staged_assets` and absent after exit. Add a handler Foodie job and fake client that assert `kol.png`, `storyboard.png`, and `voice.wav` are staged while executing and that the standard MP4 response is returned.

- [x] **Step 2: Run focused lifecycle tests**

Run: `python -m pytest tests/test_storage.py tests/test_handler.py -q`

Expected: tests pass without handler changes because staging iterates over `request.assets`; any failure reveals an implementation mismatch to fix before proceeding.

- [x] **Step 3: Verify the Foodie audio node at container startup**

Add `LoadAudio` to the exact required node-class set in `docker/verify_nodes.py` and its container-file test. Run the focused test red before changing the verifier and green afterward.

- [x] **Step 4: Document both request contracts**

Update `README.md` so operators can distinguish the backward-compatible Dance payload from the new `job_type: "foodie"` payload, including the three required roles and workflow node bindings.

- [x] **Step 5: Run the complete local suite**

Run: `python -m pytest -q`

Expected: all tests pass with no network or RunPod calls.

- [x] **Step 6: Review the branch diff and commit**

Run: `git diff --check && git status --short --branch`

Then commit:

```bash
git add .gitignore docker/verify_nodes.py tests/test_container_files.py tests/test_storage.py tests/test_handler.py README.md docs/superpowers/plans/2026-09-10-foodie-job-contract.md
git commit -m "test: cover Foodie worker lifecycle"
```
