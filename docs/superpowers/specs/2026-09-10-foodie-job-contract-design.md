# Foodie Job Contract Design

## Goal

Allow the existing MiniMax H3 RunPod worker to execute both the established Dance request and the Foodie request submitted by DBee KOL Management. The Foodie request contains one KOL image, one storyboard image, and one voice WAV file.

## Scope

This change is limited to the worker repository. It does not change the website, Supabase, Cloudflare, the RunPod endpoint, or any deployed image.

## Request dispatch

The worker selects validation by `job_type`:

- A missing `job_type` or `job_type: "dance"` uses the existing Dance contract.
- `job_type: "foodie"` uses the Foodie contract.
- Every other value is rejected before ComfyUI runs.

The Dance contract remains one image plus one video, with workflow nodes 9 (`LoadImage`), 43 (`VHS_LoadVideo`), and 42 (`VHS_VideoCombine`).

The Foodie contract requires exactly these role-aware assets:

- `kind: "image", role: "kol"`, referenced by node 9 (`LoadImage`).
- `kind: "image", role: "storyboard"`, referenced by node 48 (`LoadImage`).
- `kind: "audio", role: "voice"`, referenced by node 50 (`LoadAudio`).
- Node 42 must remain `VHS_VideoCombine`.

All assets retain the existing filename, path containment, regular-file, symlink, and unique `comfy_name` checks. Foodie voice input accepts `.wav`, matching the website's generated `voice.wav` contract.

## Execution and output

The existing handler and storage lifecycle remains shared. After validation, all request assets are staged in `/comfyui/input`, the supplied workflow runs, node 42 yields one MP4, and the worker publishes `jobs/<run-id>/output/result.mp4` with the existing response shape. Temporary staged inputs are removed after success or failure.

## Verification

Tests prove that the unchanged Dance payload still passes, the exact Foodie payload passes, and missing, duplicate, mislabeled, or unsupported Foodie assets fail before execution. A handler test proves all three Foodie inputs are staged and cleaned up. The full Python test suite must pass locally; no paid RunPod job is submitted.
