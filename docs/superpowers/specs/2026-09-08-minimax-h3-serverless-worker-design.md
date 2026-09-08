# MiniMax H3 Serverless Worker Design

## Objective

Build a reproducible RunPod queue-based Serverless worker for the KOL Dance workflow. The worker must run the checked-in MiniMax H3 Beta 4 ComfyUI API workflow, read private inputs and models from the attached RunPod network volume, and publish the generated MP4 back to that volume for retrieval by the website backend.

## Scope

This repository contains only the GPU worker image, handler, tests, and deployment documentation. Updating the KOL Dance Next.js application is a separate follow-up change after the worker passes an end-to-end RunPod test.

The repository contains no API keys, model weights, KOL images, source videos, or generated videos.

## Selected approach

Build from the RunPod worker-comfyui 5.8.6 implementation while replacing its request validation and output collection with a small MiniMax H3-specific layer. This preserves RunPod queue handling and ComfyUI process management while adding volume-backed video inputs and MP4 output support.

A thin layer over the prebuilt 5.8.6 base image was rejected because that image can differ from the CUDA and ComfyUI versions proven on the existing RTX PRO 6000 Blackwell Pod. A load-balancing endpoint was rejected because the existing deployment is queue-based and the workflow benefits from durable job status and automatic retry behavior.

## Runtime versions

The image pins the versions verified on the working Pod:

- CUDA runtime: 13.0
- PyTorch: 2.10.0 with CUDA 13.0 wheels
- ComfyUI commit: `8a33128f2f8c5585c57486c07de481241e70a39c` (`v0.34.0-16-g8a33128f`)
- ComfyUI-KJNodes commit: `c2a47f161bdcecc1e6baf3412f1d116febc26ce3`
- ComfyUI-VideoHelperSuite commit: `115de7a9d9e34410cffb9ecfd268e993b11a50fb`
- Comfyui_Minimax_h3_latent_Upscaler commit: `d7c01b9011f2e8439493f6c02c29995a27df276f`
- RunPod worker behavior baseline: `worker-comfyui` 5.8.6

The Docker build uses `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` and installs the pinned Python and Git dependencies. Build-time checks start ComfyUI in CPU quick-test mode and confirm every workflow node class is registered.

## Network-volume layout

RunPod mounts the selected network volume at `/runpod-volume`. S3 object keys map directly to these paths.

```text
/runpod-volume/
├── models/
│   ├── diffusion_models/10Eros_Max_h3_TURBO-hybrid_beta4.safetensors
│   ├── text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   ├── vae/minimax_h3_video_vae_fp16.safetensors
│   ├── vae/minimax_h3_audio_vae_fp32.safetensors
│   └── latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors
└── jobs/
    └── <run-id>/
        ├── input/
        │   ├── <run-id>-kol.<extension>
        │   └── <run-id>-source.<extension>
        └── output/
            └── result.mp4
```

ComfyUI receives `/runpod-volume/models` through `extra_model_paths.yaml`. The worker never downloads model weights during startup.

## Request contract

The website backend uploads both inputs through RunPod's S3-compatible API before submitting the job. It then sends:

```json
{
  "input": {
    "run_id": "018f-example-id",
    "workflow": {},
    "assets": [
      {
        "kind": "image",
        "volume_path": "jobs/018f-example-id/input/018f-example-id-kol.png",
        "comfy_name": "018f-example-id-kol.png"
      },
      {
        "kind": "video",
        "volume_path": "jobs/018f-example-id/input/018f-example-id-source.mp4",
        "comfy_name": "018f-example-id-source.mp4"
      }
    ],
    "output_node_id": "42"
  }
}
```

`workflow` is the complete ComfyUI API workflow. Node 9 references the image's `comfy_name`, and node 43 references the video's `comfy_name`.

The worker accepts exactly one image and one video asset. `run_id` and `comfy_name` use a conservative filename character set. Each `volume_path` must be relative, must resolve inside `/runpod-volume/jobs/<run-id>/input`, and must name an existing regular file. Symlinks, absolute paths, traversal segments, duplicate names, and unsupported file extensions are rejected.

## Processing flow

1. Validate the request and all paths before copying any file.
2. Copy the two volume-backed assets into `/comfyui/input` using their unique `comfy_name` values.
3. Confirm the workflow references both staged filenames and declares node 42 as `VHS_VideoCombine`.
4. Submit the workflow to the local ComfyUI API and wait for its terminal history result.
5. Read node 42's output entries and select exactly one regular `.mp4` file beneath `/comfyui/output`.
6. Atomically copy the MP4 to `/runpod-volume/jobs/<run-id>/output/result.mp4`.
7. Return metadata containing the private relative path, byte size, and filename.
8. Remove the two staged files from `/comfyui/input` in a `finally` block. Keep the private volume inputs and output until the website confirms retrieval.

Only one RunPod worker is configured initially, but unique run IDs and filenames keep the handler safe if concurrency is increased later.

## Success response

```json
{
  "status": "success",
  "run_id": "018f-example-id",
  "video": {
    "volume_path": "jobs/018f-example-id/output/result.mp4",
    "filename": "result.mp4",
    "size_bytes": 12345678
  }
}
```

The response never contains a public URL, credentials, or base64 video data.

## Error handling

Invalid requests return a concise validation error before ComfyUI runs. Missing model or node errors propagate as failed RunPod jobs with the relevant ComfyUI validation message. Execution errors include the failing node and exception message when ComfyUI provides them. Missing, ambiguous, non-MP4, or unsafe node 42 output fails the job instead of returning an uncertain result.

The handler writes lifecycle events to standard output without prompts, workflow text, image data, video data, S3 credentials, or RunPod API keys.

## Tests and verification

Unit tests cover:

- Required request fields and asset counts.
- Run ID and filename validation.
- Absolute paths, traversal, symlinks, wrong job prefixes, and missing files.
- Workflow filename and output-node validation.
- Safe staging and cleanup of input files.
- MP4 discovery only beneath the configured ComfyUI output directory.
- Rejection of missing and multiple MP4 candidates.
- Atomic publication to the expected volume path.
- Stable success-response structure.
- Propagation of ComfyUI execution failures.

Container checks cover dependency installation, importability, ComfyUI CPU quick-test startup, and registration of all node classes used by the current workflow.

The first live verification uses the current workflow with a short reference clip, Max workers 1, Active workers 0, the attached `j4ds1uajmj` volume, and the RTX PRO 6000 GPU. The old Pod remains available until the Serverless job completes and the downloaded MP4 passes the website's media probe.

## Deployment

RunPod builds the public GitHub repository through its GitHub integration. The endpoint attaches network volume `j4ds1uajmj` in `US-KS-2`, uses one RTX PRO 6000 GPU per worker, allows CUDA 13.0 or newer hosts, and retains the previously selected worker limits and timeouts.

Secrets are configured only in RunPod or the KOL Dance server environment. No secret is accepted as a Docker build argument or committed to Git.
