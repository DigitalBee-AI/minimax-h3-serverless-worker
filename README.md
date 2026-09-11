# MiniMax H3 Serverless Worker

This repository builds a RunPod **queue-based** Serverless worker for the DBee
Dance, Foodie, and DBee KOL Management MiniMax H3 ComfyUI workflows. It reads models and input assets
from an attached private RunPod network volume, runs ComfyUI locally, and writes
one private MP4 result back to that volume. It does not serve media publicly.

The approved [design](docs/superpowers/specs/2026-09-08-minimax-h3-serverless-worker-design.md)
and [implementation plan](docs/superpowers/plans/2026-09-08-minimax-h3-serverless-worker.md)
define the deployment and live-verification procedure.
The [Foodie contract design](docs/superpowers/specs/2026-09-10-foodie-job-contract-design.md)
defines the shared-worker extension.

## Setup and architecture

The image starts ComfyUI on loopback, confirms the required node classes, and
then starts the RunPod handler. The handler validates a Dance, Foodie, DBee AI UGC, or DBee Product Showcase
request, copies its assets from the private volume to ComfyUI's input directory,
executes the workflow, and atomically publishes node 42's only MP4 output.

The mounted network volume is `j4ds1uajmj` in `US-KS-2`, at
`/runpod-volume`. Populate this private layout before submitting a job:

```text
/runpod-volume/
├── models/
│   ├── diffusion_models/10Eros_Max_h3_TURBO-hybrid_beta4.safetensors
│   ├── text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   ├── vae/minimax_h3_video_vae_fp16.safetensors
│   ├── vae/minimax_h3_audio_vae_fp32.safetensors
│   └── latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors
└── jobs/<run-id>/
    ├── input/<Dance or Foodie assets>
    └── output/result.mp4
```

The direct runtime dependencies are `requests==2.32.5` and `runpod==1.8.1`.
The container pins CUDA `13.0.2`, PyTorch `torch==2.10.0`,
`torchvision==0.25.0`, and `torchaudio==2.10.0` from the CUDA 13.0 wheel index;
ComfyUI `8a33128f2f8c5585c57486c07de481241e70a39c`; ComfyUI-KJNodes
`57105374f47d0fbb49c9c3926fb981702e0a4b5c`; ComfyUI-VideoHelperSuite
`115de7a9d9e34410cffb9ecfd268e993b11a50fb`; and
Comfyui_Minimax_h3_latent_Upscaler `d7c01b9011f2e8439493f6c02c29995a27df276f`.

## Private job contract

`tests/fixtures/job-input.json` is a Dance **contract-validation-only** fixture and is **not runnable**:
it contains only nodes 9, 42, and 43, without the complete generation graph.
Copy it to `job-request.json` as a request template. Replace `input.workflow` with the
full approved API-format workflow before `POST /run`, retaining the complete graph
and setting nodes 9 and 43 to the uploaded assets' `comfy_name` values.

Before calling RunPod, the website backend uploads every asset to the private
volume paths in the prepared request. Each `volume_path` is relative to the
mounted volume and must resolve beneath `jobs/<run-id>/input`; it is never an
S3 URL or public URL.

Dance requests may omit `job_type` for backward compatibility or set it to
`"dance"`. They require exactly one image and one video. Node 9 must be
`LoadImage` and reference the image, node 43 must be `VHS_LoadVideo` and
reference the video, and node 42 must be `VHS_VideoCombine`.

Foodie requests set `job_type` to `"foodie"` and require exactly these assets:

```json
{
  "input": {
    "job_type": "foodie",
    "run_id": "foodie_example-id",
    "workflow": {},
    "assets": [
      {
        "kind": "image",
        "role": "kol",
        "volume_path": "jobs/foodie_example-id/input/kol.png",
        "comfy_name": "kol.png"
      },
      {
        "kind": "image",
        "role": "storyboard",
        "volume_path": "jobs/foodie_example-id/input/storyboard.png",
        "comfy_name": "storyboard.png"
      },
      {
        "kind": "audio",
        "role": "voice",
        "volume_path": "jobs/foodie_example-id/input/voice.wav",
        "comfy_name": "voice.wav"
      }
    ],
    "output_node_id": "42"
  }
}
```

Replace the empty Foodie `workflow` with the complete API-format graph. Node 9
must be `LoadImage` and reference the KOL image, node 48 must be `LoadImage` and
reference the storyboard, node 50 must be `LoadAudio` and reference the voice
WAV, and node 42 must be `VHS_VideoCombine`. Asset roles, kinds, and
`comfy_name` values are validated before ComfyUI runs.

DBee KOL Management uses two additional contracts while retaining the same
private-volume and output rules:

- `dbee-ai-ugc` requires exactly `character`, `product`, `environment`, and
  `storyboard` images plus `voice` WAV audio. They must match workflow nodes 9,
  51, 52, 48, and 50 respectively; node 42 remains the final MP4 output.
- `dbee-product-showcase` requires 1–9 ordered images named `picture-1` through
  `picture-9` with no gaps. LoadImage nodes 100–108 must match that order and
  node 11 must reference each corresponding node through
  `ref_images.ref_image_N`. Node 42 remains the final MP4 output.

DBee uses deterministic run IDs in the form `dbee-<generation-id>` and uploads
inputs below `jobs/<run-id>/input/`. The shared handler does not give any DBee
request priority over Foodie or Dance; admission limits belong to each calling
backend.

Use the configured endpoint only from server-side code. The submit route is
`POST /run`; the status route is `GET /status/{job-id}`:

```bash
curl --request POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data @job-request.json

curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/{job-id}" \
  --header "Authorization: Bearer $RUNPOD_API_KEY"
```

A successful completed job has a private result reference, not a public URL,
credential, or base64 video:

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

RunPod and S3 credentials belong only in server-side secret stores. Do not
commit them, pass them as image build arguments, expose them to browsers, or
log them.

Retrieve the completed MP4 through RunPod's private S3-compatible storage,
using a server-side `runpod` profile whose credentials are held in a secret
store. The bucket is `j4ds1uajmj`; use region `us-ks-2` and the endpoint
`https://s3api-us-ks-2.runpod.io`:

```bash
aws s3 cp \
  s3://j4ds1uajmj/jobs/018f-example-id/output/result.mp4 \
  result.mp4 \
  --profile runpod \
  --region us-ks-2 \
  --endpoint-url https://s3api-us-ks-2.runpod.io
```

## Endpoint deployment

The current shared endpoint is built from GitHub source
`DigitalBee-AI/minimax-h3-serverless-worker`, branch `main`, context path `/`, and
Dockerfile path `Dockerfile`. Attach volume `j4ds1uajmj` and retain these exact
settings:

| Field | Value |
| --- | --- |
| Type | Queue-based |
| GPU | RTX PRO 6000 |
| GPU count | one GPU per worker |
| Max workers | Max workers: 1 |
| Active workers | Active workers: 0 |
| Auto scaling | Queue delay: 1 second |
| Idle timeout | Idle timeout: 5 seconds |
| FlashBoot | FlashBoot: enabled |
| Execution timeout | Execution timeout: 3600 seconds |
| Allowed CUDA | CUDA 13.0 or newer |

Build and verify locally before changing the endpoint:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
docker build --platform linux/amd64 -t minimax-h3-serverless-worker:test .
```

## Logs and troubleshooting

Worker logs contain lifecycle states only. They must not contain full prompts,
media bytes, RunPod credentials, or S3 credentials. For a private-volume
mount investigation, deploy a diagnostic worker environment with:

```bash
NETWORK_VOLUME_DEBUG=true
```

Check that all five model files are readable from `/runpod-volume/models`, that
ComfyUI reports the required node classes, and that the input paths exist
before submitting a job. The old Pod must remain until live verification passes:
a Serverless-generated MP4 must be downloaded by the website backend and pass
its media probe before the old Pod is stopped.
