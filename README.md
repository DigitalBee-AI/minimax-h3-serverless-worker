# MiniMax H3 Serverless Worker

This repository builds a RunPod **queue-based** Serverless worker for the KOL
Dance MiniMax H3 ComfyUI workflow. It reads models and input assets from an
attached private RunPod network volume, runs ComfyUI locally, and writes one
private MP4 result back to that volume. It does not serve media publicly.

The approved [design](docs/superpowers/specs/2026-09-08-minimax-h3-serverless-worker-design.md)
and [implementation plan](docs/superpowers/plans/2026-09-08-minimax-h3-serverless-worker.md)
define the deployment and live-verification procedure.

## Setup and architecture

The image starts ComfyUI on loopback, confirms the required node classes, and
then starts the RunPod handler. The handler validates one image and one video,
copies them from the private volume to ComfyUI's input directory, executes the
workflow, and atomically publishes node 42's only MP4 output.

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
    ├── input/<run-id>-kol.png
    ├── input/<run-id>-source.mp4
    └── output/result.mp4
```

The direct runtime dependencies are `requests==2.32.5` and `runpod==1.8.1`.
The container pins CUDA `13.0.2`, PyTorch `torch==2.10.0`,
`torchvision==0.25.0`, and `torchaudio==2.10.0` from the CUDA 13.0 wheel index;
ComfyUI `8a33128f2f8c5585c57486c07de481241e70a39c`; ComfyUI-KJNodes
`c2a47f161bdcecc1e6baf3412f1d116febc26ce3`; ComfyUI-VideoHelperSuite
`115de7a9d9e34410cffb9ecfd268e993b11a50fb`; and
Comfyui_Minimax_h3_latent_Upscaler `d7c01b9011f2e8439493f6c02c29995a27df276f`.

## Private job contract

Before calling RunPod, the website backend uploads both assets to the private
volume paths in `tests/fixtures/job-input.json`. The request payload is the
fixture's `input` object. Each `volume_path` is relative to the mounted volume
and must resolve beneath `jobs/<run-id>/input`; it is never an S3 URL or public
URL. Node 9 is `LoadImage`, node 43 is `VHS_LoadVideo`, and node 42 is
`VHS_VideoCombine`.

Use the RunPod endpoint ID placeholder in server-side requests:

```bash
curl --request POST "https://api.runpod.ai/v2/<endpoint-id>/run" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data @tests/fixtures/job-input.json

curl "https://api.runpod.ai/v2/<endpoint-id>/status/{job-id}" \
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

## Endpoint deployment

Configure endpoint `minimaxh3_beta4` from GitHub source
`locust08/minimax-h3-serverless-worker`, branch `main`, context path `/`, and
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
| FlashBoot | FlashBoot: enabled |
| Execution timeout | Execution timeout: 3600 seconds |
| Allowed CUDA | CUDA 13.0 or newer |

Build and verify locally before changing the endpoint:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/pytest
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
