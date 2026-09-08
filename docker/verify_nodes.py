from __future__ import annotations

import sys

import requests


REQUIRED_NODE_CLASSES = frozenset(
    {
        "BasicGuider",
        "BasicScheduler",
        "CLIPLoader",
        "ComfyMathExpression",
        "ImageResizeKJv2",
        "KSamplerSelect",
        "LTXVConcatAVLatent",
        "LTXVSeparateAVLatent",
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
)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: verify_nodes.py <object-info-url>")

    response = requests.get(arguments[0], timeout=30)
    response.raise_for_status()
    object_info = response.json()
    if not isinstance(object_info, dict):
        raise ValueError("ComfyUI object_info response must be an object")

    missing = sorted(REQUIRED_NODE_CLASSES.difference(object_info))
    if missing:
        print(*missing, sep="\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
