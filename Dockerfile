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
