#!/bin/sh
set -eu

COMFY_PID=""
HANDLER_PID=""

terminate_children() {
    for pid in "$HANDLER_PID" "$COMFY_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    for pid in "$HANDLER_PID" "$COMFY_PID"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

on_exit() {
    status=$?
    trap - EXIT TERM INT
    terminate_children
    exit "$status"
}

on_term() {
    trap - EXIT TERM INT
    terminate_children
    exit 143
}

on_int() {
    trap - EXIT TERM INT
    terminate_children
    exit 130
}

trap on_exit EXIT
trap on_term TERM
trap on_int INT

python /comfyui/main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --extra-model-paths-config /app/extra_model_paths.yaml &
COMFY_PID=$!

deadline=$(($(date +%s) + 300))
until wget -q -O /dev/null --timeout=1 --tries=1 http://127.0.0.1:8188/system_stats; do
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
        wait "$COMFY_PID" 2>/dev/null || true
        echo "ComfyUI exited before becoming ready" >&2
        exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ComfyUI did not become ready within 300 seconds" >&2
        exit 1
    fi
    sleep 1
done

python /app/verify_nodes.py http://127.0.0.1:8188/object_info

python /app/handler.py &
HANDLER_PID=$!
wait "$HANDLER_PID"
