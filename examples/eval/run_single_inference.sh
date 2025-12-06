#!/bin/bash
#
# Single Sample Inference with LongVT
#
# This script starts a vLLM server and runs single sample inference.
# For full benchmark evaluation, use run_eval.sh instead.
#
# Usage:
#   bash run_single_inference.sh <CKPT_PATH> <VIDEO_PATH> <QUESTION> [IS_QWEN3_VL]
#
# Example:
#   bash run_single_inference.sh \
#       longvideotool/LongVT-7B-RFT \
#       /path/to/video.mp4 \
#       "What is happening in the video?"

CKPT_PATH=$1
VIDEO_PATH=$2
QUESTION=$3
IS_QWEN3_VL=${4:-False}

if [ -z "$CKPT_PATH" ] || [ -z "$VIDEO_PATH" ] || [ -z "$QUESTION" ]; then
    echo "Usage: bash run_single_inference.sh <CKPT_PATH> <VIDEO_PATH> <QUESTION> [IS_QWEN3_VL]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start vLLM server (Qwen3-VL has native tool support, others need custom template)
if [ "$IS_QWEN3_VL" == "True" ]; then
    vllm serve $CKPT_PATH \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        --trust-remote-code \
        --port 8000 &
else
    vllm serve $CKPT_PATH \
        --chat-template ${SCRIPT_DIR}/tool_call_qwen2_5_vl.jinja \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        --trust-remote-code \
        --port 8000 &
fi

VLLM_PID=$!
echo "Started vLLM server (PID: $VLLM_PID), waiting for it to be ready..."
sleep 240

# Run inference
python ${SCRIPT_DIR}/single_inference.py \
    --video_path "$VIDEO_PATH" \
    --question "$QUESTION" \
    --fps 1 \
    --max_frames 512 \
    --max_pixels 50176

# Cleanup
kill $VLLM_PID 2>/dev/null

