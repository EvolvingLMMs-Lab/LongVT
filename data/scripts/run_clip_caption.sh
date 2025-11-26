export OPENAI_BASE_URL=""
export OPENAI_API_KEY=""
export READER_BACKEND="qwen"

python launch/clip_caption.py \
    --input_path detect_results.json \
    --output_path caption_results.json \
    --server openai \
    --fps 4 --shard-size 4 --shard-index 0