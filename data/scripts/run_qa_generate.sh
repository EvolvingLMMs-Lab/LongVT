#!/bin/bash
# -*- coding: utf-8 -*-
#
# QA Generate Script
# 
# This script generates question-answer pairs from video captions.
#
# Usage:
#   ./scripts/run_qa_generate.sh --input-dir /path/to/captions --output-dir /path/to/output
#
# Environment Variables:
#   OPENAI_API_KEY      - OpenAI API key (required)
#   OPENAI_BASE_URL     - OpenAI-compatible API base URL (optional)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"

# Default configuration
MODEL="gpt-4o"
GROUP_SIZE=15
NUM_SHARDS=1
SHARD_IDX=0

# Help message
show_help() {
    echo "QA Generate Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Options:"
    echo "  --input-dir DIR      Input directory containing caption JSON files"
    echo "  --output-dir DIR     Output directory for generated QA pairs"
    echo ""
    echo "Optional Options:"
    echo "  --model MODEL        LLM model to use (default: gpt-4o)"
    echo "  --group-size N       Number of segments to merge per group (default: 15)"
    echo "  --num-shards N       Total number of shards for parallel processing (default: 1)"
    echo "  --shard-idx N        Current shard index, 0-based (default: 0)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY       OpenAI API key (required)"
    echo "  OPENAI_BASE_URL      OpenAI-compatible API base URL (optional)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 --input-dir ./captions --output-dir ./qa_output"
    echo ""
    echo "  # With parallel processing (4 shards)"
    echo "  $0 --input-dir ./captions --output-dir ./qa_output --num-shards 4 --shard-idx 0"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --shard-idx)
            SHARD_IDX="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --input-dir and --output-dir are required"
    show_help
    exit 1
fi

# Check API key
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Check input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "QA Generate Configuration"
echo "=========================================="
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model:            $MODEL"
echo "Group Size:       $GROUP_SIZE"
echo "Num Shards:       $NUM_SHARDS"
echo "Shard Index:      $SHARD_IDX"
echo "=========================================="

# Run QA generate
python "$DATA_DIR/launch/qa_generate.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --group-size "$GROUP_SIZE" \
    --num-shards "$NUM_SHARDS" \
    --shard-idx "$SHARD_IDX"

echo "QA generate complete!"

