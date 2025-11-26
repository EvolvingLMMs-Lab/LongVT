#!/bin/bash
# -*- coding: utf-8 -*-
#
# QA Filter Text Script
# 
# This script filters QA pairs using LLM-based text analysis.
#
# Usage:
#   ./scripts/run_qa_filter_text.sh --input-dir /path/to/qa --output-dir /path/to/output --summary-file /path/to/summary.json
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
TOTAL_SHARDS=1

# Help message
show_help() {
    echo "QA Filter Text Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Options:"
    echo "  --input-dir DIR      Input directory containing QA JSON files"
    echo "  --output-dir DIR     Output directory for filtered QA pairs"
    echo "  --summary-file FILE  Path to video summary JSON file"
    echo ""
    echo "Optional Options:"
    echo "  --model MODEL        LLM model to use (gpt-4o or o3, default: gpt-4o)"
    echo "  --shard-id N         Current shard ID for parallel processing (0-based)"
    echo "  --total-shards N     Total number of shards (default: 1)"
    echo "  --log-level LEVEL    Log level (DEBUG, INFO, WARNING, ERROR, default: INFO)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY       OpenAI API key (required)"
    echo "  OPENAI_BASE_URL      OpenAI-compatible API base URL (optional)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 --input-dir ./qa --output-dir ./filtered --summary-file ./summary.json"
    echo ""
    echo "  # With parallel processing"
    echo "  $0 --input-dir ./qa --output-dir ./filtered --summary-file ./summary.json --shard-id 0 --total-shards 4"
    echo ""
}

# Parse arguments
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            PYTHON_ARGS+=(--input-dir "$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            PYTHON_ARGS+=(--output-dir "$2")
            shift 2
            ;;
        --summary-file)
            SUMMARY_FILE="$2"
            PYTHON_ARGS+=(--summary-file "$2")
            shift 2
            ;;
        --model)
            MODEL="$2"
            PYTHON_ARGS+=(--model "$2")
            shift 2
            ;;
        --shard-id)
            PYTHON_ARGS+=(--shard-id "$2")
            shift 2
            ;;
        --total-shards)
            PYTHON_ARGS+=(--total-shards "$2")
            shift 2
            ;;
        --log-level)
            PYTHON_ARGS+=(--log-level "$2")
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
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]] || [[ -z "$SUMMARY_FILE" ]]; then
    echo "Error: --input-dir, --output-dir, and --summary-file are required"
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

# Check summary file
if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "Error: Summary file does not exist: $SUMMARY_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "QA Filter Text Configuration"
echo "=========================================="
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Summary File:     $SUMMARY_FILE"
echo "Model:            $MODEL"
echo "=========================================="

# Run QA filter text
python "$DATA_DIR/launch/qa_filter_text.py" "${PYTHON_ARGS[@]}"

echo "QA filter text complete!"

