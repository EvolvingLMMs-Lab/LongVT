#!/bin/bash
# -*- coding: utf-8 -*-
#
# QA Filter VL Script (VLM-based)
# 
# This script filters QA pairs using Vision-Language Models to verify
# that video segments contain sufficient visual evidence.
#
# Usage:
#   ./scripts/run_qa_filter_vl.sh --input-dir /path/to/qa --output-dir /path/to/output
#
# Environment Variables:
#   VLM_API_BASE        - VLM API base URL (default: http://localhost:8000/v1)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"

# Default configuration
QUALITY_THRESHOLD=0.85
TARGET_FPS=2.0
MAX_DURATION=300.0

# Help message
show_help() {
    echo "QA Filter VL Script (VLM-based)"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Options:"
    echo "  --input-dir DIR          Input directory containing QA JSON files"
    echo "  --output-dir DIR         Output directory for filtered results"
    echo ""
    echo "Optional Options:"
    echo "  --video-list-file FILE   Path to video list file"
    echo "  --quality-threshold N    VLM score threshold (default: 0.85)"
    echo "  --target-fps N           Target FPS for video processing (default: 2.0)"
    echo "  --max-duration N         Maximum crop duration in seconds (default: 300.0)"
    echo "  --no-skip-existing       Don't skip already processed files"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VLM_API_BASE             VLM API base URL (default: http://localhost:8000/v1)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 --input-dir ./qa --output-dir ./filtered"
    echo ""
    echo "  # With video list file"
    echo "  $0 --input-dir ./qa --output-dir ./filtered --video-list-file ./videos.txt"
    echo ""
    echo "  # With custom threshold"
    echo "  $0 --input-dir ./qa --output-dir ./filtered --quality-threshold 0.9"
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
        --video-list-file)
            PYTHON_ARGS+=(--video-list-file "$2")
            shift 2
            ;;
        --quality-threshold)
            QUALITY_THRESHOLD="$2"
            PYTHON_ARGS+=(--quality-threshold "$2")
            shift 2
            ;;
        --target-fps)
            TARGET_FPS="$2"
            PYTHON_ARGS+=(--target-fps "$2")
            shift 2
            ;;
        --max-duration)
            MAX_DURATION="$2"
            PYTHON_ARGS+=(--max-duration "$2")
            shift 2
            ;;
        --no-skip-existing)
            # Don't add --skip-existing flag
            shift
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

# Check input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "QA Filter VL Configuration"
echo "=========================================="
echo "Input Directory:     $INPUT_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Quality Threshold:   $QUALITY_THRESHOLD"
echo "Target FPS:          $TARGET_FPS"
echo "Max Duration:        ${MAX_DURATION}s"
echo "VLM API Base:        ${VLM_API_BASE:-http://localhost:8000/v1}"
echo "=========================================="

# Add skip-existing by default
PYTHON_ARGS+=(--skip-existing)

# Run QA filter VL
python "$DATA_DIR/launch/qa_filter_vl.py" "${PYTHON_ARGS[@]}"

echo "QA filter VL complete!"

