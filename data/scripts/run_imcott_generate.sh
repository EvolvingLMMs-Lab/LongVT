#!/bin/bash
# -*- coding: utf-8 -*-
#
# iMCoTT Generate Script
# 
# This script generates multi-turn reasoning traces with tool calling for video QA.
#
# Usage:
#   ./scripts/run_imcott_generate.sh --input-file /path/to/qa.json --output-dir /path/to/output --video-root /path/to/videos
#
# Environment Variables:
#   GOOGLE_API_KEY or GEMINI_API_KEY  - Gemini API key
#   OPENAI_API_KEY                    - OpenAI API key (for OpenAI-compatible APIs)
#   OPENAI_BASE_URL                   - OpenAI-compatible API base URL
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"

# Default configuration
GLOBAL_FRAMES_DIR="./global_sampling"
SEGMENT_FRAMES_DIR="./segment_sampling"

# Help message
show_help() {
    echo "iMCoTT Generate Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Options:"
    echo "  --input-file FILE        Input JSON file with QA data"
    echo "  --output-dir DIR         Output directory for generated traces"
    echo "  --video-root DIR         Root directory for video files"
    echo ""
    echo "Optional Options:"
    echo "  --global-frames-dir DIR  Directory for global frames cache (default: ./global_sampling)"
    echo "  --segment-frames-dir DIR Directory for segment frames cache (default: ./segment_sampling)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GOOGLE_API_KEY           Gemini API key"
    echo "  GEMINI_API_KEY           Gemini API key (alternative)"
    echo "  OPENAI_API_KEY           OpenAI API key (for OpenAI-compatible APIs)"
    echo "  OPENAI_BASE_URL          OpenAI-compatible API base URL"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 --input-file ./qa.json --output-dir ./output --video-root ./videos"
    echo ""
    echo "  # With custom cache directories"
    echo "  $0 --input-file ./qa.json --output-dir ./output --video-root ./videos \\"
    echo "     --global-frames-dir ./cache/global --segment-frames-dir ./cache/segment"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --video-root)
            VIDEO_ROOT="$2"
            shift 2
            ;;
        --global-frames-dir)
            GLOBAL_FRAMES_DIR="$2"
            shift 2
            ;;
        --segment-frames-dir)
            SEGMENT_FRAMES_DIR="$2"
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
if [[ -z "$INPUT_FILE" ]] || [[ -z "$OUTPUT_DIR" ]] || [[ -z "$VIDEO_ROOT" ]]; then
    echo "Error: --input-file, --output-dir, and --video-root are required"
    show_help
    exit 1
fi

# Check input file
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Check video root
if [[ ! -d "$VIDEO_ROOT" ]]; then
    echo "Error: Video root directory does not exist: $VIDEO_ROOT"
    exit 1
fi

# Check API key
if [[ -z "$GOOGLE_API_KEY" ]] && [[ -z "$GEMINI_API_KEY" ]] && [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Warning: No API key set. Please set GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY"
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GLOBAL_FRAMES_DIR"
mkdir -p "$SEGMENT_FRAMES_DIR"

echo "=========================================="
echo "iMCoTT Generate Configuration"
echo "=========================================="
echo "Input File:          $INPUT_FILE"
echo "Output Directory:    $OUTPUT_DIR"
echo "Video Root:          $VIDEO_ROOT"
echo "Global Frames Dir:   $GLOBAL_FRAMES_DIR"
echo "Segment Frames Dir:  $SEGMENT_FRAMES_DIR"
echo "=========================================="

# Run iMCoTT generate
python "$DATA_DIR/launch/imcott_generate.py" \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --video-root "$VIDEO_ROOT" \
    --global-frames-dir "$GLOBAL_FRAMES_DIR" \
    --segment-frames-dir "$SEGMENT_FRAMES_DIR"

echo "iMCoTT generate complete!"

