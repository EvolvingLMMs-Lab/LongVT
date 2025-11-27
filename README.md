
# LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling
<div align="center">
  <img src="assets/cover.png" alt="LongVT Cover" width="800"/>
</div>

<br>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.20785)
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white)](https://evolvinglmms-lab.github.io/LongVT/)
[![Code](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/EvolvingLMMs-Lab/LongVT)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/datasets/longvideotool/LongVT-Parquet)
[![Models](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/collections/lmms-lab/longvt)
[![Demo](https://img.shields.io/badge/Demo-FF6F00?style=for-the-badge&logo=gradio&logoColor=ffffff)](https://huggingface.co/spaces/longvideotool/LongVT-Demo)
[![Blog](https://img.shields.io/badge/Blog-lmms_lab?style=for-the-badge&logo=blogger&logoColor=white)](https://www.lmms-lab.com/posts/longvt/)

[![Daily Paper](https://img.shields.io/badge/🚀_Daily_Paper-FF9D00?style=for-the-badge)](https://huggingface.co/papers/2511.20785)
</div>

## 🎉 News
- **[2025-11]**: We have created a fun cartoon ([Conan_EN](https://drive.google.com/file/d/1sk9YfmtcQq0nLlI5K_G3BOziRSEjMMT0/view?usp=sharing) | [Conan_CN](https://drive.google.com/file/d/14MJEN_FBRJNJ9IYWK-fs4IbZZSgvKGAw/view?usp=sharing)) to explain LongVT. Enjoy :) Credit to the amazing [NotebookLM](https://notebooklm.google.com/) and [Gemini-3](https://blog.google/products/gemini/gemini-3/#learn-anything).
- **[2025-11]**: Join our WeChat group by scanning this [QR code](assets/qr_code.jpg).
- **[2025-11]**: We release all of our codes, data, and model checkpoints! Check out the [LongVT collection on Hugging Face](https://huggingface.co/collections/lmms-lab/longvt).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [SFT Training](#1-sft-training)
  - [RL Training](#2-rl-training)
  - [RFT Training](#3-rft-training)
  - [Evaluation](#4-evaluation)
  - [Data Pipeline](#5-data-pipeline)
- [Getting Started](#getting-started)
  - [Data Preparation](#data-preparation)
  - [SFT Training](#sft-training)
  - [RL Training](#rl-training)
  - [RFT Training](#rft-training)
  - [Evaluation](#evaluation)
  - [LLM Judge Setup](#llm-judge-setup)
  - [Data Pipeline](#data-pipeline)
- [Evaluation Results](#evaluation-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Star History](#-star-history)

## Overview

<div align="center">
  <img src="assets/teaser.png" alt="teaser" width="1000"/>
</div>

Large multimodal models (LMMs) have shown great potential for video reasoning with textual Chain-of-Thought. However, they remain vulnerable to hallucinations, especially when processing long-form videos where evidence is sparse and temporally dispersed.

Inspired by how humans comprehend long videos—by first skimming globally and then examining relevant clips for details—we introduce **LongVT**, an end-to-end agentic framework that enables "Thinking with **Long V**ideos" via interleaved Multimodal Chain-of-**T**ool-Thought.
Specifically, we exploit LMMs' inherent temporal grounding ability as a native video cropping tool to zoom in on a specific video clip and resample finer-grained video frames. This global-to-local reasoning loop continues until answers are grounded in retrieved visual evidence. Given the scarcity of fine-grained question-answering (QA) data for the long video reasoning task, we curate and will release a data suite named **VideoSIAH** to facilitate both training and evaluation. Specifically, our training dataset consists of 247.9K samples for tool-integrated cold-start supervised fine-tuning, 1.6K samples for agentic reinforcement learning, and 15.4K samples for agentic reinforcement fine-tuning, respectively. Our evaluation benchmark consists of 1,280 QA pairs that are carefully curated through a semi-automatic data pipeline with human-in-the-loop validation. With a meticulously designed three-stage training strategy and extensive empirical validation, LongVT consistently outperforms existing strong baselines across four challenging long-video understanding and reasoning benchmarks.


## Installation

### 1. SFT Training

Please follow the installation instructions in [lmms-engine](https://github.com/EvolvingLMMs-Lab/lmms-engine) to prepare the environment for supervised fine-tuning.

### 2. RL Training

We provide our source implementation of `verl`, which is a detached fork from the original [verl](https://github.com/volcengine/verl) repository. You may choose to use either our integrated version or the original `verl` library for RL training. However, for seamless reproduction, we highly recommend using our provided environment.

First, clone the repository and create a dedicated Conda environment:

```bash
git clone https://github.com/EvolvingLMMs-Lab/LongVT.git
cd LongVT

conda create -n longvt python=3.10
conda activate longvt
```

Next, install the RL training pipeline and dependencies:

```bash
# Install dependencies
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install the package in editable mode without extra dependencies
pip install --no-deps -e .
```
Note: If you encounter any issues during execution, please refer to requirement_reproduce.txt to verify your dependency versions.

We also include a verl_0.6 branch in this repository. For environment installation regarding this branch, please refer to the official verl v0.6 documentation. However, please note that we strictly recommend using the main branch (as detailed above) for reliable reproduction, as the 0.6 branch may have consistency issues.

We recommend you to use separate environments if you encounter a conflict in requirements.

### 3. RFT Training

RFT (Reinforcement Fine-Tuning) shares the same environment as SFT training. Please follow the [SFT Training](#1-sft-training) installation instructions.

### 4. Evaluation

Please follow the installation instructions in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to set up the evaluation environment.

### 5. Data Pipeline

We open-sourced our data processing pipeline and code for the community to follow. The pipeline includes video scene detection, video captioning, and event merging functionalities.

To install the dependencies for data pipeline:

```bash
cd ./data

# Option 1: Using uv (recommended)
uv pip install -e .

# Option 2: Using pip
pip install -e .

# Optional: Install development dependencies for code formatting
uv pip install -e ".[dev]"
```

**Key Dependencies:**
- `decord`: Video frame extraction
- `scenedetect`: Scene boundary detection
- `qwen-vl-utils`: Qwen VL utilities for video processing
- `openai`: OpenAI-compatible API client for LLM/VLM services
- `opencv-python`: Video processing and frame extraction for iMCoTT generation
- `requests`: HTTP client for VLM API interactions

We recommend using a separate environment if you encounter conflicts with other packages.

## Getting Started

### Data Preparation

We provide our training data annotations on Hugging Face. You can download them using the following commands:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download all annotation files
huggingface-cli download longvideotool/LongVT-Parquet --repo-type dataset --local-dir ./data
```

The dataset includes:
- **SFT Data**: ~248K samples for cold-start supervised fine-tuning
- **RL Data**: ~1.6K samples for agentic reinforcement learning  
- **RFT Data**: ~15K samples for agentic reinforcement fine-tuning

**Source Videos:** The source video and image files are available on [Hugging Face](https://huggingface.co/datasets/longvideotool/LongVT-Source). Please refer to the dataset page for download instructions.

### SFT Training

After installing [lmms-engine](https://github.com/EvolvingLMMs-Lab/lmms-engine), you can launch SFT training using either:

**Option 1: Using a configuration YAML file**

```bash
# Edit the dataset paths in longvt_7b_sft.yaml
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli config_yaml=./examples/sft/longvt_7b_sft.yaml
```

**Option 2: Using the launch script**

```bash
# Edit the dataset paths and hyperparameters in the script
bash examples/sft/run_longvt_7b_sft.sh
```

**Troubleshooting:**
- If you encounter **OOM (Out of Memory)** errors, reduce the `packing_length` parameter in your configuration.

### RL Training

**Training with Ray**

To perform training in a multi-node environment, you first need to set up a Ray cluster on your head and worker nodes. While there are various ways to launch Ray, we provide a reference script to help you get started:

```bash
bash examples/video_tools/launch.sh
```
Once the Ray cluster is active, you can submit the training job using the following script:

```bash
bash examples/video_tools/longvt_7b_rl_train.sh
```

Note: Please remember to update the corresponding variables in the scripts to match your environment before running them.

### RFT Training

RFT (Reinforcement Fine-Tuning) training follows the same procedure as SFT training. Simply use the RFT configuration and scripts:

```bash
# Edit the dataset paths in longvt_7b_rft.yaml
bash examples/rft/run_longvt_7b_rft.sh
```

### Evaluation

We provide evaluation scripts and task configurations in `lmms_eval_tasks/`. Three task types are supported with different prompting strategies:

| Task Type | Description | Example Tasks |
|-----------|-------------|---------------|
| **Non-Think** | Direct answer without reasoning (for baseline models like Qwen2.5-VL) | `videomme_w_subtitle`, `video_mmmu`, `lvbench`, `longvt_non_think` |
| **Think** | With CoT reasoning process (for reasoning models like Video-R1) | `videomme_w_subtitle_reward`, `video_mmmu_reasoning`, `lvbench_reasoning`, `longvt_reasoning` |
| **Tool** | With native tool calling enabled (for agentic models like LongVT) | `videomme_w_subtitle_reward_tool`, `video_mmmu_reasoning_tool`, `lvbench_tool`, `longvt` |

To run evaluation:

```bash
# For baseline models with direct answering
bash examples/eval/run_eval.sh /path/to/checkpoint longvt_non_think False 512

# For reasoning models without tool calling
bash examples/eval/run_eval.sh /path/to/checkpoint longvt_reasoning False 512

# For reasoning models with native tool calling
bash examples/eval/run_eval.sh /path/to/checkpoint longvt False 512
```

**Arguments:**
- `$1`: Path to model checkpoint
- `$2`: Task name (see table above)
- `$3`: Whether using Qwen3-VL model (`True`/`False`)
- `$4`: Maximum number of frames (default: 768)

### LLM Judge Setup

We use an LLM-based judge both for evaluation and for computing RL rewards.  
By default, we use `Qwen/Qwen2.5-72B-Instruct` as the judge model.

**Steps:**
1. Start a judge server with vLLM or SGLang

```bash
# Example with vLLM
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --port 1234 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 131072 \
    --tensor-parallel-size 8 \
    --served-model-name "judge" \
    --trust-remote-code
```

```bash
# Example with SGLang
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-72B-Instruct \
    --tp-size 8 \
    --dp-size 1 \
    --served-model-name judge \
    --port 1234 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.75
```

2. Configure the judge endpoint in your scripts:
   Set the judge service base URL in longvt_7b_rl_train.sh via the LLM_AS_A_JUDGE_BASE environment variable.

### Data Pipeline

To follow our data curation pipeline, we provide example scripts in `data/scripts/`. The pipeline consists of the following stages:

#### 1. Segment Detection

Detect scene boundaries in long videos and segment them into clips:

```bash
cd data
bash scripts/run_detect_segment.sh
```

Or run directly:

```bash
python launch/detect_segment.py \
    --input_file /path/to/video_list.txt \
    --output_path detect_results.json \
    --num_parts 10
```

The `video_list.txt` should contain one video path per line.

#### 2. Clip Captioning

Generate captions for each detected clip using a VLM service:

```bash
export OPENAI_BASE_URL="http://your-vlm-server:8000/v1"
export OPENAI_API_KEY="your-api-key"

bash scripts/run_clip_caption.sh
```

Or run directly:

```bash
python launch/clip_caption.py \
    --input_path detect_results.json \
    --output_path caption_results.json \
    --server openai \
    --fps 4
```

#### 3. QA Generation

Generate question-answer pairs from merged video captions:

```bash
export OPENAI_API_KEY="your-api-key"

bash scripts/run_qa_generate.sh \
    --input-dir /path/to/captions \
    --output-dir /path/to/qa_output \
    --model gpt-4o
```

For parallel processing with sharding:

```bash
# Run multiple shards in parallel
bash scripts/run_qa_generate.sh --input-dir ./captions --output-dir ./qa --num-shards 4 --shard-idx 0 &
bash scripts/run_qa_generate.sh --input-dir ./captions --output-dir ./qa --num-shards 4 --shard-idx 1 &
bash scripts/run_qa_generate.sh --input-dir ./captions --output-dir ./qa --num-shards 4 --shard-idx 2 &
bash scripts/run_qa_generate.sh --input-dir ./captions --output-dir ./qa --num-shards 4 --shard-idx 3 &
```

#### 4. QA Filtering (Text-based)

Filter QA pairs using LLM-based text analysis:

```bash
export OPENAI_API_KEY="your-api-key"

bash scripts/run_qa_filter_text.sh \
    --input-dir /path/to/qa \
    --output-dir /path/to/filtered \
    --summary-file /path/to/video_summaries.json \
    --model gpt-4o
```

#### 5. QA Filtering (VLM-based)

Filter QA pairs using Vision-Language Models to verify visual evidence:

```bash
export VLM_API_BASE="http://your-vlm-server:8000/v1"

bash scripts/run_qa_filter_vl.sh \
    --input-dir /path/to/qa \
    --output-dir /path/to/filtered \
    --quality-threshold 0.85
```

#### 6. iMCoTT Generation

Generate multi-turn reasoning traces with tool calling:

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
# Or use OpenAI-compatible API:
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="http://your-server:8000/v1"

bash scripts/run_imcott_generate.sh \
    --input-file /path/to/qa.json \
    --output-dir /path/to/traces \
    --video-root /path/to/videos
```

#### Pipeline Overview

The complete data pipeline follows this flow:

```
Long Videos
    ↓
Segment Detection (detect_segment.py)
    ↓
Clip Captioning (clip_caption.py)
    ↓
QA Generation (qa_generate.py)
    ↓
Text-based QA Filtering (qa_filter_text.py)
    ↓
VLM-based QA Filtering (qa_filter_vl.py)
    ↓
iMCoTT Generation (imcott_generate.py)
    ↓
Training Data
```

## Evaluation Results

Please refer to our [paper](https://arxiv.org/abs/2511.16334) for detailed evaluation results and analysis.

## Citation

If you find this project helpful, please consider citing our paper with:

```bibtex
@article{yang2025longvt,
    title={LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling},
    author={Yang, Zuhao and Wang, Sudong and Zhang, Kaichen and Wu, Keming and Leng, Sicong and Zhang, Yifan and Li, Bo and Qin, Chengwei and Lu, Shijian and Li, Xingxuan and Bing, Lidong},
    journal={arXiv preprint arXiv:2511.20785},
    year={2025}
}
```

## Acknowledgements

We gratefully acknowledge the following open-source projects that made this work possible:

- [**lmms-engine**](https://github.com/EvolvingLMMs-Lab/lmms-engine) for the SFT/RFT training infrastructure and tools.
- [**verl**](https://github.com/volcengine/verl) for the RL training framework.
- [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval) for providing the comprehensive evaluation framework for large multimodal models.

We thank the developers and contributors of these projects for their excellent work and for making their code publicly available.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EvolvingLMMs-Lab/LongVT&type=Date)](https://github.com/EvolvingLMMs-Lab/LongVT&Date)
