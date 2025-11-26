
# LongVT: Incentivizing “Thinking with Long Videos” via Native Tool Calling

<div align="center">

[![Models](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/OpenMMReasoner/OpenMMReasoner-RL)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/collections/lmms-lab/openmmreasoner)
[![Paper](https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.16334)
[![Project Page](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white)](https://evolvinglmms-lab.github.io/OpenMMReasoner/)
[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/EvolvingLMMs-Lab/OpenMMReasoner)
[![Static Badge](https://img.shields.io/badge/Blog-lmms_lab?style=for-the-badge)](https://www.lmms-lab.com/posts/openmmreasoner/)
</div>

## 🎉 News

- **[2025-11]**: Join our WeChat group by scanning this [QR code](assets/qr_code.jpg).
- **[2025-11]**: We release all of our code, model, data, and pipeline! Check out the [OpenMMReasoner collection on Hugging Face](https://huggingface.co/collections/lmms-lab/openmmreasoner).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [SFT Training](#1-sft-training)
  - [RL Training](#2-rl-training)
  - [Evaluation](#3-evaluation)
  - [Data Pipeline](#4-data-pipeline)
- [Getting Started](#getting-started)
  - [Data Preparation](#data-preparation)
  - [SFT Training](#sft-training)
  - [RL Training](#rl-training)
  - [Evaluation](#evaluation)
  - [LLM Judge Setup](#llm-judge-setup)
  - [Data Processing Pipeline](#data-processing-pipeline)
- [Evaluation Results](#evaluation-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

<div align="center">
  <img src="assets/teaser.png" alt="teaser" width="1000"/>
</div>

Large multimodal models (LMMs) have shown great potential for video reasoning with textual Chain-of-Thought.
However, they remain vulnerable to hallucination, especially when processing long-form videos where evidence is sparse and temporally dispersed.
Inspired by how humans comprehend long videos-by first skimming globally and then examining relevant clips for details-we introduce **LongVT**, an end-to-end agentic framework that enables ``Thinking with **Long** **V**ideos'' via interleaved Multimodal Chain-of-**T**ool-Thought.
Specifically, we exploit LMMs' inherent temporal grounding ability as a native video cropping tool to zoom in on a specific video clip and resample finer-grained video frames.

This global-to-local reasoning loop continues until answers are grounded in retrieved visual evidence.
Given the scarcity of fine-grained question-answering (QA) data for the long video reasoning task, we curate and will release a data suite named **VideoSIAH** to facilitate both training and evaluation.
Specifically, our training dataset consists of 247.9K samples for tool-integrated cold-start supervised fine-tuning, 1.6K samples for agentic reinforcement learning, and 15.4K samples for agentic reinforcement fine-tuning, respectively. 
Our evaluation benchmark consists of 1,280 QA pairs that are carefully curated through a semi-automatic data pipeline with human-in-the-loop validation.
With a meticulously designed three-stage training strategy and extensive empirical validation, LongVT consistently outperforms existing strong baselines across four challenging long-video understanding and reasoning benchmarks.


## Installation

### 1. SFT Training

Please follow the installation instructions in [lmms-engine](https://github.com/EvolvingLMMs-Lab/lmms-engine) to prepare the environment for supervised fine-tuning.

### 2. RL Training

We provide our source implementation of `verl`, which is a detached fork from the original [verl](https://github.com/volcengine/verl) repository. You may choose to use either our integrated version or the original `verl` library for RL training. However, for seamless reproduction, we highly recommend using our provided environment.

#### Installation

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

### 3. Evaluation

Please follow the installation instructions in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to set up the evaluation environment.

### 4. Data Pipeline
We open-sourced our data processing pipeline and code for the community to follow. To install requirements for Data Pipeline:

```bash
cd ./data_pipeline

uv pip install -e .
```

We recommend you to use separate environments if you encounter a conflict in requirements.

## Getting Started

### Data Preparation

We provide a convenient script to download all the required datasets from Hugging Face:

```bash
bash examples/openmmreasoner/download_data.sh [LOCAL_DIR]
```

This script will download both the SFT (874K samples) and RL (74K samples) datasets to your specified directory (defaults to `./data`).

### SFT Training

After installing [lmms-engine](https://github.com/EvolvingLMMs-Lab/lmms-engine), you can launch SFT training using either:

**Option 1: Using a configuration YAML file**

```bash
# Edit the dataset paths in sft_example_config.yaml
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli config_yaml=${CONFIG}
```

**Option 2: Using the launch script**

```bash
# Edit the dataset paths and hyperparameters in the script
bash examples/openmmreasoner/sft_example_launch.sh
```

**Troubleshooting:**
- If you encounter **OOM (Out of Memory)** errors, reduce the `packing_length` parameter in your configuration.
- If mixing text and image data causes a **hang**, consider adding a blank dummy image for text-only samples in the m1 dataset.

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

### Evaluation

After setting up [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), use the provided evaluation script:

```bash
bash examples/openmmreasoner/eval.sh <CHECKPOINT_PATH> <TASK_NAME>
```

**Image Tasks:**

```bash
bash examples/openmmreasoner/eval.sh /path/to/checkpoint "mmmu_reasoning_reward,wemath_testmini_thinking,mmmu_pro_vision_cot_reward,mmmu_pro_standard_cot_reward,mathvista_testmini_cot_reward,mathvision_reason_testmini_reward,mathvision_reason_test_reward,mathverse_testmini_reward,logicvista_thinking,dynamath,charxiv_val_descriptive_cot,charxiv_val_reasoning_cot"
```

**Text Tasks:**

```bash
bash examples/openmmreasoner/eval.sh /path/to/checkpoint "gpqa_diamond_thinking,aime_agg8"
```

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


### Data Processing Pipeline

To follow our data processing pipeline, we provide example scripts in `data_pipeline/examples/`. The pipeline supports two main operations:

#### Deduplicating RL Data

To deduplicate RL training data, follow these steps:

1. **Prepare the RL configuration**: Create a YAML config file based on `data_pipeline/examples/example_rl_config.yaml`:

```yaml
datasets:
  - path: /path/to/your/dataset.parquet
    data_folder: "/path/to/images"
    data_type: parquet
```

2. **Run embedding**: Generate embeddings for the dataset:

```bash
cd data_pipeline
bash examples/embed_data.sh /path/to/your_rl_config.yaml cache/embed rl
```

3. **Run deduplication**: Remove duplicates based on embeddings:

```bash
bash examples/deduplicate_data.sh /path/to/your_rl_config.yaml cache/embed rl cache/deduplicate
```

#### Distilling Dataset

To distill a dataset using a teacher model:

1. **Prepare the SFT configuration**: Create a YAML config file based on `data_pipeline/examples/example_sft_config.yaml`:

```yaml
datasets:
  - path: /path/to/your/dataset.parquet
    data_folder: "/path/to/images"
    data_type: parquet
```

2. **Run distillation**: Edit `data_pipeline/examples/distill_dataset.sh` to set your server addresses, then run:

```bash
cd data_pipeline
bash examples/distill_dataset.sh
```

Make sure to configure the model server and judge server URLs in the script before running.

## Evaluation Results

Our **OpenMMReasoner-7B (OMR-7B)** model demonstrates strong performance across a comprehensive suite of multimodal reasoning benchmarks. With only 874K SFT samples and 74K RL samples—significantly less data than many competing methods—our model achieves state-of-the-art or highly competitive results on 9 out of 14 benchmark tasks. Notably, OMR-7B achieves **79.5%** on MathVista testmini (best among all models), **63.8%** on MathVerse testmini (best), and **79.0%** on WeMath loose (best), demonstrating the effectiveness of our transparent two-stage training recipe. This performance validates our emphasis on data quality and rigorous training design over simply scaling dataset size.

| Model | SFT Data | RL Data | MathVista<br/>testmini | MathVision<br/>test | MathVision<br/>testmini | MathVerse<br/>testmini | DynaMath<br/>worst | WeMath<br/>loose | LogicVista<br/>test | MMMU<br/>val | MMMU-Pro<br/>standard | MMMU-Pro<br/>vision | CharXiv<br/>reas. | CharXiv<br/>desc. |
|-------|----------|---------|------------------------|---------------------|-------------------------|------------------------|--------------------|--------------------|---------------------|--------------|-----------------------|---------------------|-------------------|-------------------|
| VLAA-Thinker-Qwen2.5-7B | 126k | 25k | 68.0 | 26.4 | - | 48.2 | 22.4 | - | 48.5 | - | - | - | - | - |
| ThinkLite-7B-VL | - | 11k | 71.6 | 24.6 | - | 42.9 | 16.5 | - | 42.7 | - | - | - | - | - |
| VL-Rethinker-7B | - | 39k | 73.7 | 28.4 | - | 46.4 | 17.8 | - | 42.7 | - | 41.7 | - | - | - |
| M2-Reasoning | 6.2M | 102k | 75.0 | 42.1 | - | 40.4 | - | - | 50.6 | - | - | - | - | - |
| MMR1 | 1.6M | 15k | 72.0 | 31.8 | 29.0† | 55.4 | 27.9† | 68.0† | 48.9 | 52.4† | 41.1† | 37.1† | 43.5† | 71.1† |
| OpenVLThinker-7B | 3.3k | 9.6k | 65.3 | 23.0 | 26.9† | 38.1 | 16.8 | 61.9† | 44.5 | 55.1† | 39.7† | 38.4† | 41.0† | 69.2† |
| MM-Eureka-Qwen-7B | - | 15.6k | 72.6 | 28.1 | 32.1† | 45.4 | 23.0 | 59.8† | 46.3 | 54.4† | 40.1† | 37.1† | 42.4† | 74.1† |
| OVR-7B | 2M | 300k | 72.1 | **51.8** | 38.2† | 54.6 | 33.5 | 64.8 | **54.8** | 51.8† | **50.2** | 29.1† | 44.5 | 73.6 |
| **OMR-7B (ours)** | **874k** | **74k** | **79.5** | 43.6 | **38.8** | **63.8** | **34.9** | **79.0** | 50.0 | **57.8** | 44.1 | **40.6** | **46.1** | 73.5 |

**Note:** Bold numbers indicate the best performance, and † indicates results reproduced using the authors' checkpoints.

## Citation

If you find OpenMMReasoner useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{zhang2025openmmreasonerpushingfrontiersmultimodal,
      title={OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe}, 
      author={Kaichen Zhang and Keming Wu and Zuhao Yang and Kairui Hu and Bin Wang and Ziwei Liu and Xingxuan Li and Lidong Bing},
      year={2025},
      eprint={2511.16334},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.16334}, 
}
```

## Acknowledgements

We gratefully acknowledge the following open-source projects that made this work possible:

- [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval) for providing the comprehensive evaluation framework for large multimodal models.
- [**lmms-engine**](https://github.com/EvolvingLMMs-Lab/lmms-engine) for the SFT training infrastructure and tools.
- [**verl**](https://github.com/volcengine/verl) for the reinforcement learning training framework.

We thank the developers and contributors of these projects for their excellent work and for making their code publicly available.

