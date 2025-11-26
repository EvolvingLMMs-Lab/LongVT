
# LongVT: Incentivizing “Thinking with Long Videos” via Native Tool Calling

## 🎉 News

- **[2025-11]**: Join our WeChat group by scanning this [QR code](assets/qr_code.jpg).

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

We recommend you to use separate environments if you encounter a conflict in requirements.

## Getting Started

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


## Acknowledgements

We gratefully acknowledge the following open-source projects that made this work possible:

- [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval) for providing the comprehensive evaluation framework for large multimodal models.
- [**lmms-engine**](https://github.com/EvolvingLMMs-Lab/lmms-engine) for the SFT training infrastructure and tools.
- [**verl**](https://github.com/volcengine/verl) for the reinforcement learning training framework.

We thank the developers and contributors of these projects for their excellent work and for making their code publicly available.

