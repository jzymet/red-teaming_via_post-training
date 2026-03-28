# Red Teaming via Post-Training

This project implements a red-teaming framework using reinforcement learning to train an attacker model to generate harmful prompts that can jailbreak target AI models.

## Setup

### Environment
- Python 3.10+
- PyTorch with CUDA support
- Required packages: see requirements.txt

### Dependencies
Install with:
```bash
pip install -r requirements.txt
```

### API Key
You need a Groq API key for the evaluator model:
1. Go to https://console.groq.com/
2. Sign up and get your API key
3. Replace `"YOUR_GROQ_API_KEY"` in `main.py` with your actual key

### Models
- Attacker: Qwen/Qwen2.5-1.5B (HuggingFace)
- Target: meta-llama/Llama-3.2-3B-Instruct (HuggingFace)
- Evaluator: llama3-8b-8192 (via Groq API)

## Running the Project

This code is designed for distributed training with multiple GPUs using PyTorch FSDP.

### Requirements
- Multiple CUDA GPUs (tested with 2 GPUs)
- NCCL backend for distributed training

### Execution
```bash
python main.py
```

The script will automatically detect the number of CUDA devices and spawn processes accordingly.

### Output
- Rollouts logged to `data/rollouts.jsonl`
- Metrics printed every 5 rounds
- Final plot of training collapse

### Cloud Deployment
This project is optimized for cloud computing environments with GPU support (e.g., AWS, GCP, Azure). Ensure your instance has:
- Multiple GPUs
- Sufficient VRAM (at least 8GB per GPU recommended)
- Fast interconnect for NCCL (e.g., NVLink)