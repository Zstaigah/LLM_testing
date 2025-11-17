# Cybersecurity LLM - Red Team & Blue Team Expert

Fine-tuned LLM with knowledge of offensive (red team) and defensive (blue team) cybersecurity operations.

## Features

- **Red Team**: Penetration testing, exploit techniques, MITRE ATT&CK tactics
- **Blue Team**: Threat detection, incident response, log analysis, SIEM rules
- **Efficient Training**: QLoRA 4-bit quantization for low VRAM usage
- **Compact Dataset**: 30 training + 10 eval examples (token-optimized)

## Requirements

- NVIDIA GPU with 16GB+ VRAM (24GB recommended)
- Python 3.10+
- CUDA 11.8+

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Base Model

Edit `train.py` and set `MODEL_NAME` to one of:
- `meta-llama/Llama-2-7b-hf` (requires HuggingFace approval)
- `mistralai/Mistral-7B-v0.1`
- `NousResearch/Llama-2-7b-hf` (ungated alternative)

Login to HuggingFace:
```bash
huggingface-cli login
```

### 3. Train Model

```bash
python train.py
```

Training time: ~2-3 hours on RTX 3090 (30 examples)

### 4. Test Model

```bash
python inference.py
```

## Dataset Structure

JSON format:
```json
{
  "instruction": "Task or question",
  "input": "Optional context",
  "output": "Expected response"
}
```

## Expanding Dataset

Add more examples to `dataset_train.json`:

**Red Team Topics**: Phishing, lateral movement, privilege escalation, C2, evasion, persistence, credential access, reconnaissance

**Blue Team Topics**: Log analysis, SIEM rules, incident response, threat hunting, forensics, detection engineering, IOC analysis

Keep outputs short (<200 tokens) for efficiency.

## Model Output

The model uses this format:

```
### Instruction:
<your question>

### Response:
<model answer>
```

## Memory Usage

- Base model load: ~4GB (4-bit quantized)
- Training: ~12GB VRAM
- Inference: ~6GB VRAM

## Limitations

- Small dataset (30 examples) - for demonstration
- Production use: expand to 1000+ examples
- Does NOT replace professional security training
- For educational/authorized testing only

## Security Notice

**IMPORTANT**: This model is for:
- Authorized penetration testing
- Security research
- Defensive security operations
- CTF competitions
- Educational purposes

**DO NOT** use for unauthorized access, malicious activity, or illegal purposes.

## License

Educational use only. Comply with all applicable laws and regulations.
