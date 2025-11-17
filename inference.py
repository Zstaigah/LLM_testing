#!/usr/bin/env python3
"""
Inference script for cybersecurity LLM
Token-optimized, fast inference
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Config
MODEL_DIR = "./cybersec-model"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # Must match training

def load_model():
    """Load fine-tuned model"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Load base + LoRA weights
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model.eval()

    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text=""):
    """Generate response for instruction"""
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    return response

def main():
    model, tokenizer = load_model()

    print("\n=== Cybersecurity LLM - Interactive Mode ===")
    print("Type 'exit' to quit\n")

    # Example queries
    examples = [
        ("What is Pass-the-Hash?", ""),
        ("Analyze this login event", "EventID 4625 × 20 from IP 1.2.3.4, Account: admin"),
        ("Explain lateral movement", "")
    ]

    print("Example queries:")
    for i, (inst, inp) in enumerate(examples, 1):
        print(f"{i}. {inst}" + (f" [Input: {inp}]" if inp else ""))
    print()

    while True:
        try:
            instruction = input("Instruction: ").strip()
            if instruction.lower() == 'exit':
                break

            input_text = input("Input (optional, press Enter to skip): ").strip()

            print("\nGenerating...\n")
            response = generate_response(model, tokenizer, instruction, input_text)
            print(f"Response:\n{response}\n")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break

    print("Goodbye!")

if __name__ == "__main__":
    main()
