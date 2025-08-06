#!/usr/bin/env python
# coding=utf-8
"""
Translation script supporting HF Transformers, GGUF (llama-cpp-python), and vLLM.
Uses --max_tokens for both context window and generation max tokens, to unify between backends.
Supports resuming: skips already translated samples if present in the output file.
Prints peak GPU memory usage at the end (if CUDA is available).
For GGUF/llama-cpp-python mode, also prints current GPU memory usage using nvidia-smi.
"""

import argparse
import json
import logging
from tqdm import tqdm
import os
import sys
import traceback
import torch
import subprocess

# Backend availability flags
TRANSFORMERS_AVAILABLE = False
LLAMACPP_AVAILABLE = False
VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    pass
try:
    from vllm import LLM as VLLMModel, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ===== HF Transformers =====
def load_transformers_model(model_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_transformers(prompt, tokenizer, model, max_tokens=2048, temperature=0.2, device=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]
    max_length = min(input_len + max_tokens,
                     getattr(model.config, "max_position_embeddings", 4096))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()

# ===== llama-cpp-python (GGUF) =====
def load_gguf_model(gguf_path, n_gpu_layers=32, n_ctx=2048):
    llm = Llama(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
    return llm


def generate_gguf(prompt, llm, max_tokens=2048, temperature=0.2):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"],
        echo=False
    )
    text = output["choices"][0]["text"]
    return text.strip()

# ===== vLLM =====
def load_vllm_model(model_name_or_path):
    # Launch vLLM for inference
    llm = VLLMModel(model=model_name_or_path)
    return llm


def generate_vllm(prompt, llm, max_tokens=2048, temperature=0.2):
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                            stop_sequences=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"])
    # vLLM returns a StreamingResponse; iterate to get the first result
    for res in llm.generate([prompt], sampling_params=params):
        # Each res corresponds to one prompt
        if res.outputs:
            return res.outputs[0].text.strip()
    return ""

# ===== Utilities =====
def load_existing_predictions(output_file):
    existing = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    data = json.loads(line)
                    if "source" in data and data.get("prediction"):
                        existing.add(data["source"])
                except Exception:
                    continue
    return existing


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"\n[GPU MEMORY] Peak allocated: {max_alloc:.2f} GB | Peak reserved: {max_reserved:.2f} GB")


def print_llamacpp_gpu_usage():
    try:
        pid = os.getpid()
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        found = False
        for line in smi_out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if str(pid) == parts[0]:
                print(f"\n[llama-cpp-python GPU USAGE] pid={pid} | used_memory={parts[1]} MiB | gpu={parts[2]}")
                found = True
        if not found:
            print("\n[llama-cpp-python GPU USAGE] This process not in nvidia-smi.")
    except Exception as e:
        print(f"\n[llama-cpp-python GPU USAGE] Could not obtain GPU usage: {e}")

# ===== Main =====
def main():
    parser = argparse.ArgumentParser(
        description="Translate using HF Transformers, GGUF, or vLLM; supports resuming."
    )
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--model_path", default=None, type=str,
                        help="HF Transformers model repo or path (for Transformers)")
    parser.add_argument("--gguf_path", default=None, type=str,
                        help="GGUF file path (for llama.cpp)")
    parser.add_argument("--vllm_path", default=None, type=str,
                        help="vLLM model repo or path (for vLLM)" )
    parser.add_argument("--max_tokens", default=8192, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--device", default=None, type=str,
                        help="Device for HF transformers (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--n_gpu_layers", default=64, type=int,
                        help="GPU layers for llama-cpp (GGUF only)")
    args = parser.parse_args()

    # Choose backend
    if args.vllm_path:
        if not VLLM_AVAILABLE:
            logger.error("vLLM is not installed. Please install with: pip install vllm")
            sys.exit(1)
        llm = load_vllm_model(args.vllm_path)
        tokenizer, model = None, None
        backend = "vllm"
    elif args.gguf_path:
        if not LLAMACPP_AVAILABLE:
            logger.error("llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
            sys.exit(1)
        llm = load_gguf_model(args.gguf_path, args.n_gpu_layers, args.max_tokens)
        tokenizer, model = None, None
        backend = "gguf"
    else:
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers is not installed. Install with: pip install transformers")
            sys.exit(1)
        model_path = args.model_path or "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        tokenizer, model = load_transformers_model(model_path, args.device)
        llm = None
        backend = "transformers"

    logger.info(f"Using backend: {backend}")
    existing = load_existing_predictions(args.output_file)

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "a", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Translating"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line.")
                continue
            prompt = data.get("source", "")
            if not prompt or prompt in existing:
                continue
            try:
                if backend == "gguf":
                    translation = generate_gguf(prompt, llm, args.max_tokens, args.temperature)
                elif backend == "vllm":
                    translation = generate_vllm(prompt, llm, args.max_tokens, args.temperature)
                else:
                    translation = generate_transformers(
                        prompt, tokenizer, model,
                        args.max_tokens, args.temperature, args.device
                    )
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                traceback.print_exc()
                translation = ""
            data["prediction"] = translation
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()

    logger.info(f"Translation results have been written to {os.path.abspath(args.output_file)}.")
    print_gpu_memory_usage()
    if backend == "gguf":
        print_llamacpp_gpu_usage()


if __name__ == "__main__":
    main()
