#!/usr/bin/env python
# coding=utf-8
"""
Translation script supporting HF Transformers, GGUF (llama-cpp-python), and vLLM.
Batched generation for all backends (true batching for Transformers/vLLM; grouped loop for llama-cpp).
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
    # Ensure pad_token is set (some causal models do not define it)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_transformers_batch(prompts, tokenizer, model, max_tokens=2048, temperature=0.2, device=None):
    """
    True batched generation with Transformers.
    Uses max_new_tokens to keep semantics stable across variable-length prompts.
    """
    if len(prompts) == 0:
        return []

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    if device:
        enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        **enc,
        do_sample=(temperature is not None and temperature > 0.0),
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # Decode per item and attempt to remove the original prompt prefix
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for i, full in enumerate(decoded):
        prefix = prompts[i]
        if full.startswith(prefix):
            gen = full[len(prefix):]
        else:
            # Fallback: leave as-is if exact prefix matching fails
            gen = full
        results.append(gen.strip())
    return results

# ===== llama-cpp-python (GGUF) =====
def load_gguf_model(gguf_path, n_gpu_layers=32, n_ctx=2048):
    llm = Llama(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
    return llm


def generate_gguf_batch(prompts, llm, max_tokens=2048, temperature=0.2):
    """
    llama-cpp-python currently lacks a native list-batch API in common versions.
    Iterate within the batch and return outputs aligned with the input order.
    """
    results = []
    for p in prompts:
        out = llm(
            p,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"],
            echo=False
        )
        text = out["choices"][0]["text"].strip()
        results.append(text)
    return results

# ===== vLLM =====
def load_vllm_model(model_name_or_path):
    # Launch vLLM for inference
    llm = VLLMModel(model=model_name_or_path)
    return llm


def generate_vllm_batch(prompts, llm, max_tokens=2048, temperature=0.2):
    """
    True batched generation with vLLM by passing a list of prompts.
    """
    if len(prompts) == 0:
        return []
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"],
    )
    # vLLM returns a list of RequestOutput aligned with the input order
    outputs = llm.generate(prompts, sampling_params=params, n=8)
    results = []
    for res in outputs:
        if res.outputs:
            results.append(res.outputs[0].text.strip())
        else:
            results.append("")
    return results

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

def chunk_iterable(it, size):
    """Yield lists of up to `size` items from iterable `it` while preserving order."""
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

# ===== Main =====
def main():
    parser = argparse.ArgumentParser(
        description="Translate using HF Transformers, GGUF, or vLLM; supports resuming and batched inference."
    )
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--model_path", default=None, type=str,
                        help="HF Transformers model repo or path (for Transformers)")
    parser.add_argument("--gguf_path", default=None, type=str,
                        help="GGUF file path (for llama.cpp)")
    parser.add_argument("--vllm_path", default=None, type=str,
                        help="vLLM model repo or path (for vLLM)")
    parser.add_argument("--max_tokens", default=8192, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--device", default=None, type=str,
                        help="Device for HF transformers (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--n_gpu_layers", default=64, type=int,
                        help="GPU layers for llama-cpp (GGUF only)")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size for generation")
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

    # Read all lines to enable clean batching while preserving order
    with open(args.input_file, "r", encoding="utf-8") as fin:
        raw_lines = [ln for ln in fin if ln.strip()]

    # Pre-filter (skip invalid JSON or already predicted) and keep parallel arrays
    records = []
    for line in raw_lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON line.")
            continue
        prompt = data.get("source", "")
        if not prompt or prompt in existing:
            continue
        records.append((prompt, data))

    total = len(records)
    logger.info(f"Total remaining samples to translate: {total}")

    # Open output and process in batches
    with open(args.output_file, "a", encoding="utf-8") as fout:
        for batch in tqdm(list(chunk_iterable(records, args.batch_size)), desc="Translating (batched)"):
            prompts = [p for p, _ in batch]
            try:
                if backend == "gguf":
                    gens = generate_gguf_batch(prompts, llm, args.max_tokens, args.temperature)
                elif backend == "vllm":
                    gens = generate_vllm_batch(prompts, llm, args.max_tokens, args.temperature)
                else:
                    gens = generate_transformers_batch(prompts, tokenizer, model,
                                                       args.max_tokens, args.temperature, args.device)
            except Exception as e:
                logger.error(f"Error during batch generation: {e}")
                traceback.print_exc()
                gens = [""] * len(prompts)

            # Write outputs in the same order
            for (_, data), pred in zip(batch, gens):
                data["prediction"] = pred
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()

    logger.info(f"Translation results have been written to {os.path.abspath(args.output_file)}.")
    print_gpu_memory_usage()
    if backend == "gguf":
        print_llamacpp_gpu_usage()


if __name__ == "__main__":
    main()
