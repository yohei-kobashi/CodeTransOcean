#!/usr/bin/env python
# coding=utf-8
"""
Translation script supporting HF Transformers, GGUF (llama-cpp-python), and vLLM.
Batched generation for all backends (true batching for Transformers/vLLM; grouped loop for llama-cpp).
Robust resume: trims a partial line at the end of the output (if any), loads completed keys, skips processed items,
and fsyncs periodically to minimize data loss on crashes.
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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        AutoModelForMultimodalLM = None
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

CODEGEEX_SYSTEM_PROMPT = (
    "You are an expert software engineer proficient in a wide range of programming languages."
)

# =========================
# Resume helpers
# =========================
def get_record_key(d: dict, key_field: str | None) -> str | None:
    """Return a stable key for a record: prefer user-specified key_field, else id -> task_id -> source."""
    if key_field and key_field in d:
        return str(d[key_field])
    for k in ("id", "task_id", "source"):
        if k in d and d[k] is not None:
            return str(d[k])
    return None

def repair_trailing_partial_line(path: str, tail_bytes: int = 1 << 16) -> None:
    """If the output file ends with a partial (non-newline-terminated) JSON line, truncate it safely."""
    if not os.path.exists(path):
        return
    data = list(open(path).readlines())
    try:
        json.loads(data[-1])
    except:
        open(path, "w").write("\n".join(data[:-1]))
        logger.warning("Output had a partial last line; truncating file to last complete newline.")

def load_existing_keys(output_file: str, key_field: str | None, require_prediction_field: bool) -> set[str]:
    """
    Load keys that have already been written to output.
    If require_prediction_field=True, only count lines that contain a 'prediction' field.
    """
    existing: set[str] = set()
    if not os.path.exists(output_file):
        return existing
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                # Ignore malformed (shouldn't exist after repair), treat as not processed
                continue
            if require_prediction_field and "prediction" not in d:
                continue
            key = get_record_key(d, key_field)
            if key is not None:
                existing.add(key)
    return existing

# =========================
# HF Transformers
# =========================
def load_transformers_model(model_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # Ensure pad_token is set (some causal models do not define it)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation must be left padded. Right padding can make
    # generation continue from a pad token for shorter prompts.
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available() and device != "cpu"
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
        "device_map": device or ("auto" if use_cuda else None),
        "trust_remote_code": True,
    }
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        if model_type == "qwen3_5":
            model_cls = AutoModelForMultimodalLM or AutoModelForImageTextToText
            if model_cls is None:
                raise ValueError("qwen3_5 is not supported by this Transformers version")
        else:
            model_cls = AutoModelForCausalLM
        model = model_cls.from_pretrained(model_name_or_path, **load_kwargs)
    except (KeyError, ValueError) as exc:
        if "qwen3_5" in str(exc).lower() or "qwen3.5" in model_name_or_path.lower():
            raise RuntimeError(
                "This Transformers installation does not support Qwen3.5. "
                "Install the latest Transformers from its main branch as documented "
                "by Qwen (see requirements-qwen35.txt)."
            ) from exc
        raise
    model.eval()
    return tokenizer, model


def format_chat_prompts(prompts, tokenizer, enable_thinking=False, use_chat_template=True,
                        system_prompt=CODEGEEX_SYSTEM_PROMPT):
    """Format plain user prompts for instruct/chat models."""
    if not use_chat_template:
        return prompts
    if not getattr(tokenizer, "chat_template", None):
        logger.warning("Tokenizer has no chat template; using raw prompts.")
        return prompts

    conversations = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Recent tokenizers render a list of conversations in one call, avoiding
    # Python/Jinja call overhead for large offline batches.
    try:
        rendered = tokenizer.apply_chat_template(
            conversations,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        if isinstance(rendered, list) and len(rendered) == len(prompts):
            return rendered
    except (TypeError, ValueError):
        pass

    formatted = []
    for messages in conversations:
        # Qwen3/Qwen3.5 accept enable_thinking. Retry without it for strict
        # templates belonging to other model families.
        try:
            text = tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                **kwargs,
            )
        formatted.append(text)
    return formatted


def generate_transformers_batch(prompts, tokenizer, model, max_tokens=2048, temperature=0.2,
                                device=None, enable_thinking=False, use_chat_template=True,
                                system_prompt=CODEGEEX_SYSTEM_PROMPT, top_p=1.0, top_k=0,
                                min_p=0.0, repetition_penalty=1.0, seed=42):
    """
    True batched generation with Transformers.
    Uses max_new_tokens to keep semantics stable across variable-length prompts.
    """
    if len(prompts) == 0:
        return []

    model_prompts = format_chat_prompts(
        prompts, tokenizer, enable_thinking=enable_thinking,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
    enc = tokenizer(
        model_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_device = device or model.device
    enc = {k: v.to(input_device) for k, v in enc.items()}

    gen_kwargs = dict(
        **enc,
        do_sample=(temperature is not None and temperature > 0.0),
        max_new_tokens=max_tokens,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    if gen_kwargs["do_sample"]:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # generate() returns prompt + completion. Slice by the padded token width;
    # string-prefix removal is unreliable after chat templating/tokenization.
    generated_ids = outputs[:, enc["input_ids"].shape[1]:]
    return [
        text.strip()
        for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ]

# =========================
# llama-cpp-python (GGUF)
# =========================
def load_gguf_model(gguf_path, n_gpu_layers=32, n_ctx=2048):
    llm = Llama(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
    return llm


def generate_gguf_batch(prompts, llm, max_tokens=2048, temperature=0.2,
                        top_p=1.0, top_k=0, repetition_penalty=1.0, seed=42):
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
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            seed=seed if seed is not None else -1,
            stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"],
            echo=False
        )
        text = out["choices"][0]["text"].strip()
        results.append(text)
    return results

# =========================
# vLLM
# =========================
def load_vllm_model(model_name_or_path, max_model_len=32768,
                    gpu_memory_utilization=0.95, kv_cache_dtype="auto",
                    max_num_batched_tokens=32768, max_num_seqs=512,
                    tensor_parallel_size=1, enforce_eager=False,
                    enable_prefix_caching=False, language_model_only=False,
                    mtp_tokens=0):
    # Launch vLLM for inference
    kwargs = dict(
        model=model_name_or_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=True,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
    )
    if kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if language_model_only:
        kwargs["language_model_only"] = True
    if mtp_tokens:
        kwargs["speculative_config"] = {
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": mtp_tokens,
        }
    logger.info("vLLM engine arguments: %s", kwargs)
    try:
        llm = VLLMModel(**kwargs)
    except (KeyError, ValueError, RuntimeError) as exc:
        if "qwen3_5" in str(exc).lower() or "qwen3.5" in model_name_or_path.lower():
            raise RuntimeError(
                "Failed to load Qwen3.5 with vLLM. Qwen3.5 requires a recent "
                "vLLM nightly build; see requirements-qwen35.txt."
            ) from exc
        raise
    return llm


def generate_vllm_batch(prompts, llm, max_tokens=2048, temperature=0.2,
                        enable_thinking=False, use_chat_template=True,
                        system_prompt=CODEGEEX_SYSTEM_PROMPT, top_p=1.0, top_k=0,
                        min_p=0.0, presence_penalty=0.0,
                        repetition_penalty=1.0, seed=42):
    """
    True batched generation with vLLM by passing a list of prompts.
    """
    if len(prompts) == 0:
        return []
    tokenizer = llm.get_tokenizer()
    model_prompts = format_chat_prompts(
        prompts, tokenizer, enable_thinking=enable_thinking,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"]
    )
    # vLLM returns a list of RequestOutput aligned with the input order
    outputs = llm.generate(model_prompts, sampling_params=params)
    results = []
    for res in outputs:
        if res.outputs:
            results.append(res.outputs[0].text.strip())
        else:
            results.append("")
    return results

# =========================
# Utilities
# =========================
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

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Translate using HF Transformers, GGUF, or vLLM; supports robust resume and batched inference."
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
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=0, type=int)
    parser.add_argument("--min_p", default=0.0, type=float)
    parser.add_argument("--presence_penalty", default=0.0, type=float,
                        help="vLLM sampling penalty; ignored by Transformers/GGUF.")
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int,
                        help="Per-request sampling seed (default: 42).")
    parser.add_argument("--device", default=None, type=str,
                        help="Device for HF transformers (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--n_gpu_layers", default=64, type=int,
                        help="GPU layers for llama-cpp (GGUF only)")
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size for generation")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen thinking mode (disabled by default for direct translations).")
    parser.add_argument("--raw_prompt", action="store_true",
                        help="Do not apply the tokenizer's chat template.")
    parser.add_argument("--no_system_prompt", action="store_true",
                        help="Omit the CodeGeeX-compatible system message from the chat template.")
    parser.add_argument("--max_model_len", default=32768, type=int,
                        help="vLLM context length. A smaller value reduces KV-cache memory.")
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float,
                        help="Fraction of GPU memory available to vLLM.")
    parser.add_argument("--kv_cache_dtype", default="auto", choices=("auto", "fp8", "fp8_e4m3"),
                        help="vLLM KV-cache dtype. 'auto' is portable; FP8 requires supported hardware.")
    parser.add_argument("--max_num_batched_tokens", default=32768, type=int,
                        help="Maximum tokens scheduled per vLLM iteration; increase for throughput.")
    parser.add_argument("--max_num_seqs", default=512, type=int,
                        help="Maximum sequences scheduled concurrently by vLLM.")
    parser.add_argument("--tensor_parallel_size", default=1, type=int,
                        help="Number of GPUs used for tensor parallelism.")
    parser.add_argument("--enforce_eager", "--enforce-eager", action="store_true",
                        help="Disable CUDA graphs. Slower; use only for compatibility/debugging.")
    parser.add_argument("--enable_prefix_caching", action="store_true",
                        help="Enable vLLM automatic prefix caching.")
    parser.add_argument("--language_model_only", action="store_true",
                        help="Skip Qwen3.5 vision encoder loading/profiling for text-only translation.")
    parser.add_argument("--mtp_tokens", default=0, type=int,
                        help="Qwen3.5 MTP speculative tokens (0 disables speculative decoding).")
    # Resume-related options
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from the existing output by skipping already processed records.")
    parser.add_argument("--key_field", type=str, default=None,
                        help="Field name to use as a unique key (fallback: id -> task_id -> source).")
    parser.add_argument("--require_prediction_field", action="store_true", default=False,
                        help="Only treat a line as completed if it has a 'prediction' field.")
    parser.add_argument("--sync_interval", type=int, default=50,
                        help="fsync the output file every N written lines (0 to disable).")
    args = parser.parse_args()
    system_prompt = None if args.no_system_prompt else CODEGEEX_SYSTEM_PROMPT
    if args.batch_size <= 0 or args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        parser.error("batch_size, max_num_seqs, and max_num_batched_tokens must be positive.")
    if args.mtp_tokens < 0:
        parser.error("mtp_tokens must be non-negative.")
    if not 0.0 <= args.top_p <= 1.0 or not 0.0 <= args.min_p <= 1.0:
        parser.error("top_p and min_p must be between 0 and 1.")
    if not -2.0 <= args.presence_penalty <= 2.0:
        parser.error("presence_penalty must be between -2 and 2.")
    if args.repetition_penalty <= 0:
        parser.error("repetition_penalty must be positive.")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    # Choose backend
    if args.vllm_path:
        if not VLLM_AVAILABLE:
            logger.error("vLLM is not installed. Please install with: pip install vllm")
            sys.exit(1)
        llm = load_vllm_model(
            args.vllm_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=args.enable_prefix_caching,
            language_model_only=args.language_model_only,
            mtp_tokens=args.mtp_tokens,
        )
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

    # Repair a potentially partial trailing line first (for robust resume)
    if args.resume:
        repair_trailing_partial_line(args.output_file)

    # Load existing keys from the output (those are considered completed)
    existing = load_existing_keys(
        args.output_file,
        key_field=args.key_field,
        require_prediction_field=args.require_prediction_field,
    )
    logger.info(f"Found {len(existing)} completed records in existing output.")

    # Read all lines to enable clean batching while preserving order
    with open(args.input_file, "r", encoding="utf-8") as fin:
        raw_lines = [ln for ln in fin if ln.strip()]

    # Pre-filter (skip invalid JSON or already completed) and keep parallel arrays
    records = []
    for line in raw_lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON line.")
            continue
        key = get_record_key(data, args.key_field)
        if key is None:
            logger.warning("Skipping a record without a usable key (id/task_id/source missing).")
            continue
        if args.resume and key in existing:
            continue
        prompt = data.get("source", "")
        if not prompt:
            continue
        # Keep the key on the record so we can add it if needed to the output
        data["_key"] = key
        records.append((prompt, data))

    total = len(records)
    logger.info(f"Total remaining samples to translate: {total}")

    # Open output and process in batches
    written_since_sync = 0
    with open(args.output_file, "a", encoding="utf-8") as fout:
        batch_starts = range(0, total, args.batch_size)
        for start in tqdm(
            batch_starts,
            total=(total + args.batch_size - 1) // args.batch_size,
            desc="Translating (batched)",
        ):
            batch = records[start:start + args.batch_size]
            prompts = [p for p, _ in batch]
            try:
                if backend == "gguf":
                    gens = generate_gguf_batch(
                        prompts, llm, args.max_tokens, args.temperature,
                        args.top_p, args.top_k, args.repetition_penalty, args.seed,
                    )
                elif backend == "vllm":
                    gens = generate_vllm_batch(
                        prompts, llm, args.max_tokens, args.temperature,
                        enable_thinking=args.enable_thinking,
                        use_chat_template=not args.raw_prompt,
                        system_prompt=system_prompt,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        min_p=args.min_p,
                        presence_penalty=args.presence_penalty,
                        repetition_penalty=args.repetition_penalty,
                        seed=args.seed,
                    )
                else:
                    gens = generate_transformers_batch(
                        prompts, tokenizer, model, args.max_tokens, args.temperature,
                        args.device, enable_thinking=args.enable_thinking,
                        use_chat_template=not args.raw_prompt,
                        system_prompt=system_prompt,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        min_p=args.min_p,
                        repetition_penalty=args.repetition_penalty,
                        seed=args.seed,
                    )
            except Exception as e:
                logger.error(f"Error during batch generation: {e}")
                traceback.print_exc()
                gens = [""] * len(prompts)

            # Write outputs in the same order
            for (_, data), pred in zip(batch, gens):
                data["prediction"] = pred
                # Ensure the chosen key is present in the output (helps future resume)
                if "_key" in data and ("id" not in data and "task_id" not in data and "source" not in data):
                    data["id"] = data["_key"]
                data.pop("_key", None)

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                written_since_sync += 1

                if args.sync_interval and written_since_sync >= args.sync_interval:
                    fout.flush()
                    try:
                        os.fsync(fout.fileno())
                    except Exception:
                        pass
                    written_since_sync = 0

        # Final flush/fsync
        fout.flush()
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass

    logger.info(f"Translation results have been written to {os.path.abspath(args.output_file)}.")
    print_gpu_memory_usage()
    if backend == "gguf":
        print_llamacpp_gpu_usage()


if __name__ == "__main__":
    main()
