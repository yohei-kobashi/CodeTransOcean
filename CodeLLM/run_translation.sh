#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ngpus=1
#PBS -l walltime=36:00:00
#PBS -P gcb50389

set -euo pipefail

module purge
module load cuda/12.8 python/3.12

# Some PBS environments expose the allocated GPU as a UUID. Older vLLM
# versions expect a numeric CUDA_VISIBLE_DEVICES entry.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
  allocated_gpu_uuid="${CUDA_VISIBLE_DEVICES%%,*}"
  allocated_gpu_index="$(
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
      awk -F, -v uuid="$allocated_gpu_uuid" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1)
          gsub(/^[ \t]+|[ \t]+$/, "", $2)
          if ($2 == uuid) {
            print $1
            exit
          }
        }
      '
  )"
  if [[ -z "$allocated_gpu_index" ]]; then
    echo "Could not map allocated GPU UUID: $allocated_gpu_uuid" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$allocated_gpu_index"
fi

source env_vllm/bin/activate
cd CodeTransOcean/CodeLLM

export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Submit once per model, for example:
#   qsub -v MODEL_SIZE=4B run_translation.sh
#   qsub -v MODEL_SIZE=9B run_translation.sh
model_size="${MODEL_SIZE:-4B}"
case "$model_size" in
  4B|9B) ;;
  *)
    echo "MODEL_SIZE must be 4B or 9B (received: $model_size)" >&2
    exit 2
    ;;
esac

model_name="Qwen/Qwen3.5-${model_size}"
input_file="data/preprocessed_niche_test_codegeex.json"

# Qwen3.5 official sampling presets. THINKING=1 uses the precise-coding
# thinking preset; THINKING=0 uses the non-thinking instruct preset.
thinking="${THINKING:-0}"
seed="${SEED:-42}"
case "$thinking" in
  1|true|TRUE|yes|YES)
    mode_name="thinking"
    temperature="0.6"
    top_p="0.95"
    top_k="20"
    min_p="0.0"
    presence_penalty="0.0"
    repetition_penalty="1.0"
    thinking_args=(--enable_thinking)
    ;;
  0|false|FALSE|no|NO)
    mode_name="instruct"
    temperature="0.7"
    top_p="0.8"
    top_k="20"
    min_p="0.0"
    presence_penalty="1.5"
    repetition_penalty="1.0"
    thinking_args=()
    ;;
  *)
    echo "THINKING must be 0/1 or false/true (received: $thinking)" >&2
    exit 2
    ;;
esac
output_file="output/Qwen3.5-${model_size}_niche_codegeex_${mode_name}_seed${seed}.json"

# GH200 is available with both 96 GB HBM3 and 144 GB HBM3e. Leave more
# headroom on the 96 GB variant for CUDA graphs, MTP buffers, and UMA memory
# accounting. Every value can be overridden at qsub time for benchmark sweeps.
gpu_memory_mib="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
    awk -F, 'NR == 1 {gsub(/ /, "", $1); print $1}'
)"
if [[ "$gpu_memory_mib" -ge 120000 ]]; then
  default_gpu_memory_utilization="0.97"
  default_max_num_seqs="512"
else
  default_gpu_memory_utilization="0.95"
  if [[ "$model_size" == "9B" ]]; then
    default_max_num_seqs="384"
  else
    default_max_num_seqs="512"
  fi
fi

gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-$default_gpu_memory_utilization}"
max_num_seqs="${MAX_NUM_SEQS:-$default_max_num_seqs}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-65536}"
request_batch_size="${REQUEST_BATCH_SIZE:-4096}"
mtp_tokens="${MTP_TOKENS:-2}"

echo "GH200 HBM: ${gpu_memory_mib} MiB"
echo "vLLM tuning: model=${model_size} gpu_memory_utilization=${gpu_memory_utilization}" \
     "max_num_seqs=${max_num_seqs} max_num_batched_tokens=${max_num_batched_tokens}" \
     "request_batch_size=${request_batch_size} mtp_tokens=${mtp_tokens}"
echo "Sampling: mode=${mode_name} temperature=${temperature} top_p=${top_p}" \
     "top_k=${top_k} min_p=${min_p} presence_penalty=${presence_penalty}" \
     "repetition_penalty=${repetition_penalty} seed=${seed}"

# Build the leak-free CodeGeeX-style prompts once if they are not present.
if [[ ! -s "$input_file" ]]; then
  niche_languages="Fortran,Mathematica,Arturo,Julia,REXX,Swift,C,OCaml,PowerShell,Delphi,Racket,MATLAB,Rust,Ruby,C#,Java,C++,Common_Lisp,Elixir,Lua,Python,Tcl,Groovy,J,Pascal,AWK,Scala,Nim,Haskell,Clojure,Erlang,Factor,D,R,Ada,PHP,Icon,VB,Forth,BBC_Basic,AutoHotKey,COBOL,Perl,Go,F#"
  python run_preprocess.py \
    --input_file data/niche_test.json \
    --output_file "$input_file" \
    --source_names "$niche_languages" \
    --target_names "$niche_languages" \
    --sub_task RareTrans \
    --prompt_name CodeGeeX
fi

python run_translation.py \
  --input_file "$input_file" \
  --output_file "$output_file" \
  --vllm_path "$model_name" \
  --temperature "$temperature" \
  --top_p "$top_p" \
  --top_k "$top_k" \
  --min_p "$min_p" \
  --presence_penalty "$presence_penalty" \
  --repetition_penalty "$repetition_penalty" \
  --seed "$seed" \
  "${thinking_args[@]}" \
  --max_tokens 4096 \
  --batch_size "$request_batch_size" \
  --max_model_len 16384 \
  --max_num_batched_tokens "$max_num_batched_tokens" \
  --max_num_seqs "$max_num_seqs" \
  --gpu_memory_utilization "$gpu_memory_utilization" \
  --kv_cache_dtype fp8 \
  --tensor_parallel_size 1 \
  --enable_prefix_caching \
  --language_model_only \
  --mtp_tokens "$mtp_tokens" \
  --sync_interval 0 \
  --require_prediction_field
