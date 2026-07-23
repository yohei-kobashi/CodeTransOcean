#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=36:00:00
#PBS -P gcb50389

module purge
module load cuda/12.8 python/3.12
export CUDA_VISIBLE_DEVICES=$(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
  awk -v U="$CUDA_VISIBLE_DEVICES" 'BEGIN{gsub(/ /,"",U)} $2==U{print $1}'
)
source env_vllm/bin/activate
cd CodeTransOcean/CodeLLM

# python run_translation.py --input_file data/preprocessed_multilingual_test_codegeex.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_multilingual_codegeex.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
# python run_translation.py --input_file data/preprocessed_dl_test_codegeex.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_dl_codegeex.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
# python run_translation.py --input_file data/preprocessed_LLMTrans_codegeex.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_LLMTrans_codegeex.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
python run_translation.py --input_file data/preprocessed_niche_test_codegeex.json --output_file output/Qwen/Qwen3.5-4B_niche_codegeex.json --vllm_path Qwen/Qwen3.5-4B
python run_translation.py --input_file data/preprocessed_niche_test_codegeex.json --output_file output/Qwen/Qwen3.5-9B_niche_codegeex.json --vllm_path Qwen/Qwen3.5-9B