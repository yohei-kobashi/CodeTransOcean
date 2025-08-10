#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gcb50389

module purge
module load cuda/12.8 python/3.12
export CUDA_VISIBLE_DEVICES=$(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
  awk -v U="$CUDA_VISIBLE_DEVICES" 'BEGIN{gsub(/ /,"",U)} $2==U{print $1}'
)
source env_vllm/bin/activate
cd CodeTransOcean/CodeLLM

# python run_translation.py --input_file data/preprocessed_multilingual_test.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_multilingual.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
python run_translation.py --input_file data/preprocessed_multilingual_test_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_multilingual_base.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
# python run_translation.py --input_file data/preprocessed_dl_test.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_dl.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
# python run_translation.py --input_file data/preprocessed_dl_test_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_dl_base.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
# python run_translation.py --input_file data/preprocessed_LLMTrans.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_LLMTrans.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
# python run_translation.py --input_file data/preprocessed_LLMTrans_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_LLMTrans_base.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
python run_translation.py --input_file data/preprocessed_niche_test.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_niche.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct
python run_translation.py --input_file data/preprocessed_niche_test_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_niche_base.json --vllm_path Qwen/Qwen3-Coder-30B-A3B-Instruct