#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe
module purge
module load cuda/12.8
module load cudnn/9.10.1.4
module load nvidia/25.3
module load nv-hpcx/25.3
source /work/gj26/b20048/miniconda3/etc/profile.d/conda.sh
conda activate inference_env
export CUDA_VISIBLE_DEVICES=0
export PATH="$CONDA_PREFIX/bin:/opt/rh/gcc-toolset-14/root/usr/bin:$PATH"

export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export TRITON_CC="$CC"
export TRITON_CXX="$CXX"
export CUDAHOSTCXX="$CXX"

export PYTHONNOUSERSITE=1
cd CodeTransOcean/CodeLLM

# python run_translation.py --input_file data/preprocessed_multilingual_test.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_multilingual.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
# python run_translation.py --input_file data/preprocessed_multilingual_test_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_multilingual_base.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
# python run_translation.py --input_file data/preprocessed_dl_test.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_dl.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
python run_translation.py --input_file data/preprocessed_dl_test_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_dl_base.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
# python run_translation.py --input_file data/preprocessed_LLMTrans.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_LLMTrans.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
python run_translation.py --input_file data/preprocessed_LLMTrans_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_LLMTrans_base.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
# python run_translation.py --input_file data/preprocessed_niche_test.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_niche.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826
python run_translation.py --input_file data/preprocessed_niche_test_base.json --output_file output/Qwen3-Coder-30B-A3B-Instruct_0826_niche_base.json --vllm_path /work/go25/share/model/Qwen3-Coder-30B-A3B-Instruct-mcore-hf_code_trans_489pairs_0826