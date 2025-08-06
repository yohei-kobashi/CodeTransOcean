# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script evaluates the performance of translated code.
For each (source, target) combination (languages), it computes the
exact match (EM), BLEU, and CodeBLEU scores.
It also computes overall scores across all samples.
"""

import logging
import argparse
import numpy as np
import json
import re
import random
import subprocess
import sys

from evaluator.CodeBLEU import calc_code_bleu
# We'll use nltk's BLEU for our evaluation. Ensure nltk is installed.
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str, help="Path to the JSONL file with evaluation samples.")
    parser.add_argument("--output_file", required=True, type=str, help="Path to output JSON file to write scores.")
    parser.add_argument('--source_names', type=str, required=True,
                        help="Comma-separated list of source languages.")
    parser.add_argument('--target_names', type=str, required=True,
                        help="Comma-separated list of target languages.")
    parser.add_argument('--codebleu', action='store_true', help="If set, compute CodeBLEU.")
    parser.add_argument('--execute', action='store_true', help="If set, execute source code.")
    parser.add_argument('--naive', action='store_true', help="Use source code as hypothesis instead of prediction.")
    args = parser.parse_args()

    # Containers for overall evaluation
    overall_dev_accs = []
    overall_hypotheses = []
    overall_references = []

    # Group data by (source, target) combination.
    # Each value is a dict with keys: dev_accs, hypotheses, references.
    group_data = {}

    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line)
            # Extract languages from the source prompt using regex.
            matches = re.search(r"^Translate (\S+) to (\S+): ", json_data['source'])
            if not matches:
                matches = re.search(r"Here is the \S+ code:", json_data['source'])
                if not matches:
                    continue
                langs = re.search(r"Your job is to translate code from (\S+) to (\S+)\.", json_data['source'])
                source_lang, target_lang = langs.groups()
            else:
                source_lang, target_lang = matches.groups()
            # Remove the prefix to obtain the original source code.
            source_code = json_data['source'][matches.end():]
            # Check if this sample is in the specified language lists.
            if source_lang in args.source_names.split(',') and target_lang in args.target_names.split(','):
                # Depending on args.naive, choose hypothesis as the source code or the model's prediction.
                if args.naive:
                    hypothesis = source_code
                    dev_acc = (source_code == json_data['target'].strip())
                else:
                    hypothesis = re.sub("```\n.*$", "", re.sub("^.*?```[^\n]+\n", "", json_data['prediction'], flags= re.MULTILINE | re.DOTALL), flags= re.MULTILINE | re.DOTALL).strip()
                    hypothesis = hypothesis.replace("```", "").strip()
                    dev_acc = (hypothesis == json_data['target'].strip())
                reference = json_data['target'].strip()
                output = json_data.get('output', '')
                
                overall_dev_accs.append(dev_acc)
                overall_hypotheses.append(hypothesis)
                overall_references.append(reference)
                
                key = (source_lang, target_lang)
                if key not in group_data:
                    group_data[key] = {'dev_accs': [], 'hypotheses': [], 'references': [], 'output': []}
                group_data[key]['dev_accs'].append(dev_acc)
                group_data[key]['hypotheses'].append(hypothesis)
                group_data[key]['references'].append(reference)
                group_data[key]['output'].append(output)

    # Function to compute corpus BLEU using nltk.
    def compute_bleu(refs, hyps):
        # nltk.corpus_bleu expects list of references per hypothesis; tokenize by splitting on whitespace.
        # Each reference is a list of tokens inside a list.
        tokenized_refs = [[ref.split()] for ref in refs]
        tokenized_hyps = [hyp.split() for hyp in hyps]
        # corpus_bleu returns a score between 0 and 1.
        return corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=SmoothingFunction().method1) * 100


    def run_python_script(code):
        open('tmp.py', 'w').write(code)
            
        # sys.executableを使って同じPython実行環境で起動
        cmd = [sys.executable, 'tmp.py']
        try:
            # 20秒のタイムアウト付きで実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20
            )
            output = (result.stdout or "") + (result.stderr or "")
            k = 0 if result.returncode == 0 else 1
        except subprocess.TimeoutExpired:
            output = "timeout"
            k = 1
        except subprocess.CalledProcessError as e:
            output = (e.stdout or "") + (e.stderr or "")
            k = 1
        return output, k
    
    def deal_output(output):
        output = re.sub(r'\n', ' ', output)
        output = re.sub(r'\s+', ' ', output)
        output = re.sub(r'^[\s\n]+|[\s\n]+$', '', output)
        return output

    # Compute scores for each (source, target) combination.
    per_pair_results = {}
    for key, data in group_data.items():
        em = round(np.mean(data['dev_accs']) * 100, 2)
        bleu = round(compute_bleu(data['references'], data['hypotheses']), 2)
        per_pair_results[f"{key[0]}_to_{key[1]}"] = {'em': em, 'bleu': bleu}
        if args.codebleu:
            # calc_code_bleu.get_codebleu_list expects references in a list (containing one list of references)
            codebleu = calc_code_bleu.get_codebleu_list([data['references']], data['hypotheses'], 'python')
            codebleu = round(codebleu * 100, 2)
            per_pair_results[f"{key[0]}_to_{key[1]}"]['codebleu'] = codebleu
        if args.execute:
            execute_N = 0
            correct_N = 0
            for output, code in zip(data['output'], data['hypotheses']):
                if output:
                    execute_N += 1
                    code_output, err = run_python_script(code)
                    if not err and deal_output(output) == deal_output(code_output):
                        correct_N += 1
            per_pair_results[f"{key[0]}_to_{key[1]}"]['execute_N'] = execute_N
            per_pair_results[f"{key[0]}_to_{key[1]}"]['correct_N'] = correct_N
            if execute_N:
                per_pair_results[f"{key[0]}_to_{key[1]}"]['DSR'] = correct_N / execute_N

    # Compute overall scores.
    overall_em = round(np.mean(overall_dev_accs) * 100, 2)
    overall_bleu = round(compute_bleu(overall_references, overall_hypotheses), 2)
    overall_result = {'em': overall_em, 'bleu': overall_bleu}
    if args.codebleu:
        overall_codebleu = calc_code_bleu.get_codebleu_list([overall_references], overall_hypotheses, 'python')
        overall_codebleu = round(overall_codebleu * 100, 2)
        overall_result['codebleu'] = overall_codebleu
    if args.execute:
        overall_execute_N = sum([result['execute_N'] for result in per_pair_results.values()])
        if overall_execute_N:
            overall_correct_N = sum([result['correct_N'] for result in per_pair_results.values()])
            overall_DSR = overall_correct_N / overall_execute_N
            overall_result['DSR'] = overall_DSR

    # Create the final results dictionary.
    result = {
        'per_pair': per_pair_results,
        'overall': overall_result
    }
    # Write the results to the specified output file.
    with open(args.output_file, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
