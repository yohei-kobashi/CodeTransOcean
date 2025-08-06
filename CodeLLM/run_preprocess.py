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
This script processes an input JSONL file and creates new JSON data based on specified
source_names and target_names. Additionally, it uniformly samples a fixed number of examples
for each (source, target) combination if desired. It can dynamically include few-shot examples
from a training set based on source_lang/target_lang.
Templates for prompts are loaded from the prompts/ directory as .txt files.

Languages which can be included in the data are below:
AWK,Ada,Arturo,AutoHotKey,BBC_Basic,C,C#,C++,COBOL,Clojure,Common_Lisp,D,Delphi,Elixir,Erlang,F#,Factor,Forth,Fortran,Go,Groovy,Haskell,Icon,J,Java,Julia,Lua,MATLAB,Mathematica,Nim,OCaml,PHP,Pascal,Perl,PowerShell,Python,R,REXX,Racket,Ruby,Rust,Scala,Swift,Tcl,VB,mxnet,paddle,pytorch,tensorflow
"""

import logging
import argparse
import json
import random
import os
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_train_examples(train_path, source_lang, target_lang, num_examples=2):
    """
    Load up to num_examples examples from train_path matching the given languages.
    """
    examples = []
    try:
        with open(train_path, 'r', encoding='utf-8') as tf:
            for line in tf:
                ex = json.loads(line)
                if source_lang in ex and target_lang in ex:
                    examples.append(ex)
                    if len(examples) >= num_examples:
                        break
    except FileNotFoundError:
        logger.warning(f"Training file not found: {train_path}")
    return examples


def build_fewshot_text(examples, source_lang, target_lang):
    """
    Build a few-shot prefix from example list.
    """
    text = ''
    for i, ex in enumerate(examples, start=1):
        text += f"Example {i}:\n```{source_lang}\n{ex[source_lang]}```\n```{target_lang}\n{ex[target_lang]}```\n\n"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str, help="Path to input JSONL file.")
    parser.add_argument("--output_file", required=True, type=str, help="Path to output JSONL file.")
    parser.add_argument('--source_names', type=str, required=True,
                        help="Comma-separated list of source names. C,C++,C#,Java,Go,PHP,Python,VB")
    parser.add_argument('--target_names', type=str, required=True,
                        help="Comma-separated list of target names. C,C++,C#,Java,Go,PHP,Python,VB")
    parser.add_argument('--sub_task', type=str, default=None,
                        help="Sub-task name (e.g., MultilingualTrans, RareTrans, LLMTrans, etc.).")
    parser.add_argument('--sample_per_pair', type=int, default=0,
                        help="Number of samples to select per source-target combination. 0 means no sampling.")
    parser.add_argument('--prompt_name', type=str, default=None,
                        help="Name of the prompt template file (in prompts/ without .txt) to use.")
    parser.add_argument('--train_file', type=str, default='data/multilingual_train.json',
                        help="Path to multilingual training examples for few-shot.")
    parser.add_argument('--fewshot_num', type=int, default=2,
                        help="Number of few-shot examples to include per source-target pair.")
    parser.add_argument('--seed', type=int, default=42,
                        help="seed for sampling.")
    args = parser.parse_args()

    # Load prompt template
    if args.prompt_name:
        prompt_path = os.path.join('prompts', args.prompt_name + '.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as pf:
                prompt_template = pf.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            return
    else:
        prompt_template = 'Translate {source_lang} to {target_lang}: {source_code}'

    # Cache few-shot examples per language pair
    fewshot_cache = {}

    group_dict = {}
    with open(args.input_file, 'r', encoding='utf-8') as fi:
        for line in tqdm(fi):
            json_data = json.loads(line)
            cur_keys = list(json_data.keys())
            if args.sub_task in ['MultilingualTrans', 'RareTrans']:
                source_lang = cur_keys[2]
                target_lang = cur_keys[3]
            elif args.sub_task == 'LLMTrans':
                source_lang = cur_keys[3]
                target_lang = cur_keys[2]
            else:
                source_lang = cur_keys[1]
                target_lang = cur_keys[2]

            if source_lang in args.source_names.split(',') and target_lang in args.target_names.split(','):
                code_snippet = json_data[source_lang]

                # Construct the prompt
                # few-shot if prompt suggests it
                if args.prompt_name and any(k in args.prompt_name.lower() for k in ['few', 'fs']):
                    examples_text = ''
                    key = (source_lang, target_lang)
                    if key not in fewshot_cache:
                        fewshot_cache[key] = load_train_examples(
                            args.train_file,
                            source_lang,
                            target_lang,
                            num_examples=args.fewshot_num
                        )
                    examples = fewshot_cache[key]
                    examples_text = build_fewshot_text(examples, source_lang, target_lang)
                    source_prompt = prompt_template.format(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        source_code=code_snippet,
                        examples=examples_text
                    )
                else:
                    source_prompt = prompt_template.format(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        source_code=code_snippet,
                    )
                    
                target = json_data[target_lang]

                # Remove original language fields
                json_data.pop(source_lang, None)
                json_data.pop(target_lang, None)

                json_data['source'] = source_prompt
                json_data['target'] = target

                group_dict.setdefault((source_lang, target_lang), []).append(json_data)

    # Write output with optional sampling
    with open(args.output_file, 'w', encoding='utf-8') as fw:
        for _, samples in group_dict.items():
            if args.sample_per_pair > 0 and len(samples) > args.sample_per_pair:
                random.seed(args.seed)
                sampled = random.sample(samples, args.sample_per_pair)
            else:
                sampled = samples
            for sample in sampled:
                fw.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
