# coding=utf-8
"""
This script reads an input JSONL file and counts the number of samples for each
(source, target) combination based on provided source_names, target_names, and sub_task.
It then prints out the counts so you can decide how many samples to select per pair.
"""

import argparse
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str, help="Path to input JSONL file.")
    parser.add_argument('--source_names', type=str, required=True,
                        help="Comma-separated list of source names.")
    parser.add_argument('--target_names', type=str, required=True,
                        help="Comma-separated list of target names.")
    parser.add_argument('--sub_task', type=str, required=True,
                        help="Sub-task name (e.g., MultilingualTrans, RareTrans, LLMTrans, etc.).")
    args = parser.parse_args()
    
    # Dictionary to count samples per (source_lang, target_lang)
    combination_counts = {}

    with open(args.input_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            try:
                json_data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping an invalid JSON line.")
                continue
            cur_keys = list(json_data.keys())
            # Determine source and target language keys based on sub_task
            if args.sub_task in ['MultilingualTrans', 'RareTrans']:
                source_lang = cur_keys[2]
                target_lang = cur_keys[3]
            elif args.sub_task in ['LLMTrans']:
                source_lang = cur_keys[3]
                target_lang = cur_keys[2]
            else:
                source_lang = cur_keys[1]
                target_lang = cur_keys[2]

            # Only count if the source and target are in the specified lists
            if source_lang in args.source_names.split(',') and target_lang in args.target_names.split(','):
                key = (source_lang, target_lang)
                if key not in combination_counts:
                    combination_counts[key] = 0
                combination_counts[key] += 1

    # Output the counts for each (source, target) combination
    print("Counts for each (source, target) combination:")
    for key, count in sorted(combination_counts.items()):
        print(f"{key[0]} -> {key[1]}: {count}")


if __name__ == "__main__":
    main()
