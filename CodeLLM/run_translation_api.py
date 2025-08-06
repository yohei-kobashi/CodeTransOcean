#!/usr/bin/env python
# coding=utf-8
"""
Translation script using the OpenAI, Gemini or Anthropic API.
This script reads a preprocessed JSONL file (each line must contain a "source" field),
translates each sample via the OpenAI API, and writes the output with the translation
stored under the "prediction" field to an output JSONL file.
"""

import argparse
import json
import logging
import openai
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm
import os
import sys
import time
import traceback

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def llm_api(text='', dotenv_path=None, platform=None, model='gpt-3.5-turbo', reasoning=False, delay=1.0, i=0):
    load_dotenv(dotenv_path)
    messages = []
    messages.append({"role": "user", "content": text})
    while True:
        try:
            claude_max_tokens = 20000
            if model == "claude-3-5-haiku-20241022":
                claude_max_tokens = 8192
            claude_budget_tokens = 16000
            max_completion_tokens = claude_max_tokens + claude_budget_tokens
            if model == "gpt-4o":
                max_completion_tokens = 16384
            if platform == "Anthropic":
                client = anthropic.Anthropic(
                    api_key=os.environ["ANTHROPIC_API_KEY"]
                )
                params = {
                    "model": model,
                    "max_tokens": claude_max_tokens,
                    # "temperature": 0,
                    "messages": messages,
                }
                if reasoning is True:
                    params["thinking"] = {"type": "enabled", "budget_tokens": claude_budget_tokens}
                response = client.messages.create(**params)
                time.sleep(delay)
                text = response.content[-1].text if reasoning is True else response.content[0].text
                
            elif platform == "Google":
                client = openai.OpenAI(
                    api_key=os.environ["GOOGLE_API_KEY"],
                    base_url="https://generativelanguage.googleapis.com/v1beta/"
                )
                response = client.chat.completions.create(
                    model=model,
                    # temperature=0,
                    # top_p=0,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens
                )
                time.sleep(delay)
                text = response.choices[0].message.content
                
            else:
                openai.key = os.environ["OPENAI_API_KEY"]
                response = openai.chat.completions.create(
                    model=model,
                    # temperature=0,
                    # top_p=0,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens
                )
                time.sleep(delay)
                text = response.choices[0].message.content
            
            return text.strip()
            
        except Exception as e:
            print("An error occurred: ", str(e))
            sys.stdout.flush()
            traceback.print_exc()
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description="Translate using OpenAI API")
    parser.add_argument("--input_file", required=True, type=str, help="Input JSONL file (each line contains a 'source' field)")
    parser.add_argument("--output_file", required=True, type=str, help="Output JSONL file")
    parser.add_argument('--dotenv_path', '-dotenv_path', help="Path of .env")
    parser.add_argument('--platform', '-platform', help="Platform of the api. Options available:['OpenAI', 'Google', 'Anthropic]")
    parser.add_argument('--reasoning', '-reasoning',action='store_true',help="Using a reasoning model.")
    parser.add_argument('--delay', '-delay',type=float,default=1.0,help="Number of seconds to wait between consecutive API requests.")
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str, help="OpenAI model to use (e.g., gpt-3.5-turbo)")
    args = parser.parse_args()

    # Process each line from the input file
    print(args)
    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping an invalid JSON line.")
                continue

            # Retrieve the translation instruction from the "source" field
            prompt = data.get("source", "")
            
            if not prompt:
                logger.warning("Skipping a line with an empty 'source' field.")
                continue

            try:
                translation = llm_api(prompt, args.dotenv_path, args.platform, args.model, args.reasoning, args.delay)
                
            except Exception as e:
                logger.error(f"Error during OpenAI API call: {e}")
                translation = ""

            # Add the generated translation under the 'prediction' field
            data["prediction"] = translation
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info(f"Translation results have been written to {os.path.abspath(args.output_file)}.")


if __name__ == "__main__":
    main()
