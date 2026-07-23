import json
import random

max_N = 10
seed = 42

lang_pairs = {}
for row in open("data/niche_test.json"):
    data = json.loads(row)
    langs = tuple(list(data.keys())[2:])
    if not langs in lang_pairs:
        lang_pairs[langs] = []
    lang_pairs[langs].append(data)

with open("data/niche_test_sampled.jsonl", "w") as f:
    for pairs in lang_pairs.values():
        random.seed(seed)
        if len(pairs) > max_N:
            pairs = random.sample(pairs, max_N)
        f.write("\n".join([json.dumps(row) for row in pairs]) + "\n")