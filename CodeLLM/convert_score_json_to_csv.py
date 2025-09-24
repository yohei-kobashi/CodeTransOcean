import json
import csv
import glob

output = []
for json_file in glob.glob("scores/*.json"):
    data = json.load(open(json_file))
    for score_name in ["blue", "codeblue", "DSR"]:
        if not score_name in data["overall"]:
            continue
        output.append([json_file, "overall", score_name, data["overall"][score_name]])
        for pair,v in data["per_pair"].items():
            output.append([json_file, pair, score_name, data["per_pair"][pair][score_name]])

csv.writer(open("scores/all_scores.tsv", "w"), delimiter="\t").writerows(output)
            