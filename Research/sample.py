import argparse
import json
import random
import sys

parser = argparse.ArgumentParser(description="download")
parser.add_argument(
    "--baseline", type=str, default="log/gsm8k-platinum-baseline-output.jsonl",
    help="baseline result file"
)
parser.add_argument(
    "--correct", type=int, default=100, help="correct questions"
)
parser.add_argument(
    "--incorrect", type=int, default=100, help="incorrect questions"
)
parser.add_argument(
    "--output", type=str, default="data/biased_platinum.jsonl", help="output_file"
)
args = parser.parse_args()
correct = {}
incorrect = {}
with open(args.baseline, 'r') as file:
    for line in file:
        json_object = json.loads(line)
        is_correct = (json_object['expected_answer'] == json_object['actual_answer'])
        data = {
            'question': json_object['question'],
            'answer': '#### ' + json_object['expected_answer'],
            'index': json_object['index']
        }
        if is_correct:
            correct[json_object['index']] = data
        else:
            incorrect[json_object['index']] = data

if len(correct) < args.correct or len(incorrect) < args.incorrect:
    raise ValueError(f"Not enough questions to pull from: correct{len(correct)}, incorrect {len(incorrect)}")

problem_set = []
for i in range(args.correct):
    print(f"{len(correct)} remaining correct questions")
    problem = random.choice(list(correct.keys()))
    problem_set.append(correct[problem])
    del correct[problem]

for i in range(args.incorrect):
    print(f"{len(incorrect)} remaining incorrect questions")
    problem = random.choice(list(incorrect.keys()))
    problem_set.append(incorrect[problem])
    del incorrect[problem]

random.shuffle(problem_set)
with open(args.output, 'w') as output_file:
    for json_object in problem_set:
        print(json_object)
        output_file.write(json.dumps(json_object) + '\n')



