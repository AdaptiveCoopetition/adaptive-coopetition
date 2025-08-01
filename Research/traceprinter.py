import argparse
import asyncio
import json
import random
import os
import re
import sys
from util import *

def parse_arg():    
    parser = argparse.ArgumentParser(description="printer")
    parser.add_argument(
        "--file", type=str, default="log/output.jsonl",help="result logging, also serves as a checkpoint."
    )
    parser.add_argument("--solvers", type=int, default=3, help="number of math solvers in the debate.")
    parser.add_argument("--index", type = str, default = "", help = "analyze any specific question")
    parser.add_argument("--pickcorrect", action='store_true', help = "when index is -1, whether or not to pick an incorrect answer" )
    parser.add_argument("--exclude", type = list[int], default = [], help = "problems to exclude")
    parser.add_argument("--mode", type = str, choices=["basic", "summary", "verbose"], default="summary", help="ways to organize output")
    args = parser.parse_args()
    return args

class LogPrinter:
    def __init__(self, solvers, verbose, answer_parser):
        self._answer_parser = answer_parser
        self._solvers = solvers
        self._verbose = verbose
        self._round = 0
        self._log_by_solver = {}
        for i in range(self._solvers):
            self._log_by_solver["MathSolver"+str(i)] = ["Round 0 starts."]

    def flush_round(self):
        self._round = self._round + 1
        for k, v in self._log_by_solver.items():
            self._log_by_solver[k].append(f"Round {self._round} starts.")

    def parse_line(self, debug_line):
        line = debug_line.replace('\n', ' ')
        match = re.search(r"[-]+[ ]+Solver (.*)/default Answer:[ ]+(.*)", line)
        if match is not None:
            if self._verbose:
                self._log_by_solver[match.group(1)].append(f"Response: {match.group(2)}")
            else:
                answer = asyncio.run(self._answer_parser.parse_answer(match.group(2)))
                self._log_by_solver[match.group(1)].append(f"Answer: {answer}")
            return False
        match = re.search(r"Positive probability for (.*) is: \[(.*)\]", line)
        if match is not None:
            self._log_by_solver[match.group(1)].append("PRM score: " + match.group(2))
            return False
        match = re.search(r"(.*) executed last strategy (.*)$", line)
        if match is not None:
            self._log_by_solver[match.group(1)].append("Last strategy: " + match.group(2))
            return False
        match = re.search(r"(.*) picked strategy (.*)$", line)
        if match is not None:
            self._log_by_solver[match.group(1)].append("Next strategy to try: " + match.group(2))
            return False
        if self._verbose:
            match = re.search(r"Solver (.*) received critique from (.*)$", line)
            if match is not None:
                self._log_by_solver[match.group(1)].append("Critique from " + match.group(2))
                return False
        match = re.search(r"received responses from all", line)
        return match is not None

    def run(self, lines):
        for debug_line in lines:
            if self.parse_line(debug_line):
                self.flush_round()

        for k, v in self._log_by_solver.items():
            print(f"{'-'*80}\nSolver {k} debating flow:\n{'-'*80}")
            # remove last line, which is start of an empty round.
            v.pop()
            for line in v:
                print(line)

correct = []
incorrect = []
multiagent_index_to_jsonl = {}

api_key = os.environ.get('API_KEY')
os_api_key = os.environ.get('OPEN_SOURCE_API_KEY')
llm_factory = LLMModelClientFactory(api_key, '', os_api_key)
logger = Logger()
llm_client = llm_factory.create_llm_client("answer_parser", "gpt-4o", logger)

args = parse_arg()
exclude_set = set(args.exclude)
with open(args.file, 'r') as file:
    for line in file:
        json_object = json.loads(line)
        index = json_object['index']
        if index in exclude_set:
            continue
        multiagent_index_to_jsonl[index] = json_object
        is_correct = (json_object['expected_answer'] == json_object['actual_answer'])
        if is_correct:
            correct.append(index)
        else:
            incorrect.append(index)
            
question_array = []
if args.index:
    questions = args.index.split(',')
    for q in questions:
        question_array.append(int(q))
else:
    if not args.pickcorrect:
        if len(incorrect)!=0:
            question_array.append(random.choice(incorrect))
    else:
        if len(correct)!=0:
            print("Here")
            question_array.append(random.choice(correct))

for question_index in question_array:
    if question_index not in multiagent_index_to_jsonl:
        print(f"Invalid question index: {question_index}")
        continue

    queried_json_object = multiagent_index_to_jsonl[question_index]
    print(f"question index: {question_index}")
    print(queried_json_object['question'])
    print(f"expected answer: {queried_json_object['expected_answer']} \n\n")
    print(f"multiagent response: \n")
    if args.mode == "basic":
        for line in queried_json_object['lines']:
            print(line)
    else:
        if args.mode == "verbose":
            verbose = True
        else:
            verbose = False
        log_printer = LogPrinter(args.solvers, verbose, RegexAnswerParser(llm_client, logger))
        log_printer.run(queried_json_object['lines'])
    print("\n\n")



