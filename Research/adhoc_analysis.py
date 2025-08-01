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
    args = parser.parse_args()
    return args

class StatsAggregator:
    def __init__(self, solvers, answer_parser):
        self._answer_parser = answer_parser
        self._solvers = solvers
        self._total = 0
        self._correct = 0
        self._total_only_correct_answers = 0
        self._total_only_incorrect_answers = 0
        self._total_mixed_answers = 0
        self._correct_at_round = {}
        self._total_round = {}
        self._switch_to_correct = {'Competitive': 0, 'Collaborative': 0}
        self._switch_to_incorrect = {'Competitive': 0, 'Collaborative': 0}
        self._correct_has_highest_prm_score = 0
        self._incorrect_has_highest_prm_score = 0
        self._total_critiques = 0
        self._critique_contains_answer = 0
        self._total_collaborative = 0
        self._collaborative_contradictions = 0
        self._done = False
        #self._round_map = {}

    def print_stats(self):
        print(f"total questions = {self._total}, {self._correct} correct answers.")
        print(f"In {self._total_only_correct_answers} questions, all LLMs returned the right answer.")
        print(f"In {self._total_only_incorrect_answers} questions, the correct answer never appeared.")
        print(f"Number of times an LLM switched from an incorrect answer to a correct answer {self._switch_to_correct}")
        print(f"Number of times an LLM switched from a correct answer to an incorrect answer {self._switch_to_incorrect}")
        print(f"For {self._correct_has_highest_prm_score} cases, the correct answer has the highest prm score")
        print(f"For {self._incorrect_has_highest_prm_score} cases, an incorrect answer has the highest prm score")
        print(f"{self._critique_contains_answer} out of {self._total_critiques} critique responses included an answer to the question.")
        print(f"{self._collaborative_contradictions} out of {self._total_collaborative} collaborative steps combined two solutions with different answers.")
        print(f"{self._done}")
        #print(self._round_map)

    def parse_lines(self, expected_answer, lines):
        has_correct_answer = False
        has_incorrect_answer = False
        debate_round = 0
        solver_to_answer = {}
        solver_to_score = {}
        solver_to_strategy = {}
        last_solver_to_answer = {}
        last_solver_to_strategy = {}

        for debug_line in lines:
            self._done = True
            line = debug_line.replace('\n', ' ')
            match = re.search(r"[-]+[ ]+Solver (.*)/default Answer:[ ]+(.*)", line)
            if match is not None:
                solver = match.group(1)
                try:
                  answer = asyncio.run(self._answer_parser.parse_answer(match.group(2)))
                except Exception as e:
                    print(f"Failed to parse answer match.group(2), error {e}")
                    answer = None
                if answer == expected_answer:
                    has_correct_answer = True
                    if solver in last_solver_to_strategy and solver in last_solver_to_answer:
                        if last_solver_to_answer[solver] != expected_answer:
                            self._switch_to_correct[last_solver_to_strategy[solver]] += 1
                elif answer is not None:
                    has_incorrect_answer = True
                    if solver in last_solver_to_strategy and solver in last_solver_to_answer:
                        if last_solver_to_answer[solver] == expected_answer:
                            self._switch_to_incorrect[last_solver_to_strategy[solver]] += 1
                solver_to_answer[match.group(1)] = answer
                continue
            match = re.search(r"Positive probability for (.*) is: \[(.*)\]", line)
            if match is not None:
                solver_to_score[match.group(1)] = float(match.group(2))
                continue
            match = re.search(r"(.*) picked strategy (.*?),", line)
            if match is not None:
                solver_to_strategy[match.group(1)] = match.group(2)
                continue
            match = re.search(r"Solver (.*) received critique from (.*)$", line)
            if match is not None:
                self._total_critiques += 1
                try:
                    answer = asyncio.run(self._answer_parser.parse_answer(match.group(2)))
                except Exception as e:
                    print(f"Failed to parse answer {match.group(2)}, error: {e}")
                    answer = None
                if answer is not None:
                    self._critique_contains_answer += 1
                continue
            match = re.search(r"Solver (.*)/default .*solution_2: (.*)$", line)
            if match is not None:
                self._total_collaborative += 1
                solver = match.group(1)
                try:
                    answer = asyncio.run(self._answer_parser.parse_answer(match.group(2)))
                except Exception as e:
                    print(f"Failed to parse answer {match.group(2)}, error: {e}")
                    answer = None
                if solver in last_solver_to_answer and answer != last_solver_to_answer[solver]:
                    self._collaborative_contradictions += 1
                continue
            match = re.search(r"received responses from all", line)
            if match is not None:
                debate_round += 1
                if debate_round not in self._total_round:
                    self._total_round[debate_round] = 0
                self._total_round[debate_round] += 1
                if len(solver_to_answer) == self._solvers:
                    answers = list(solver_to_answer.values())
                    majority_answer = max(set(answers), key=answers.count)
                    count = answers.count(majority_answer)
                    if count > self._solvers / 2 and majority_answer == expected_answer:
                        if debate_round not in self._correct_at_round:
                            self._correct_at_round[debate_round] = 0
                        self._correct_at_round[debate_round] += 1
                correct_prm_score = 0.0
                incorrect_prm_score = 0.0
                for solver, answer in solver_to_answer.items():
                    if answer == expected_answer:
                        correct_prm_score = max(correct_prm_score, solver_to_score[solver])
                    else:
                        incorrect_prm_score = max(incorrect_prm_score, solver_to_score[solver])
                if correct_prm_score > 0 and incorrect_prm_score > 0:
                    if correct_prm_score > incorrect_prm_score:
                        self._correct_has_highest_prm_score += 1
                    else:
                        self._incorrect_has_highest_prm_score += 1
                last_solver_to_answer = solver_to_answer
                last_solver_to_strategy = solver_to_strategy
                solver_to_answer = {}
                solver_to_strategy = {}
        #self._round_map.setdefault(debate_round, 0)
        #self._total_round[debate_round] = self._total_round[debate_round] + 1
        if not has_correct_answer:
            self._total_only_incorrect_answers += 1
        elif not has_incorrect_answer:
            self._total_only_correct_answers += 1
        else:
            self._total_mixed_answers += 1

    def add(self, json_object):
        print(f"Processing question {json_object['index']}")
        expected_answer = json_object['expected_answer']
        self._total += 1
        if json_object['actual_answer'] == expected_answer:
            self._correct += 1
        self.parse_lines(expected_answer, json_object['lines'])

api_key = os.environ.get('API_KEY')
os_api_key = os.environ.get('OPEN_SOURCE_API_KEY')
llm_factory = LLMModelClientFactory(api_key, '', os_api_key)
logger = Logger()
llm_client = llm_factory.create_llm_client("answer_parser", "gpt-4o", logger)

args = parse_arg()
stats_aggregator = StatsAggregator(args.solvers, RegexAnswerParser(llm_client, logger))
try:
    with open(args.file, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            stats_aggregator.add(json_object)
except Exception as e:
    print(f"Encountered exception: {e}")
stats_aggregator.print_stats()
