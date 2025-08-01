from autogen_core.models import (
    ChatCompletionClient,
    ModelFamily,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dataclasses import dataclass
import json
import math
import re
import threading
import time
from mathmessage import *

class MathRunner:
    def __init__(self, input_file_name:str, output_file_name:str, continue_from_checkpoint:bool, max_num_questions:int):
        self._correct= {}
        self._incorrect = {}
        self._input_file = open(input_file_name, 'r')
        self.open_output_file(output_file_name, continue_from_checkpoint)
        self._problem_index = -1
        self._problem_filter = None
        self._lock = threading.Lock()
        self._eof = False
        self._interuppted = False
        self._max_num_questions = max_num_questions
        self._question_count = 0

    def __del__(self):
        self.print_status()
        self._input_file.close()
        self._output_file.close()

    def open_output_file(self, output_file_name, continue_from_checkpoint):
        if continue_from_checkpoint:
            write_mode = 'a'
            with open(output_file_name, 'r') as file:
                for line in file:
                    json_object = json.loads(line)
                    index = json_object['index']
                    is_correct = (json_object['expected_answer'] == json_object['actual_answer'])
                    if is_correct:
                        self._correct[index] = True
                    else:
                        self._incorrect[index] = True
        else:
            write_mode = 'w'
        self._output_file = open(output_file_name, write_mode)

    def apply_filter(self, filter_file_name):
        with self._lock:
            self._problem_filter = {}
            with open(filter_file_name, 'r') as file:
                for line in file:
                    json_object = json.loads(line)
                    index = json_object['index']
                    if json_object['expected_answer'] != json_object['actual_answer']:
                        self._problem_filter[json_object['index']] = True
            print(f"Appling problem filter: {self._problem_filter.keys()}")

    def stop(self):
        with self._lock:
            self._interuppted = True

    def eof(self): 
        with self._lock:
            return self._eof 

    def get_next_question(self):
        json_object = None
        with self._lock:
            if self._interuppted:
                return None

            while not self._eof:
                if self._question_count >= self._max_num_questions and self._max_num_questions != -1:
                    self._eof = True
                    return None
                line = self._input_file.readline()
                self._problem_index = self._problem_index + 1
                if not line:
                    self._eof = True
                    return None
                json_object = json.loads(line.strip())
                if 'index' not in json_object:
                    json_object['index'] = self._problem_index
                if self._problem_filter is not None and json_object['index'] not in self._problem_filter:
                    print(f"skip filtered question: {json_object['index']}")
                    continue
                if json_object['index'] in self._correct or json_object['index'] in self._incorrect:
                    print(f"skip answered question: {json_object['index']}")
                    continue
                self._question_count += 1
                return json_object

    def flush(self, question, index, expected, actual, logger):
        with self._lock:
            data = {
                'question': question,
                'index': index,
                'lines': logger.lines(),
                'expected_answer': expected,
                'actual_answer': actual,
            }
            if expected != actual:
                self._incorrect[index] = True
                print(f"The answer for question {index} is incorrect: expected {expected}, got {actual}. {len(self._correct)} out of {len(self._incorrect) + len(self._correct)} answers are correct.")
            else:
                self._correct[index] = True
                print(f"The answer for question {index} is correct. {len(self._correct)} out of {len(self._incorrect) + len(self._correct)} answers are correct.")
            self._output_file.write(json.dumps(data) + '\n')
            self._output_file.flush()
            logger.reset()

    def print_status(self):
        print(f"Evaluated {len(self._incorrect) + len(self._correct)} questions, got {len(self._correct)} correct answers")
        if len(self._incorrect) > 0:
            print(f"Incorrect question indexes are {self._incorrect.keys()}")

    def parallel_run(self, workers):     
        threads = []
        for worker in workers:
            method = getattr(worker, 'run')
            thread = threading.Thread(target=method, args=(self,))
            threads.append(thread)
            thread.start()
        try:
            while not self.eof():
                time.sleep(2)
        except KeyboardInterrupt:
            print("caught exception, will exit after finishing current problems")
            self.stop()
        for thread in threads:
            thread.join()

class Logger:
    def __init__(self):
        self._lines = []

    def log(self, line):
        # print(line)
        self._lines.append(line)

    def log_error(self, line):
        print(line)
        self._lines.append(line)

    def reset(self):
        self._lines = []

    def lines(self):
        return self._lines

# an upper confidence bound implmentation with two arms.
class UCBStrategy:
    def __init__(self, logger, solver, fix_ucb):
        self._logger = logger
        self._solver = solver
        self._fix_ucb = fix_ucb
        self.reset()

    def reset(self):
        self._counts = {'Competitive': 0, 'Collaborative': 0}
        self._values = {'Competitive': 0.0, 'Collaborative': 0.0}
        self._rounds = 0
        self._last_score = 0.0
        self._first = True

    # when self._fix_ucb is '' (default), then the algorithm will decide between Competitive and Collaborative approach
    # when self._fix_ucb is 'comp' then the next strategy will be fixed, as Competitive
    # when self._fix_ucb is 'coll' then the next strategy will be fixed, as Collaborative
    def next_strategy(self, last_strategy, score):
        if self._fix_ucb == 'comp':
            self._logger.log(f"{self._solver} picked fixed strategy, competitive")
            return 'Competitive';
        if self._fix_ucb == 'coll':
            self._logger.log(f"{self._solver} picked fixed strategy, collaborative")
            return 'Collaborative'; 
        # update reward from the last round.
        if not self._first:
            reward = score - self._last_score
            self._logger.log(f"{self._solver} executed last strategy {last_strategy}, reward {reward}")
            self._values[last_strategy] = self._values[last_strategy] + reward 
            self._counts[last_strategy] = self._counts[last_strategy] + 1
        else:
            self._first = False
        self._last_score = score

        for s, c in self._counts.items():
            if c == 0:
                self._logger.log(f"{self._solver} picked strategy {s} for the first time")
                self._rounds = self._rounds + 1
                return s

        max_ucb = {}
        for s, c in self._counts.items():
            confidence_bound = math.sqrt(1.5 * math.log(self._rounds + 1) / c)
            max_ucb[s] = self._values[s] / c + confidence_bound
        if (max_ucb['Collaborative'] >= max_ucb['Competitive']):
            strategy = 'Collaborative'
        else:
            strategy = 'Competitive'
        self._logger.log(f"{self._solver} picked strategy {strategy}, ucb {max_ucb}")
        self._rounds = self._rounds + 1
        return strategy

class BasicStrategy:
    def __init__(self, logger, solver):
        self._logger = logger
        self._solver = solver

    def reset(self):
        pass

    def next_strategy(self, last_strategy, score):
        if score >= 0.5:
            strategy = 'Collaborative'
        else:
            strategy = 'Competitive'
        self._logger.log(f"{self._solver} picked strategy {strategy}, prm score {score}")
        return strategy

class CompetitiveStrategy:
    def reset(self):
        pass 

    def next_strategy(self, last_strategy, score):
        return 'Competitive'

class CollaborativeStrategy:
    def reset(self):
        pass 

    def next_strategy(self, last_strategy, score):
        return 'Collaborative'

class LLMModelClient:
    def __init__(self, solver, llm_client, logger):
        self._solver = solver
        self._llm_client = llm_client
        self._logger = logger
  
    async def chat(self, system_message, prompt, temperature = None):
        error = None
        if temperature is None:
            extra_create_args = {}
        else:
            extra_create_args = {'temperature': temperature}
        for i in range(5):
            try:
                return await self._llm_client.create(
                    messages = [system_message, UserMessage(content=prompt, source="user")],
                    extra_create_args = extra_create_args
                )
            except Exception as e:
                time.sleep(i + 1)
                error = e
                self._logger.log_error(f"Encountered {e} from LLM {self._solver}, backoff {i} seconds")
        return Answer(content=f"Encountered errors")
  
    async def multi_chat(self, system_message, prompt, temperature, samples):
        error = None
        extra_create_args = {
            'temperature' : temperature
        }
        responses = []
        for i in range(samples * 5):
            try:
                response = await self._llm_client.create(
                    messages = [system_message, UserMessage(content=prompt, source="user")],
                    extra_create_args = extra_create_args,
                )
                responses.append(response.content)
                if len(responses) >= samples:
                    break
            except Exception as e:
                time.sleep(i + 1)
                error = e
                self._logger.log_error(f"Encountered {e} from LLM {self._solver}, backoff {i} seconds")
        return responses

class LLMModelClientFactory:
    _openai_models = ["gpt-4o-mini", "gpt-4o"]
    _os_model_info = {
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.R1,
        "structured_output": True,
    }

    def __init__(self, openai_api_key, opensource_base_url, opensource_api_key):
        self._openai_api_key = openai_api_key
        self._opensource_base_url = opensource_base_url
        self._opensource_api_key = opensource_api_key

    def create_client(self, model):
        # only single choice allowed in autogen
        if model in LLMModelClientFactory._openai_models:
            return OpenAIChatCompletionClient(
                model=model,
                api_key=self._openai_api_key,
            )
        else:
            return OpenAIChatCompletionClient(
                model=model,
                api_key=self._opensource_api_key,
                base_url=self._opensource_base_url,
                model_info=LLMModelClientFactory._os_model_info,
            )

    def create_llm_client(self, solver, model, logger):
        return LLMModelClient(solver, self.create_client(model), logger)

class RegexAnswerParser:
    def __init__(self, llm_client, logger):
        self._llm_client = llm_client
        self._system_message = SystemMessage(
            content=("You are a helpful math assistant. Please reduce the provided answer to a single numeric value. "
                     "state the answer using the format: 'The answer is #### [final numerical answer].'"))
        self._logger = logger

    async def parse_answer(self, result:str, call_llm = True):
        match = re.search(r"The answer is\D+?(\d+(\,\d{3})+)\D+?$", result)
        if match is not None:
            return self.format_answer(match.group(1))
        match = re.search(r"The answer is\D+?(\-?\d+(\.\d+)?)\D+?$", result)
        if match is not None:
            return self.format_answer(match.group(1))
        match = re.search(r"####\D+?(\d+(\,\d{3})+)", result)
        if match is not None:
            return self.format_answer(match.group(1))
        match = re.search(r"####\D+?(\-?\d+(\.\d+)?)", result)
        if match is not None:
            return self.format_answer(match.group(1))
        if call_llm:
            match = re.search(r"The answer is (.*)$", result)
            if match is not None:
                result = await self._llm_client.chat(self._system_message, f"Please reduce the provided answer to a single numeric value. Input: The answer is {match.group(1)}. Return answer as a numeric value, using the format: 'The answer is #### [final numerical answer].'")
                self._logger.log("Called LLM to parse answer, got " + result.content)
                return await self.parse_answer(result.content, call_llm = False)
        return None

    def format_answer(self, match:str):
        if ',' in match:
            return ''.join(match.split(','))
        else:
            match_trailing = re.search(r"(\-?\d+)\.0+$", match)
            if match_trailing is not None:
                return str(int(match_trailing.group(1)))
            else:
                answer = match
                if '.' in answer:
                    return str(float(answer))
                else:
                    return str(int(answer))
