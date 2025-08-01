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
        print(f"WSJ: mad_util.py MathRunner.parallel_run, num_workers={len(workers)}")
        for worker in workers:
            print(f"The current worker is {worker.get_name()}")
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

class LLMModelClient:
    def __init__(self, solver, llm_client, logger):
        self._solver = solver
        self._llm_client = llm_client
        self._logger = logger
  
    async def chat(self, system_message, prompt):
        for i in range(5):
            try:
                return await self._llm_client.create(
                    messages = [system_message, UserMessage(content=prompt, source="user")]
                )
            except Exception as e:
                time.sleep(i + 1)
                error = e
                self._logger.log_error(f"Encountered {e} from LLM {self._solver}, backoff {i} seconds")
        return Answer(content=f"Encountered errors")

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
        print(f"mad_util.py LLMModelClientFactory.create_client model={model}")
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