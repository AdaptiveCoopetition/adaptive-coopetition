import argparse
import asyncio
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
import os
from util import *
from mathmessage import *

class BaselineWorker:
    def __init__(self, model, llm_factory, logger):
        self._model_client = llm_factory.create_llm_client("solver", model, logger)
        self._system_message = SystemMessage(
            content=("""
You are assisting with a math reasoning problem by providing the next step in the solution 
process. Your explanation should be clear, concise and generating only one extra step.

# Steps

1. Analyze the given math problem and the previous steps provided.
2. Create a clear summary of the previous steps and include them in your response. 
3. Identify the next logical step to progress the solution.
4. Explain the step clearly, showing how it advances the problem solving process.
5. If this step leads to the final answer, present it using the format: 'The answer is #### [numerical answer].'

# Output Guidelines

- Create a clear summary of the previous steps, and include only one additional step in the response.
- Use the final answer format if the solution is complete: 'The answer is #### [numerical answer]'
- Keep your response under 100 words.

# Notes

- Focus on clarity and logical reasoning.
- Ensure continuity by building directly from previous steps."""
            )
        )
        self._prompt = (
            "Now given the following math problem and previous steps, generate the next step."
            "Problem: {content}\n"
            "Previous steps: {prev_steps}\n"
        )
        self._logger = logger
        self._answer_parser = RegexAnswerParser(llm_factory.create_llm_client("answer_parser", "gpt-4o", logger), logger)

    def run(self, runner):
        while True:
            json_object = runner.get_next_question()
            if json_object is None:
                return
            prev_steps = ''
            actual_answer = None
            for i in range(20):
                content = self._prompt.format(content = json_object['question'], prev_steps = prev_steps)
                model_result = asyncio.run(self._model_client.chat(self._system_message, content))
                actual_answer = asyncio.run(self._answer_parser.parse_answer(model_result.content))
                if actual_answer is not None:
                    self._logger.log(f"Answer: \n{model_result.content}")
                    break
                else:
                    self._logger.log(f"Round {i}:\n{model_result.content}")
                    prev_steps = model_result.content
            if actual_answer is None:
                self._logger.log("Answer: None")
            trueans = re.search(r"#### (\-?\d+(\.\d+)?)", json_object['answer'])
            runner.flush(json_object['question'], json_object['index'], trueans.group(1), actual_answer, self._logger)

def parse_arg():    
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument(
        "--input", type=str, default="data/gsm8k.jsonl",help="input data set with question and answer fields."
    )
    parser.add_argument(
        "--output", type=str, default="log/baseline_output.jsonl",help="result logging, also serves as a checkpoint."
    )
    parser.add_argument(
        "--continue_from_checkpoint", type=bool, default=False,help="when true, skip already answered questions from previous run."
    )
    parser.add_argument(
        "--parallelism", type=int, default=10, help="number of threads to run."
    )
    parser.add_argument(
        "--opensource_base_url", type=str, default="https://api.novita.ai/v3/openai", help="Base URL for open source model requests"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="model to run baseline on."
    )
    parser.add_argument(
        "--max_num_questions", type=int, default=-1, help="maximum number of questions to run, per runner"
    )
    args = parser.parse_args()
    return args

args = parse_arg()
api_key = os.environ.get('API_KEY')
os_api_key = os.environ.get('OPEN_SOURCE_API_KEY')
llm_factory = LLMModelClientFactory(api_key, args.opensource_base_url, os_api_key)
runner = MathRunner(args.input, args.output, args.continue_from_checkpoint, args.max_num_questions)
workers = []
for _ in range(args.parallelism):
    logger = Logger()
    workers.append(BaselineWorker(args.model, llm_factory, logger))
runner.parallel_run(workers)






