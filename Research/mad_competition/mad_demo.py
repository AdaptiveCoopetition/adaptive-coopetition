import argparse
import asyncio
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    ModelFamily,
    SystemMessage,
    UserMessage,
)
import json
import os
import re
import threading
from typing import Callable, Dict, List
from mad_util import *
from mad import *

class MultiAgentDebate:
    def __init__(self, name, llm_factory, model_list):
        self._runtime = SingleThreadedAgentRuntime()    
        self._answers = {}
        self._name = name
        self._logger = Logger()
        asyncio.run(self.setup(llm_factory, model_list))

    def get_name(self):
        return self._name
        
    # model_list: list of model names
    async def setup(self, llm_factory, model_list) :
        # List of the name of the solvers
        solvers = [] 
        for i in range(len(model_list)):
            solvers.append("MathSolver"+str(i))
            print("Mathsolver"+str(i)+" refers to "+model_list[i])

        print(f"WSJ: mad_demo.py, MultiAgentDebate, setup, model_list size={len(model_list)}")

        model_client = llm_factory.create_client(model_list[0])
        await MathSolver.register(
                self._runtime,
                # solver name
                solvers[0],
                lambda: MathSolver(model_client=model_client, topic_type=solvers[0],
                                    num_neighbors=2, max_round=5, 
                                   logger = self._logger),
            )

        model_client = llm_factory.create_client(model_list[1])
        await MathSolver.register(
                self._runtime,
                # solver name
                solvers[1],
                lambda: MathSolver(model_client=model_client, topic_type=solvers[1],
                                    num_neighbors=2, max_round=5, 
                                   logger = self._logger),
            )

        model_client = llm_factory.create_client(model_list[2])
        await MathSolver.register(
                self._runtime,
                # solver name
                solvers[2],
                lambda: MathSolver(model_client=model_client, topic_type=solvers[2],
                                    num_neighbors=2, max_round=5, 
                                   logger = self._logger),
            )
        
        #count = 0
        #for model in model_list:
        #    current_solver = solvers[count]
        #    print(f"WSJ: mad_demo.py, MultiAgentDebate, setup, count={count}, model={model}, current_solver={current_solver}")
        #   model_client = llm_factory.create_client(model)
        #    await MathSolver.register(
        #        self._runtime,
        #        # solver name
        #        current_solver,
        #        lambda: MathSolver(model_client=model_client, topic_type=current_solver,
        #                            num_neighbors=2, max_round=3, 
        #                           logger = self._logger),
        #    )
        #    count = count + 1

        print("Math Solvers are registered.")
        await MathAggregator.register(
            self._runtime,
            "MathAggregator",
            lambda: MathAggregator(num_solvers = len(model_list), logger = self._logger, answers=self._answers)
        )
        print("Math Aggregator is registered")
        
        for i in range(len(model_list)):
            await self._runtime.add_subscription(TypeSubscription(solvers[i], solvers[(i+1) % len(model_list)]))
            self._logger.log(f"{solvers[i]} subscribes to {solvers[(i+1) % len(model_list)]}")
            print(f"{solvers[i]} subscribes to {solvers[(i+1) % len(model_list)]}")
            await self._runtime.add_subscription(TypeSubscription(solvers[i], solvers[(i+2) % len(model_list)]))
            self._logger.log(f"{solvers[i]} subscribes to {solvers[(i+2) % len(model_list)]}")
            print(f"{solvers[i]} subscribes to {solvers[(i+2) % len(model_list)]}")
        print("Math solvers have subscribed to topics");
        
    async def execute(self, index, question): 
        self._runtime.start()
        await self._runtime.publish_message(Question(content=question, index = index), DefaultTopicId())
        await self._runtime.stop_when_idle()
        #await model_client.close()

    def run(self, runner):
        while True:
            json_object = runner.get_next_question()
            if json_object is None:
                return
            index = json_object['index']
            # start the debate of the question of index
            prev_answer_size = len(self._answers)
            print(f"WSj mad_demo.py MultiAgentDebate.run question index={index}; answer_size={prev_answer_size}, execute MAD starts")
            asyncio.run(self.execute(index, json_object['question']))
            print(f"WSj mad_demo.py MultiAgentDebate.run question index={index}; {json_object['answer']} execute MAD ends")
            trueans = re.search(r"#### (\-?\d+(\.\d+)?)", json_object['answer'])
            if prev_answer_size < len(self._answers):
                print(f"WSj mad_demo.py MultiAgentDebate.run index={index}; self._answers size={len(self._answers)}")
                runner.flush(json_object['question'], index, trueans.group(1), self._answers[index], self._logger)
            else:
                print("Error: no answer to flush. Index = "+str(index))
                return

def parse_arg():    
    parser = argparse.ArgumentParser(description="mad")
    parser.add_argument(
        "--opensource_base_url", type=str, default="https://api.novita.ai/v3/openai", help="Base URL for open source model requests"
    )
    parser.add_argument(
        "--input", type=str, default="data/gsm8k.jsonl",help="input data set with question and answer fields."
    )
    parser.add_argument(
        "--output", type=str, default="log/output.jsonl",help="result logging, also serves as a checkpoint."
    )
    parser.add_argument(
        "--filter_output", type=str, default="log/baseline_output.jsonl",help="when specified, only run incorrect questions"
    )
    parser.add_argument(
        "--continue_from_checkpoint", type=bool, default=False, help="when set, skip already answered questions from previous run."
    )
    parser.add_argument(
        "--parallelism", type=int, default=5, help="number of threads to run."
    )
    parser.add_argument(
        "--model_list", type=str, default=" google/gemma-3-27b-it,gpt-4o,deepseek/deepseek-v3-0324", help="list of models to run."
    )
    parser.add_argument(
        "--max_num_questions", type=int, default=-1, help="maximum number of questions to run, per runner"
    )
    args = parser.parse_args()
    return args

args = parse_arg()

OpenAI_api_key = os.environ.get('API_KEY')
os_api_key = os.environ.get('OPEN_SOURCE_API_KEY')

llm_factory = LLMModelClientFactory(OpenAI_api_key, args.opensource_base_url, os_api_key)
runner = MathRunner(args.input, args.output, args.continue_from_checkpoint, args.max_num_questions)
if args.filter_output:
    runner.apply_filter(args.filter_output)
workers = []
count = 0
for _ in range(args.parallelism):
    count += 1
    name = "MAD-"+str(count)
    workers.append(MultiAgentDebate(name, llm_factory, args.model_list.split(",")))
runner.parallel_run(workers)