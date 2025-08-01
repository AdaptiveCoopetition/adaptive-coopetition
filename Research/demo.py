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
from mathcontroller import MathAggregator
from mathmessage import (
    Question,
    SolverRequest,
    SolverResponse,
)
from mathworker import MathSolver
import os
from prm import MathPRM
import re
import threading
from typing import Callable, Dict, List
from util import *

class MultiAgentDebateWorker:
    def __init__(self, llm_factory, prm_addr, prm_api_key, model_list, strategy, do_sampling, fix_ucb):
        self._runtime = SingleThreadedAgentRuntime()    
        self._answers = {}
        self._logger = Logger()
        asyncio.run(self.setup(llm_factory, prm_addr, prm_api_key, model_list, strategy, do_sampling, fix_ucb))

    async def setup(self, llm_factory, prm_addr, prm_api_key, model_list, strategy, do_sampling, fix_ucb) :
        solvers = [] 
        for i in range(len(model_list)):
            solvers.append("MathSolver"+str(i))
        prm_agent = "MathPRM"
        count = 0
        for model in model_list:
            model_client = llm_factory.create_llm_client(solvers[count], model, self._logger)
            answer_parser = RegexAnswerParser(llm_factory.create_llm_client("answer_parser", "gpt-4o", self._logger), self._logger)
            await MathSolver.register(
                self._runtime,
                solvers[count],
                lambda: MathSolver(model_client=model_client, solvers = solvers, prm_agent = prm_agent,
                                   logger = self._logger, strategy=strategy, answer_parser = answer_parser,
                                   do_sampling=do_sampling, fix_ucb=fix_ucb),
            )
            count = count + 1
        await MathAggregator.register(
            self._runtime,
            "MathAggregator",
            lambda: MathAggregator(solvers = solvers, answers = self._answers, logger = self._logger)
        )
        await MathPRM.register(
            self._runtime,
            prm_agent,
            lambda: MathPRM(openai_api_base = prm_addr, api_key = prm_api_key, logger = self._logger)
        )

    async def execute(self, index, question): 
        self._runtime.start()
        await self._runtime.send_message(Question(content = question, index = index), AgentId("MathAggregator", "default"))
        await self._runtime.stop_when_idle()

    def run(self, runner):
        while True:
            json_object = runner.get_next_question()
            if json_object is None:
                return
            index = json_object['index']
            asyncio.run(self.execute(index, json_object['question']))
            trueans = re.search(r"#### (\-?\d+(\.\d+)?)", json_object['answer'])
            if index < len(self._answers):
                runner.flush(json_object['question'], index, trueans.group(1), self._answers[index], self._logger)
            else:
                print("Error: no answer to flush. Index = "+str(index))
                return

def parse_arg():    
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument(
        "--prm_addr", type=str, default="http://34.125.14.114:8001/v1",
        help="prm address "
    )
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
        "--continue_from_checkpoint", action='store_true',help="when set, skip already answered questions from previous run."
    )
    parser.add_argument(
        "--do_sampling", action='store_true',help="when set, generate multiple answer samples during the critique phase."
    )
    parser.add_argument(
        "--parallelism", type=int, default=5, help="number of threads to run."
    )
    parser.add_argument(
        "--model_list", type=str, default="gpt-4o,deepseek/deepseek-r1,qwen/qwq-32b", help="list of models to run."
    )
    parser.add_argument(
       "--strategy", type=str, choices=["basic", "ucb", "competitive", "collaborative"], default="basic", help="strategy algorithm to use")
    # to use fix_ucb, the strategy must be set as ucb
    parser.add_argument(
        "--fix_ucb", type=str, choices=['comp', 'coll', ''], default='', help="fixed ucb strategy to use: comp is competitive; coll is collaborative"
    )
    parser.add_argument(
        "--max_num_questions", type=int, default=-1, help="maximum number of questions to run, per runner"
    )
    parser.add_argument(
        "--deact_prm", type=bool, default=False, help="whether or not to deactivate PRM usage: true to not call PRM server, false to continue calling PRM server (default)"
    )
    args = parser.parse_args()
    return args

args = parse_arg()
if args.fix_ucb != '' and args.strategy != 'ucb':
    print("When the UCB strategy is fixed, the strategy algorithm must be defined as UCB. Please add --strategy='ucb' to your command.")
elif args.deact_prm and args.fix_ucb == '':
    print("When PRM is deactivated, the strategy algorithm must be defined as a fixed UCB strategy. Please add both --strategy='ucb' and either --fix_ucb='comp' or --fix_ucb='coll' to your command.")
else:
    OpenAI_api_key = os.environ.get('API_KEY')
    os_api_key = os.environ.get('OPEN_SOURCE_API_KEY')
    prm_api_key = os.environ.get('PRM_API_KEY')
    if args.deact_prm:
        prm_api_key = ""
        print("Info: the PRM key is invalidated.") 
    llm_factory = LLMModelClientFactory(OpenAI_api_key, args.opensource_base_url, os_api_key)
    runner = MathRunner(args.input, args.output, args.continue_from_checkpoint, args.max_num_questions)
    if args.filter_output:
        runner.apply_filter(args.filter_output)
    workers = []
    for _ in range(args.parallelism):
        workers.append(MultiAgentDebateWorker(llm_factory, args.prm_addr, prm_api_key, args.model_list.split(","), args.strategy, args.do_sampling, args.fix_ucb))
    runner.parallel_run(workers)

OpenAI_api_key = os.environ.get('API_KEY')
os_api_key = os.environ.get('OPEN_SOURCE_API_KEY')
prm_api_key = os.environ.get('PRM_API_KEY')
print(OpenAI_api_key)
llm_factory = LLMModelClientFactory(OpenAI_api_key, args.opensource_base_url, os_api_key)
runner = MathRunner(args.input, args.output, args.continue_from_checkpoint)
if args.filter_output:
    runner.apply_filter(args.filter_output)
workers = []
for _ in range(args.parallelism):
    workers.append(MultiAgentDebateWorker(llm_factory, args.prm_addr, prm_api_key, args.model_list.split(","), args.strategy, args.do_sampling))
runner.parallel_run(workers)
