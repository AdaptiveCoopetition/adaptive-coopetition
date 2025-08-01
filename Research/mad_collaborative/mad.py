import re
from dataclasses import dataclass
from typing import Dict, List
from mad_util import *

from autogen_core import (
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
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

###################################################################
#  Communication
###################################################################
@dataclass
class Question:
    content: str
    index: str


@dataclass
class Answer:
    content: str


@dataclass
class SolverRequest:
    content: str
    question: str
    do_reset: bool


@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int


@dataclass
class FinalSolverResponse:
    solver: str
    answer: str

###################################################################
#  Math Solver
###################################################################
@default_subscription
class MathSolver(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, topic_type: str, num_neighbors: int, max_round: int, logger) -> None:
        super().__init__("A debator.")
        self._logger = logger
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateSolverResponse]] = {}
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant with expertise in mathematics and reasoning. "
                    "Your task is to assist in solving a math reasoning problem by providing "
                    "a clear and detailed solution. Limit your output within 100 words, "
                    "and your final answer should be a single numerical number, "
                    "in the form of {{answer}}, at the end of your response. "
                    "For example, 'The answer is {{42}}.'"
                )
            )
        ]
        self._round = 0
        self._max_round = max_round
        print(f"mad.py MathSolver {self.id} init topic_type={self._topic_type}")

    def reset(self) -> None:
        self._round = 0
        self._history.clear()
        self._buffer.clear()

    @message_handler
    async def handle_request(self, message: SolverRequest, ctx: MessageContext) -> None:
        if message.do_reset:
            print(f"MathSolver {self.id}, topic_type is {self._topic_type}, handle_request. reset fisrt.")
            self.reset()
        print(f"MathSolver {self.id}, topic_type is {self._topic_type}, handle_request, round={self._round} starts")
        # Add the question to the memory.
        self._history.append(UserMessage(content=message.content, source="user"))
        # Make an inference using the model.
        model_result = await self._model_client.create(self._system_messages + self._history)
        assert isinstance(model_result.content, str)
        # Add the response to the memory.
        self._history.append(AssistantMessage(content=model_result.content, source=self.metadata["type"]))
        print(f"{'-'*80}\nSolver {self.id} round {self._round}:\n{model_result.content}")
        self._logger.log(f"{'-'*80}\nSolver {self.id} round {self._round}:\n{model_result.content}")
        # Extract the answer from the response.
        match = re.search(r"\{\{(\-?\d+(\.\d+)?)\}\}", model_result.content)
        if match is None:
            print(f"The model response does not contain the answer. model result is {model_result.content}")
            answer=''
        else:
            answer = match.group(1)
        # Increment the counter.
        self._round += 1
        if self._round > self._max_round:
            print(f"Error max_round violation")
        elif self._round == self._max_round:
            # If the counter reaches the maximum round, publishes a final response.
            print(f"WSJ mad.py MathSolver {self.id}, topic_type={self._topic_type}, handle_request max_round is reached. Publish final answer = {answer}.")
            solver_name="MathSolver"+str(self.id)
            await self.publish_message(FinalSolverResponse(answer=answer, solver=solver_name), topic_id=DefaultTopicId())
        else:
            # Publish intermediate response to the topic associated with this solver.
            print(f"WSJ mad.py MathSolver{self.id}, topic_type={self._topic_type}, handle_request publish intermediate response answer={answer}.")
            await self.publish_message(
                IntermediateSolverResponse(
                    content=model_result.content,
                    question=message.question,
                    answer=answer,
                    round=self._round,
                ),
                topic_id=DefaultTopicId(type=self._topic_type),
            )

    @message_handler
    async def handle_response(self, message: IntermediateSolverResponse, ctx: MessageContext) -> None:
        # Add neighbor's response to the buffer.
        print(f"WSJ mad.py MathSolver{self.id}, topic_type={self._topic_type}, handle_response, round={self._round} ")
        self._buffer.setdefault(message.round, []).append(message)
        # Check if all neighbors have responded.
        if len(self._buffer[message.round]) == self._num_neighbors:
            print(
                f"{'-'*80}\nSolver {self.id} round {message.round}:\nReceived all responses from {self._num_neighbors} neighbors."
            )
            self._logger.log(f"{'-'*80}\nSolver {self.id} round {message.round}:\nReceived all responses from {self._num_neighbors} neighbors.")
            # Prepare the prompt for the next question.
            prompt = "These are the solutions to the problem from other agents:\n"
            for resp in self._buffer[message.round]:
                prompt += f"One agent solution: {resp.content}\n"
            prompt += (
                "Using the solutions from other agents as additional information, "
                "can you provide your answer to the math problem? "
                f"The original math problem is {message.question}. "
                "Your final answer should be a single numerical number, "
                "in the form of {{answer}}, at the end of your response."
            )
            # Send the question to the agent itself to solve.
            await self.send_message(SolverRequest(content=prompt, question=message.question, do_reset=False), self.id)
            # Clear the buffer.
            self._buffer.pop(message.round)

###################################################################
#  Math Aggregator
###################################################################
@default_subscription
class MathAggregator(RoutedAgent):
    def __init__(self, num_solvers: int, logger, answers) -> None:
        super().__init__("Math Aggregator")
        print(f"WSJ mad.py MathAggregator.init num_solvers={num_solvers}")
        self._num_solvers = num_solvers
        self._logger = logger
        self._answers = answers
        self._question_index = -1
        self._buffer: List[FinalSolverResponse] = []

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        print(f"{'-'*80}\nAggregator {self.id} received question (index={message.index}):\n{message.content}")
        self._logger.log(f"{'-'*80}\nAggregator {self.id} received question (index={message.index}):\n{message.content}")
        self._question_index = message.index
        prompt = (
            f"Can you solve the following math problem?\n{message.content}\n"
            "Explain your reasoning. Your final answer should be a single numerical number, "
            "in the form of {{answer}}, at the end of your response."
        )
        print(f"{'-'*80}\nAggregator {self.id} publishes initial solver request.")
        self._logger.log(f"{'-'*80}\nAggregator {self.id} publishes initial solver request.")
        await self.publish_message(SolverRequest(content=prompt, question=message.content, do_reset=True), topic_id=DefaultTopicId())

    @message_handler
    async def handle_final_solver_response(self, message: FinalSolverResponse, ctx: MessageContext) -> None:
        print(f"mad.py MathAggregator.handle_final_solver_response solver={message.solver}")
        self._buffer.append(message)
        if len(self._buffer) == self._num_solvers:
            print(f"{'-'*80}\nAggregator {self.id} received all final answers from {self._num_solvers} solvers.")
            self._logger.log(f"{'-'*80}\nAggregator {self.id} received all final answers from {self._num_solvers} solvers.")
            # Find the majority answer.
            answers = [resp.answer for resp in self._buffer]
            majority_answer = max(set(answers), key=answers.count)
            # Publish the aggregated response.
            print(f"mad.py MathAggregator.handle_final_solver_response majority_answer={majority_answer}. Append to answers.")
            if self._question_index>=0:
                self._answers[self._question_index] = majority_answer
            else:
                print(f"mad.py MathAggregator.handle_final_solver_response invalid question index={self._question_index}")
            #await self.publish_message(Answer(content=majority_answer), topic_id=DefaultTopicId())
            # Clear the responses.
            self._buffer.clear()
            print(f"{'-'*80}\nAggregator {self.id} publishes final answer:\n{majority_answer}")
            self._logger.log(f"{'-'*80}\nAggregator {self.id} publishes final answer:\n{majority_answer}")
