from autogen_core import (
    AgentId,
    DefaultTopicId,
    default_subscription,
    MessageContext,
    message_handler,
    RoutedAgent,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient, 
    LLMMessage,
    SystemMessage,
    UserMessage,
)
import copy
from mathmessage import *
import random
from util import *

@default_subscription
class MathAggregator(RoutedAgent):
    def __init__(self, solvers, answers, logger) -> None:
        super().__init__("Math Aggregator")
        self._solvers = solvers
        self._answers = answers
        self._logger = logger
        self._max_rounds = 20
        self._target_rounds = 5
        self.reset()

    # call this to prepare for the next question.
    def reset(self):
        # map from solver to response.
        self._buffer = {}
        self._agent_answers = {}
        self._rounds = 0
        self._question = None
        self._disagreement = False

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        if self._question is None:
            self._logger.log_error(f"{'-'*80}\nAggregator {self.id} received question {message.index}:\n {message.content}")
            self._question = message
        else:
            self._logger.log_error(f"{'-'*80}\nAggregator {self.id} publishes solver request for question {message.index} , round = {self._rounds}, answers={len(self._agent_answers)}.")
        request = SolverRequest(question=message.content, index = message.index, iteration = self._rounds)
        await self.publish_message(request,  topic_id=DefaultTopicId())

    @message_handler
    async def handle_solver_response(self, message: SolverResponse, ctx: MessageContext) -> None:
        sender = ctx.sender.type
        self._buffer[sender] = message
        if len(self._buffer) == len(self._solvers):
            self._logger.log(f"{'-'*80}\nAggregator {self.id} received responses from all {len(self._solvers)} solvers.")
            await self.next_round()
    
    async def next_round(self):
        for worker, response in self._buffer.items():
            if response.answer is not None:
                self._agent_answers[worker] = response.answer
                self._logger.log(f"{'-'*80}\nAgent {worker} reached answer {response.answer}.")

        majority_answer, count = self.majority_answer()
        if (not self._disagreement and count == len(self._solvers) and self._rounds >= 2) or (self._rounds >= self._target_rounds and majority_answer is not None):
            self._answers[self._question.index] = majority_answer
            self._logger.log(f"{'-'*80}\nAggregator {self.id} publishes final answer:\n{majority_answer}, solver answers: {self._agent_answers}\n")
            self.reset()
        elif self._rounds >= self._max_rounds:
            self._logger.log(f"Cannot reach final answer after 10 debating rounds, solver answers: {self._agent_answers} ")
            self._answers[self._question.index] = None
            self.reset()
        else:
            self._rounds = self._rounds + 1
            self._buffer.clear()
            await self.send_message(self._question, self.id)

    def majority_answer(self):
        if len(self._agent_answers) < len(self._solvers):
            return None, None
        answers = list(self._agent_answers.values())
        majority_answer = max(set(answers), key=answers.count)
        count = answers.count(majority_answer)
        if count < len(self._solvers):
            self._disagreement = True
        if count > len(self._solvers) / 2:
            return majority_answer, count
        return None, None

