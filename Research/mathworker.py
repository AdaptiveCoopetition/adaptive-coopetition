from autogen_core import (
    AgentId,
    DefaultTopicId,
    default_subscription,
    MessageContext,
    message_handler,
    RoutedAgent,
)
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from mathmessage import SolverRequest, CritiqueRequest, CritiqueResponse, SolverResponse, PRMRequest, PRMResponse, Action
import random
import re
import time
from util import *

@default_subscription
class MathSolver(RoutedAgent):
    """MathSolver, references:
    1. https://platform.openai.com/docs/guides/prompt-generation for prompt formatting and examples.
    2. https://proceedings.neurips.cc/paper_files/paper/2024/file/32e07a110c6c6acf1afbf2bf82b614ad-Paper-Conference.pdf for the collaborative and competitive strategies.
    
    Our improvements: 
    1) We introduce PRM scoring and information diversity as two measures to stablize the final output during the debate. To be specific, workers will decide their action based on the strategy decided by those two measures, instead of answering yes or no and directly changing the output.
    2) Instead of using a shared-all information model, we use randomly selected information from the peers. (Need to confirm further)
    """
    def __init__(self, model_client: ChatCompletionClient, solvers: list[str], prm_agent: str, logger: Logger,
                 strategy:str, answer_parser: RegexAnswerParser, do_sampling: bool, fix_ucb: str) -> None:
        super().__init__("A debator.")
        self._model_client = model_client
        self._logger = logger
        self._solvers = solvers.copy()
        self._solvers.remove(self.id.type)
        self._prm_agent = prm_agent
        self._fix_ucb = fix_ucb
        self._prompts = {
            Action.NEXT_STEP: (
                SystemMessage(content=("""
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
                ),
                (
                    "Now given the following math problem and previous steps, add the next step."
                    "Problem: {content}\n"
                    "Previous steps: {prev_steps}\n"
                )
            ),
            Action.CRITIQUE: (
                SystemMessage(content=("""
Your task is to review a partial solution to a math problem and identify any errors.

# Steps

1. **Understand the Problem**: read and comprehend the math reasoning problem.
2. **Review the Partial Solution**: Check for mistakes in logic or calculation.
3. **Critique**: explain any errors found clearly.

# Output Format

- Provide a concise critique to the partial solution; do not provide the final answer in the response.
- Keep your responses under 100 words.

# Notes

- Focus on accuracy in identifying mistakes
- Ensure your explanation is clear and to the point.
""")
                ),
                (
                    "Now given the following math problem and partial solution, please carefully inspect the solution and point out any mistakes."
                    "Problem: {content}\n"
                    "Partial solution: {peer_response}\n"
                )
            ),
            Action.NEXT_STEP_WITH_FEEDBACK: (
                SystemMessage(content=("""
Your task is to review a partial solution and its critique for a math reasoning problem, correct any
errors, and provide the next correct step in the solution.

# Steps

1. **Understand the problem**: read and interpret the math problem.
2. **Review the partial solution**: identify any mistakes or gaps.
3. **Evaluate the Critique**: assess the critique's accuracy.
4. **Address the Critique**: replace the partial solution with a corrected solution. If the final answer hasn’t been reached, provide only the next logical step.

# Output format

- Add only one step per response.
- Clearly explain your reasoning.
- If reaching the final answer, use the format: The answer is #### [numerical answer].
- Limit your response to 100 words.
""")
                ),
                (
                    "Now given the following math problem, previous steps and critique, please carefully consider the critique and correct any mistakes as the next step."
                    "Problem: {content}\n"
                    "Previous steps: {prev_steps}\n"
                    "Critique: {critique}\n"
                )
            ),
            Action.NEXT_STEP_MERGE: (
                SystemMessage(content=("""
You are a math reasoning assistant. Your role is to solve a problem step by step by integrating
the best parts of two given partial solutions.

# Steps

1. Carefully read and understand the math problem.
2. Review both partial solutions thoroughly.
3. Extract and combine the strongest reasoning from each partial solution to create a unified solution.
4. If the final answer hasn’t been reached, provide only the next logical step.

# Output Format

- Rewrite the combined solution. If the final answer is still incomplete, provide just one additional step per response.
- Keep response under 100 words.
- If this step solves the problem, present the answer as: The answer is #### [numerical answer].
""")
                ),
                (
                    "Now given the following math problem, two partial solutions, please generate the next step."
                    "Problem: {content}\n"
                    "solution_1: {solution_1}\n"
                    "solution_2: {solution_2}\n"
                )
            ),
            Action.NEXT_STEP_PICK_FROM_CANDIDATES: (
                SystemMessage(content=("""
You are a math reasoning assistant. Your role is pick the best partial solution for a math reasoning
problem from a list of candidates.

# Steps

1. Carefully read and understand the math problem.
2. Review all partial solutions thoroughly.
3. Return the best partial solution to the problem and summarize the reasoning behind your choice.

# Output Format

- Summarize the reasoning behind your choice.
- Return the best partial solution without any modification.
- Your response should be format as follows within 100 words:
Reason: <your reasoning>
Choice: <the best partial solution>
""")
                ),
                (
                    "Now given the following math problem, {n} partial solutions, please output your choice and reason."
                    "Problem: {content}\n"
                    "Candidate solution:\n{solution_list}"
                )
            ),
        }
        if strategy == 'ucb':
            self._strategy_picker = UCBStrategy(logger, self.id.type)
        elif strategy == 'basic':
            self._strategy_picker = BasicStrategy(logger, self.id.type)
        elif strategy == 'competitive':
            self._strategy_picker = CompetitiveStrategy()
        else:
            self._strategy_picker = CollaborativeStrategy()
        self._question = None
        self._answer_parser = answer_parser
        self._do_sampling = do_sampling

    def maybe_reset(self, question: str, index: int):
        if self._question is None or self._question != question:
            self._question = question
            self._index = index
            self._prev_response = ''
            self._next_strategy = 'Collaborative'
            self._peer_responses = {}
            self._answer = None
            self._strategy_picker.reset()
            self._answer_scores = {}
            self._answer_counts = {}
            self._solver_scores = {}

    @message_handler
    async def handle_solver_request(self, message: SolverRequest, ctx: MessageContext) -> None:
        self.maybe_reset(message.question, message.index)
        if self._next_strategy == 'Collaborative' or self._prev_response == '':
            await self.do_collaborative(message)
        else:
            peer = self.pick_peer_to_critique()
            cr = CritiqueRequest(question = self._question, peer_response = self._prev_response)
            await self.send_message(cr, AgentId(peer, "default"))

    async def do_collaborative(self, message: SolverRequest):
        if len(self._peer_responses) == 0:
            system_message, prompt = self._prompts[Action.NEXT_STEP]
            content = prompt.format(content = message.question, prev_steps = self._prev_response)
        else:
            peer_response = self.pick_peer_response_to_combine()
            system_message, prompt = self._prompts[Action.NEXT_STEP_MERGE]
            content = prompt.format(content = message.question, solution_1 = self._prev_response, solution_2 = peer_response.response)
            self._logger.log(f"Solver {self.id} collaborative prompt: {content}")
        model_result = await self._model_client.chat(system_message, content, 0.3)
        await self.analyze_result(model_result.content)

    async def analyze_result(self, model_result:str):
        self._logger.log(f"{'-'*80}\nSolver {self.id} Answer: \n{model_result}")
        self._answer = await self._answer_parser.parse_answer(model_result)
        self._prev_response = model_result
        await self.compute_PRM_score(self._prev_response)

    @message_handler
    async def handle_critique_request(self, message:CritiqueRequest, ctx:MessageContext) -> None:
        system_message, prompt = self._prompts[Action.CRITIQUE]
        content = prompt.format(content = message.question, peer_response = message.peer_response)
        model_result = await self._model_client.chat(system_message, content)
        cr = CritiqueResponse(question = message.question, critique = model_result.content)
        self._logger.log(f"Solver {ctx.sender.type} received critique from {self.id.type}: \n{model_result.content}")
        await self.send_message(cr, AgentId(ctx.sender.type, "default"))

    @message_handler
    async def handle_critique_response(self, message:CritiqueResponse, ctx:MessageContext) -> None:
        system_message, prompt = self._prompts[Action.NEXT_STEP_WITH_FEEDBACK]
        content = prompt.format(content = message.question, prev_steps = self._prev_response, critique = message.critique)
        self._logger.log(f"Solver {self.id} competitive prompt: {content}")
        if self._do_sampling:
            await self.handle_critique_with_sampling(message.question, system_message, content)
        else:
            await self.handle_critique_without_sampling(system_message, content)

    async def handle_critique_without_sampling(self, system_message, content):
        model_result = await self._model_client.chat(system_message, content)
        self._logger.log(f"{self.id} competitive response: {model_result.content}")
        await self.analyze_result(model_result.content)

    async def handle_critique_with_sampling(self, question, system_message, content):
        solution_list = await self._model_client.multi_chat(system_message, content, 1.0, 5)
        self._logger.log(f"{self.id} candidate next steps: {'\n\n'.join(solution_list)}")
        system_message, prompt = self._prompts[Action.NEXT_STEP_PICK_FROM_CANDIDATES]
        content = prompt.format(content = question, n = len(solution_list), solution_list = '\nCandidate solution:\n'.join(solution_list))
        model_result = await self._model_client.chat(system_message, content, 0.3)
        self._logger.log(f"{self.id} picked next step: {model_result.content}")
        await self.analyze_result(model_result.content)

    @message_handler
    async def handle_solver_response(self, message: SolverResponse, ctx: MessageContext) -> None:
        if message.answer is not None:
            if message.answer in self._answer_scores:
                self._answer_scores[message.answer] = self._answer_scores[message.answer] + message.prm_score
                self._answer_counts[message.answer] = self._answer_counts[message.answer] + 1
            else:
                self._answer_scores[message.answer] = message.prm_score
                self._answer_counts[message.answer] = 1
            self._peer_responses[message.answer] = message
        if ctx.sender.type in self._solver_scores:
            self._solver_scores[ctx.sender.type] = self._solver_scores[ctx.sender.type] + message.prm_score
        else:
            self._solver_scores[ctx.sender.type] = message.prm_score

    async def compute_PRM_score(self, model_result) :
        prm_request = PRMRequest(agent = self.id.type, question = self._question, response = [model_result])
        await self.send_message(prm_request, AgentId(self._prm_agent, "default"))

    @message_handler
    async def handle_prm_response(self, message: PRMResponse, ctx: MessageContext) -> None: 
        self._prm_score = message.probabilities[0]
        self._next_strategy = self._strategy_picker.next_strategy(self._next_strategy, self._prm_score)
        s = SolverResponse(response=self._prev_response, answer =self._answer, prm_score = self._prm_score)
        await self.publish_message(s, topic_id=DefaultTopicId())

    def pick_peer_response_to_combine(self):
        choices = []
        weights = []
        if len(self._answer_scores) > 1:
            for s, v in self._answer_scores.items():
                choices.append(s)
                avg = v / self._answer_counts[s]
                weights.append(avg)
        else:
            for s, _ in self._peer_responses.items():
                choices.append(s)
                weights.append(1)
        biased_choice = random.choices(choices, weights, k = 1)
        self._logger.log(f"Solver {self.id} picked {biased_choice[0]} from {choices}, weights {weights}")
        return self._peer_responses[biased_choice[0]]

    def pick_peer_to_critique(self):
        solver = random.choices(list(self._solver_scores.keys()), list(self._solver_scores.values()), k = 1)
        self._logger.log(f"Solver {self.id} picked {solver[0]} from {self._solver_scores} to critique")
        return solver[0]
