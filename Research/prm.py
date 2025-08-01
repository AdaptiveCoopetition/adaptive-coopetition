from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
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
from mathmessage import *
import random
import torch
from openai import OpenAI
from transformers import AutoTokenizer
from util import *

# pip install vllm
# vllm serve Qwen/Qwen2.5-Math-PRM-7B --host 0.0.0.0  --port 8001 --enforce-eager --dtype half --task reward

@default_subscription
class MathPRM(RoutedAgent):
    def __init__(self, openai_api_base: str, api_key: str, logger: Logger) -> None:
        super().__init__("An evaluator.")
        self._client = OpenAI(api_key=api_key, base_url=openai_api_base)
        self._logger = logger
        self._model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        self._api_key = api_key

    @message_handler
    async def handle_request(self, message: PRMRequest, ctx: MessageContext) -> None:
        if self._api_key == "":
            prob = [0.8]
            self._logger.log(f"Used fixed PRM result (PRM key is invalid): {prob}")
            prm_response = PRMResponse(probabilities = prob, agent = message.agent)
            await self.send_message(prm_response, AgentId(ctx.sender.type, "default"))
            return
        messages = [
            {
                "role": "system", 
                "content": "Please reason step by step, and put the answer in the form of #### answer at the end of your response. For example, 'The answer is #### 42.'"
            },
            {"role": "user", "content": message.question},
            {"role": "assistant", "content": "<extra_0>".join(message.response) + "<extra_0>"},
        ]
        conversation_str = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        try:
            responses = self._client.embeddings.create(input=conversation_str, model=self._model_name)
            assert len(responses.data) == 1
            data = responses.data[0]
            data = torch.tensor(data.embedding).view(-1, 2)
            prob = data[:,1].tolist()
        except Exception as e:
            self._logger.log_error(f"{self.id.type} Encountered {e} with {conversation_str}")
            prob = [random.random()]
        self._logger.log(f"PRM's Positive probability for {message.agent} is: {prob}")
        prm_response = PRMResponse(probabilities = prob, agent = message.agent)
        await self.send_message(prm_response, AgentId(ctx.sender.type, "default"))




