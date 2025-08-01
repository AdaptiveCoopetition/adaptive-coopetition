from dataclasses import dataclass
from enum import Enum

@dataclass
class Question:
    content: str
    index : int 


@dataclass
class Answer:
    content: str

#list of valid actions
@dataclass
class Action(Enum):
    NEXT_STEP = 1 # solve problems step by step- do the next step
    CRITIQUE = 2  # critique a passed in peer's reponse
    NEXT_STEP_WITH_FEEDBACK = 3 # do the next step with a peer's feedback
    NEXT_STEP_MERGE = 4 # generate the next step by combining self response and a peer's response 
    NEXT_STEP_PICK_FROM_CANDIDATES = 5 # pick from a list of N candidates
    
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash(self.value)

@dataclass
class SolverRequest:
    # original question
    question: str

    # question index
    index: int

    iteration: int 


@dataclass
class SolverResponse:
    # partial repsonse for the question
    response: str

    # final answer if available
    answer: str
    prm_score: float

@dataclass
class PRMRequest:
    question: str 
    response: list[str]
    agent: str

@dataclass
class PRMResponse:
    probabilities: list[float]
    agent: str

@dataclass
class CritiqueRequest:
    question: str
    peer_response: str

@dataclass
class CritiqueResponse:
    question: str
    critique: str
