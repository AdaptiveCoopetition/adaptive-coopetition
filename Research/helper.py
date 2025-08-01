from enum import Enum

class MathSolverActions(Enum):
    GREEDY = 1
    NONGREEDY = 2
    EXPLORE = 3
    EXPLOIT = 4

def determine_strategy(PrmScore: List[float]) -> MathSolverActions, Tuple[float, float] :
    """
    Given list of PRM scores, select one of the MathSolverActions and its corresponding ML model temperature range.

    Args:
        PrmScore (List[float]): the PRM Score of each step

    Returns:
        MathSolverActions: the predefined MathSolverActions
        Tuple[float, float]: (lower_bound, upper_bound)
        lower_bound: lower bound of the temperature range
        upper_bound: upper bound of the temperature range
    """
  if PrmScore[0] <= 0.5: # collaborative
    return MathSolverActions.NONGREEDY, (0.1, 0.3) # temperature ranges have placeholder values
  else : # competitive
    return MathSolverActions.GREEDY, (0.7, 0.9) # temperature ranges have placeholder values
