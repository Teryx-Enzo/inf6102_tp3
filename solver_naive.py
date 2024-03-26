from utils import *
from itertools import permutations

def solve(instance: Instance) -> Solution:
    """
    This function generates the best shuffling of the choreographies amongst 500 permutations

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution, this is always a valid solution
    """
    best_sol, best_score = instance.costumes_choreographies_matrix, instance.objective_score(Solution(instance.costumes_choreographies_matrix))
    for i,x in enumerate(permutations(instance.choreographies_costumes_matrix)):
        if i>500:
            break
        sol = np.array(x).T
        if instance.objective_score(Solution(sol))<best_score:
            best_sol,best_score=sol,instance.objective_score(Solution(sol))

    return Solution(best_sol)
