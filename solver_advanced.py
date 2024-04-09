from utils import *
import random
from copy import deepcopy
import random
from tqdm import tqdm
from itertools import combinations


class CustomSolution(Solution):
    """ 
    You are completely free to extend classes defined in utils,
    this might prove useful or enhance code readability.
    """

    def __init__(self, *args, n_dancers, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_cc_mat = self.raw.copy()

        self.choreographies_order = list(self.choreographies_dict.keys())

        self.n_dancers = n_dancers


def dance_permutation_neighborhood(sol: CustomSolution):
    """
    """
    cc_mat = sol.raw
    
    for col1, col2 in combinations(range(cc_mat.shape[1]), 2):
        mat_cop = cc_mat.copy()
        mat_cop[:, [col2, col1]] = mat_cop[:, [col1, col2]]

        yield (col1, col2), mat_cop


def backstage_neighborhood(sol: CustomSolution):
    """
    """
    cc_mat = sol.raw

    # On ajoute les indices dans la matrice où on peut ajouter un costume (chorégraphies qui n'ont pas encore D danseurs)
    admissible_indices = [coord + [0]
                          for coord in np.transpose(np.nonzero((np.sum(cc_mat, axis=0) - sol.n_dancers)*(1-cc_mat))).tolist()]

    # 1 facultatifs à l'aide de self.choreographies_order
    for i, c in enumerate(sol.choreographies_order):
        admissible_indices += [[idx, i, 1] for idx in np.nonzero((1- sol.initial_cc_mat[:, c]) * cc_mat[:, i])[0].tolist()]

    for repr in admissible_indices:
        mat_cop = cc_mat.copy()
        if repr[2] == 0: # 0 changeable en 1
            mat_cop[repr[0], repr[1]] = 1

            yield repr, mat_cop
        
        else: # 1 changeable en 0
            mat_cop[repr[0], repr[1]] = 0
            yield repr, mat_cop


def solve(instance: Instance) -> Solution:
    """
    Write here your solution for the homework
    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """

    n_dancers, n_costumes = instance.n_dancers, instance.ncostumes
    sol = CustomSolution(instance.costumes_choreographies_matrix, n_dancers = n_dancers)

    neighborhoods = [dance_permutation_neighborhood, backstage_neighborhood]

    k = 0

    while k < 2:
        # s' = argmin du voisinage actuel
        operations_and_scores = [[s[0], instance.objective_score(Solution(s[1]))] # [encodage d'opération, score]
                                 for s in neighborhoods[k](sol)]
        
        #print(operations_and_scores)
        optimal_operation, _ = operations_and_scores[np.argmin([o_s[1] for o_s in operations_and_scores])]

        if k == 0: # permutation des danses
            dance1, dance2 = optimal_operation
            new_sol = deepcopy(sol)
            new_sol.raw[:, [dance2, dance1]] = new_sol.raw[:, [dance1, dance2]]
            new_sol.choreographies_order[dance2], new_sol.choreographies_order[dance1] = new_sol.choreographies_order[dance1], new_sol.choreographies_order[dance2]
        else:      # costume en coulisses
            row, col, flip = optimal_operation
            new_sol = deepcopy(sol)
            new_sol.raw[row, col] = 1 - flip
            

        # s,k = neighborhood_change(s, s', k)
        if instance.objective_score(new_sol) < instance.objective_score(sol):
            sol = new_sol
            k = 0
        else:
            k += 1
    

    return sol