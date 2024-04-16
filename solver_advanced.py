# Enzo Pelous           2306632
# Pierre Gallou-Guyot   2227094

from utils import *
import random
from copy import deepcopy
from time import time
import random
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
    Génère des voisins d'une solution en considérant tous les échanges de colonnes de la matrice costumes chorégraphies.

    Args:
        sol (CustomSolution) : la solution actuelle du problème
    
    Yields:
        operation (Tuple[int, int]) : les indices des colonnes col1 et col2 échangées dans mat_cop
        mat_cop (np.ndarray) : une copie de la matrice costumes chorégraphies de sol dont les colonnes col1 et col2 ont été échangées
    """
    cc_mat = sol.raw
    
    for col1, col2 in combinations(range(cc_mat.shape[1]), 2):
        mat_cop = cc_mat.copy()
        mat_cop[:, [col2, col1]] = mat_cop[:, [col1, col2]]

        yield (col1, col2), mat_cop


def backstage_neighborhood(sol: CustomSolution):
    """
    Génère des voisins d'une solution en considérant toutes les inversions des 1 facultatifs et des 0 de colonnes non pleines.

    Args:
        sol (CustomSolution) : la solution actuelle du problème
    
    Yields:
        repr (List[int, int, int]) : les coordonées i,j de l'élément à inverser, et sa valeur (0 ou 1)
        mat_cop (np.ndarray) : une copie de la matrice costumes chorégraphies de sol dont l'élément i,j a été changé en 1 ou en 0
    """
    cc_mat = sol.raw

    # On ajoute les indices dans la matrice où on peut ajouter un costume (chorégraphies qui n'ont pas encore D danseurs)
    admissible_indices = [coord + [0]
                          for coord in np.transpose(np.nonzero((np.sum(cc_mat, axis=0) - sol.n_dancers)*(1-cc_mat))).tolist()]

    # On ajoute les indices des costumes facultatifs pour chaque chorégraphie (costumes en coulisse)
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


def perturbation(instance: Instance, sol: CustomSolution, t2: float) -> CustomSolution:
    """
    Perturbe une solution en inversant jusqu'à 20% des éléments de la matrice chorégraphies-costumes d'une solution.
    Le vecteur des chorégraphies est mis à jour au besoin.
    Si on ne trouve pas de perturbation valide pendant 5 secondes, on garde la matrice actuelle.

    Args:
        instance (Instance) : l'instance du problème à résoudre
        sol (CustomSolution) : la solution actuelle du problème
        t2 (float) : le temps d'horloge à la fin de la phase de recherche locale

    Returns:
        sol (CustomSolution) : la solution perturbée
    """
    sol_size = sol.raw.size
    max_perturbation_size = int(sol_size*0.2) #np.random.randint(1, int(sol_size*0.5))

    valid = False
    while not valid and time() - t2 < 5:
        sol_cop = deepcopy(sol)
        perturbation_size = np.random.randint(2, max_perturbation_size+1) # Borne supérieure exclue avec randint
        indices = np.random.choice(sol_size, perturbation_size)
        for index in indices:
            idx = np.unravel_index(index, sol_cop.raw.shape)
            sol_cop.raw[idx] *= -1
            sol_cop.raw[idx] += 1
        valid = instance.is_valid(sol_cop)

    if valid: # On retourne la solution sans modification si on n'a pas trouvé pendant 5 secondes
        # Mise à jour du vecteur choreographies_order de la nouvelle solution, avec assignation bipartite entre les colonnes
        # de raw et celles de la matrice initiale.
        temp_sol = Solution(sol_cop.raw)

        adj_matrix = np.zeros((len(instance.choreography2costume_int),len(temp_sol.choreography2costume_int)))
        for i,x in enumerate(instance.choreography2costume_int.values()):
            for j,y in enumerate(temp_sol.choreography2costume_int.values()):
                if x&y==x:
                    adj_matrix[i][j]=1
        
        _, col_ind = linear_sum_assignment(-adj_matrix.T)

        sol_cop.choreographies_order = col_ind.tolist()

        return sol_cop
    else:
        return sol


def solve(instance: Instance) -> Solution:
    """
    Write here your solution for the homework
    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0

    if 'A' in instance.filepath:
        time_credit = 120
    elif 'D' in instance.filepath:
        time_credit = 900
    else:
        time_credit = 600

    sol = CustomSolution(instance.costumes_choreographies_matrix, n_dancers = instance.n_dancers)

    neighborhoods = [dance_permutation_neighborhood, backstage_neighborhood]
    alpha = 0.35

    best_sol = sol
    best_cost = instance.objective_score(sol)
    
    while ((time()-t0) + iteration_duration) < time_credit - 5:
        # Instant du début de l'itération
        t1 = time()

        k = 0
        while k < 2:
            # Sélection aléatoire dans le voisinage actuel
            operation, _ = random.choice(list(neighborhoods[k](sol)))

            new_sol = deepcopy(sol)
            if k == 0: # permutation des danses
                dance1, dance2 = operation
                new_sol.raw[:, [dance2, dance1]] = new_sol.raw[:, [dance1, dance2]]
                new_sol.choreographies_order[dance2], new_sol.choreographies_order[dance1] = new_sol.choreographies_order[dance1], new_sol.choreographies_order[dance2]
            else:      # costume en coulisses
                row, col, flip = operation
                new_sol.raw[row, col] = 1 - flip
            

            # VND à partir de new_sol pour obtenir un minimum local pour tous les voisinages
            kvnd = 0
            while kvnd < 2:
                vnd_operations_and_scores = [[s[0], instance.objective_score(Solution(s[1]))] # [encodage d'opération, score]
                                            for s in neighborhoods[kvnd](new_sol)]
                
                vnd_optimal_operation, _ = vnd_operations_and_scores[np.argmin([o_s[1] for o_s in vnd_operations_and_scores])]

                vnd_new_sol = deepcopy(new_sol)
                if kvnd == 0: # permutation des danses
                    dance1, dance2 = vnd_optimal_operation
                    vnd_new_sol.raw[:, [dance2, dance1]] = vnd_new_sol.raw[:, [dance1, dance2]]
                    vnd_new_sol.choreographies_order[dance2], vnd_new_sol.choreographies_order[dance1] = vnd_new_sol.choreographies_order[dance1], vnd_new_sol.choreographies_order[dance2]
                else:      # costume en coulisses
                    row, col, flip = vnd_optimal_operation
                    vnd_new_sol.raw[row, col] = 1 - flip

                vnd_new_cost = instance.objective_score(vnd_new_sol)
                new_cost = instance.objective_score(new_sol)
                
                if vnd_new_cost < new_cost:
                    new_sol = vnd_new_sol
                    kvnd = 0
                else:
                    kvnd += 1
            
            cost = instance.objective_score(sol)
            new_cost = instance.objective_score(new_sol)
            delta = np.sum((new_sol.raw - sol.raw)**2) # Distance entre les deux matrices de solution
            
            # Critère d'acceptation biaisé
            if new_cost < cost + alpha*delta:
                sol = new_sol
                k = 0
            else:
                k += 1
        
        cost = instance.objective_score(sol)
        if cost < best_cost:
            best_sol = sol
            best_cost = cost
        
        t2 = time()
        iteration_duration = t2 - t1

        # ILS : perturbation de la solution avant de recommencer la recherche locale
        sol = perturbation(instance, sol, t2)
    

    return best_sol