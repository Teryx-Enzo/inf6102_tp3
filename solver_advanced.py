from utils import *
import random
from copy import deepcopy


from itertools import permutations


class Individu():
    """
    Individu que nous allons utiliser l'algorithme genetique
    """

    def __init__(self, ordre, matrix, n_dancers, n_costumes):
        """
        Args:
           ordre (List): la liste des chorégraphies dans un certain ordre
           matrix (List[List]) : matrice de correspondance costume/chore
           nombre_danceurs (Int) : nombre de danceurs dans l'instance

        """
        self.max = max(ordre)
        self.ordre = ordre
        self.matrix = dict()
        for i,ligne in enumerate(matrix):
            self.matrix[i] = ligne


        self.n_dancers = n_dancers
        self.n_costumes = n_costumes

    def mutation(self):
        """
        Change, aléatoirement, une des chorégraphie en ajoutant, si possible, un costume porté (en espérant éviter un changement)
        """
        change = False
        i = 0
        while not change and i<10:

            mutation_index = int(np.random.uniform(0,self.max))
            if np.count_nonzero(self.matrix[mutation_index])<self.n_dancers:
                dance_mutation_index = int(np.random.uniform(0,self.n_costumes))
                self.matrix[mutation_index][dance_mutation_index] = 1
                change = True
            i+= 1
    def return_sol(self):
        sol = [self.matrix[i] for i in self.ordre]

        print(sol)
        return np.array(sol)
    
    def scored(self, instance):


        score = instance.objective_score(Solution(self.return_sol()))
        print(score)
        return score

def crossover(indiv1, indiv2):
     
    """
    Args:
        indiv1 : l'individu dont on veut conserver l'ordre des choregrapie
        indiv2 : l'individu dont on veut conserver le choix des constumes
    """
    indiv1.matrix = indiv2.matrix.copy()

    return deepcopy(indiv1)

def generate_pop(pop_size, ordre_initial,matrice_initiale,n_dancers,n_costumes):

    """
        Args:
            pop_size (Int) : taille de la population
            ordre_initial (List): la liste des chorégraphies dans l'odre initial
            matrice_initiale (List[List]) : matrice de correspondance costume/chore
            n_dancers (Int) : nombre de danceurs de l'instance
            n_costumes (Int) : nombre de costumes de l'instance

        Returns:
            population (List[Individu]) : liste d'individu
        """

    population = [Individu(sorted(ordre_initial, key=lambda k: random.random()), matrice_initiale, n_dancers, n_costumes) for _ in range(pop_size)]



    return population

class CustomSolution(Solution):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def solve(instance: Instance) -> Solution:
    """
    Write here your solution for the homework
    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """

    n_dancers, n_costumes = instance.n_dancers, instance.ncostumes
    matrice_initiale = instance.costumes_choreographies_matrix.copy()
    ordre_initial = [i for i in range(len(matrice_initiale))]


    pop_size = 10
    gen_max = 10

    population = generate_pop(pop_size, ordre_initial,matrice_initiale,n_dancers,n_costumes)
    
    # for i,x in enumerate(permutations(instance.choreographies_costumes_matrix)):
    #     if i>500:
    #         break
    #     sol = np.array(x).T
    #     if instance.objective_score(Solution(sol))<best_score:
    #         best_sol,best_score=sol,instance.objective_score(Solution(sol))

    for _ in range(gen_max):

        for indiv in population:
            indiv.mutation()


    pop_finale_sorted = sorted(population, key=lambda x: x.scored(instance), reverse=True)

    best_sol = pop_finale_sorted[0].return_sol()
    return Solution(best_sol)