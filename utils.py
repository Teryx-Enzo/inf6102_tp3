from matplotlib.colors import ListedColormap
import colorsys
from scipy.optimize import linear_sum_assignment
import os
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
make_universal = lambda x: os.sep.join(x.split('/'))

def bool2int(x):
    """
        Turns a boolean array x in a integer
    """
    y = 0
    for i,j in enumerate(x[::-1]):
        y += j<<i
    return y

def int2bool(x,dim):
    """
        Turns an integer x into a boolean array
    """
    return np.array([int(x) for x in str(bin(x)[2:].zfill(dim))])



def minimize_differences(x1, x2):
    """
        Reorders the elements of x2 such that the number of occurrences where x1[i]=x2[i] is maximal.
        This is mainly used for plotting.
    """


    # Create a binary cost matrix where the (i, j)-th element is 0 if x1[i] != x2[j], else 1
    cost_matrix = np.equal.outer(x1, x2).astype(int)

    # Solve the linear sum assignment problem to maximize the total matches
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # Reorder x2 based on the optimal assignment
    best_match = x2[col_ind]

    return best_match

class Solution:
    def __init__(self,matrix:np.ndarray):
        """
        Args:
            matrix (np.array): a boolean list A such that A_{i,j}=1 if the costume i is worn during the j-th choreography
        """
        self.raw = np.array(matrix)
        self.choreographies_dict = {i:matrix[:,i].ravel() for i in range(matrix.shape[1])}
        self.choreography2costume_list = {i:np.where(matrix[:,i])[0] for i in range(matrix.shape[1])}
        self.choreography2costume_int = {k:bool2int(v) for k,v in self.choreographies_dict.items()}


class Instance:
    def __init__(self,filepath: str):
        self.filepath = make_universal(filepath)
        with open(self.filepath) as f:
            lines = list([[int(x.strip()) for x in x.strip().split(' ')] for x in f.readlines()])
    
            self.nchoreographies = lines[0][0]
            """The number of choregraphies"""

            self.ncostumes = lines[1][0]
            """The number of costumes"""

            self.n_dancers = lines[2][0]
            """The number of dancers"""
            
            self.costumes_choreographies_matrix = np.array(lines[3:])
            """A (ncostumes)x(nchoregraphies) matrix, `costumes_choreographies_matrix[i][j]=1` when the costume i must appear in the choreography `j` """

            self.choreographies_costumes_matrix = self.costumes_choreographies_matrix.T
            """A (nchoreographies)x(ncostumes) matrix, the transpose of `costumes_choreographies_matrix`"""
            
            self.choreographies_dict = {i:self.costumes_choreographies_matrix[:,i].ravel() for i in range(self.nchoreographies)}
            """A dictionnary of choreographies as 1D line-vectors, `choreographies_dict[i][j]=choreographies_costume_matrix[i][j]` """

            self.choreography2costume_list = {i:np.where(self.costumes_choreographies_matrix[:,i])[0] for i in range(self.costumes_choreographies_matrix.shape[1])}
            """`choreography2costume_list[i]` is a list of indices corresponding to the costumes that the choreography i requires such that `choreographies_costumes_matrix[i][choreography[i][j]]=1` forall `j`. """

            self.choreography2costume_int = {k:bool2int(v) for k,v in self.choreographies_dict.items()}
            """ The binary representation of `choreography2costume_int[i]` is `choreography2costume_list[i]`"""
    
    def generate_distinct_colors(self,num_colors):
        """
            Generates an array of #num_colors colors such that
            the colors are the most distinct possible
        """
        # Generate equally spaced hues
        hues = np.linspace(0, 1, num_colors, endpoint=False)

        # Set constant saturation and value
        saturation = 0.9
        value = 0.9

        # Convert HSV to RGB
        rgb_colors = []
        for hue in hues:
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb_colors.append((r, g, b, .4))
            rgb_colors.append((r, g, b, 1))

        return rgb_colors



    def solution_from_file(self,filepath):
        """
            Loads a solution from a solution file according to the problem statement's specifications
        """
        with open(filepath) as f:
            matrix = np.array(list([[int(x.strip()) for x in x.strip().split(',')] for x in f.readlines()]))
        assert matrix.shape == (self.ncostumes,self.nchoreographies)
        return Solution(matrix)

    def are_choreographies_respected(self,sol:Solution):
        """
            Check if all choreographies requirements are met in the final solution
        """
        adj_matrix = np.zeros((len(self.choreography2costume_int),len(sol.choreography2costume_int)))

        # Create a bipartite graph between the original choreographies and the one found in the solution
        # An edge is created if a solution choreography dominates the original one
        for i,x in enumerate(self.choreography2costume_int.values()):
            for j,y in enumerate(sol.choreography2costume_int.values()):
                if x&y==x:
                    adj_matrix[i][j]=1

        # Find the maximum cardinality bipartite matching
        row_ind, col_ind = linear_sum_assignment(-adj_matrix)
        # Check if the number of matched vertices is equal to the number of original choreographies
        return sum([adj_matrix[a,b] for a,b in zip(row_ind,col_ind)]) == len(self.choreography2costume_list)


    def is_valid(self,sol:Solution)->bool:
        """
        Checks if a solution is valid
        """
        sol = Solution(sol.raw)
        return sol.raw.shape==self.costumes_choreographies_matrix.shape \
               and sum(sol.raw.sum(axis=0)<=self.n_dancers)==self.nchoreographies \
               and self.are_choreographies_respected(sol)
    
    def objective_score(self,sol:Solution)->int:
        """Returns the number of times dancers must put on a new costume for a given solution

        Args:
            sol (Solution): the solution

        Returns:
            int: the number of costume changes
        """
        return self.get_stress_matrix(sol.raw).sum()

    def get_stress_matrix(self,binary_matrix:np.ndarray)->np.ndarray:
        """Returns a matrix A similar to the solution matrix but
           the elements are A_{i,j}=1 when a costume switch occurs

        Args:
            binary_matrix : the solution matrix

        Returns:
            stress_matrix : the stress matrix
        """
        matrix = binary_matrix
        stress_matrix = np.zeros(matrix.shape)

        added_costumes = np.zeros(matrix.shape[0], dtype=bool)
        for j in range(matrix.shape[1]):
            for i in range(matrix.shape[0]):
                if matrix[i, j] and not added_costumes[i]:
                    stress_matrix[i,j]=1
                added_costumes[i] = matrix[i, j]
        
        return stress_matrix

    def solution_matrix2stage_matrix(self,binary_matrix:np.ndarray)->np.ndarray:
        """Turns the binary solution matrix into a (self.n_dancers)x(n_choreographies) matrix
           that gives the sequence of stage configurations across choreographies

        Args:
            binary_matrix (np.ndarray[np.ndarray[int]]): the solution matrix

        Returns:
            np.ndarray[np.ndarray[int]]: the stage matrix, -1 denotes an empty dancer
        """
        matrix = binary_matrix
        for i in range(matrix.shape[0]):
            matrix[i][np.where(matrix[i])[0]]+=i
        stage_matrix = np.array([np.pad(matrix[np.where(matrix[:,i])[0],i],(0,self.n_dancers-len(matrix[np.where(matrix[:,i])[0]])),mode='constant') for i in range(matrix.shape[1])])

        for i in range(1,stage_matrix.shape[0]):
            stage_matrix[i]=minimize_differences(stage_matrix[i-1],stage_matrix[i])
        stage_matrix = stage_matrix.T
        return stage_matrix-1

    def stage_matrix2solution_matrix(self, stage_matrix:np.ndarray)->np.ndarray:
        """Turns the stage matrix back into the solution matrix, reverts the bin_matrix2stage_matrix function

        Args:
            stage_matrix (np.ndarray[np.ndarray[int]]): the stage matrix

        Returns:
            np.ndarray[np.ndarray[int]]: the solution matrix
        """
        solution_matrix = np.zeros((self.ncostumes,self.nchoreographies))
        for a in stage_matrix:
            for choreography_i,b in enumerate(a):
                if b>=0:
                    solution_matrix[int(b)][choreography_i]=1
        return solution_matrix

    def plot_solution(self,solution:Solution):
        """
            Generates a visualization of the solution
        """
        matrix = solution.raw.copy()
        fig, ax = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [9, 4, 2, 0]})

        # Add gridlines to separate colors
        ax[0].set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
        ax[0].set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
        ax[0].grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
        added_colors = np.zeros(matrix.shape[0], dtype=bool)
        colors = ['white']+self.generate_distinct_colors(matrix.shape[0])
        cmap = ListedColormap(colors)

        stress_matrix = np.zeros(matrix.shape)
        for j in range(matrix.shape[1]):
            for i in range(matrix.shape[0]):
                if matrix[i, j] and not added_colors[i]:
                    stress_matrix[i,j]=1
                added_colors[i] = matrix[i, j]
        for i in range(matrix.shape[0]):
            matrix[i][np.where(matrix[i])[0]]+=2*i

        matrix = stress_matrix+matrix
        ax[0].imshow(matrix, cmap=cmap, aspect='auto',alpha=.8)

        ax[0].set_xticks(np.arange(matrix.shape[1]))
        ax[0].set_yticks(np.arange(matrix.shape[0]))

        cumulative_count = np.cumsum(np.sum(stress_matrix, axis=0))

        ax[2].plot(np.arange(len(cumulative_count)),cumulative_count,marker='o')
        ax[2].set_xticks(np.arange(len(cumulative_count)))
        ax[2].set_xticklabels([f'{j}\n(#{int(cumulative_count[j])})' for j in range(matrix.shape[1])])
        ax[2].grid()
        ax[2].set_title('Cumulative stress indicator')
        stage_matrix = np.array([np.pad(matrix[np.where(matrix[:,i])[0],i],(0,self.n_dancers-len(matrix[np.where(matrix[:,i])[0]])),mode='constant') for i in range(matrix.shape[1])])
        stage_matrix+=stage_matrix%2
        for i in range(1,stage_matrix.shape[0]):
            stage_matrix[i]=minimize_differences(stage_matrix[i-1],stage_matrix[i])
        stage_matrix = stage_matrix.T

        stage_matrix_choreographies = stage_matrix//2-1
        for i in range(stage_matrix_choreographies.shape[0]):    
            for j in range(stage_matrix_choreographies.shape[1]):   
                if stage_matrix_choreographies[i,j]>=0:
                    ax[1].text(j,i, int(stage_matrix_choreographies[i,j]), ha='center', va='center', color='black', fontsize='12',backgroundcolor=(1,1,1,.5))

        ax[1].set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
        ax[1].set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
        ax[1].grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax[1].imshow(stage_matrix, cmap=cmap, aspect='auto',alpha=.8)
        ax[1].set_xticks(np.arange(stage_matrix.shape[1]))
        ax[1].set_yticks(np.arange(stage_matrix.shape[0]))
        ax[1].set_yticklabels(np.arange(stage_matrix.shape[0])+1)
        ax[1].set_xticklabels(np.arange(stage_matrix.shape[1])+1)
        ax[1].set_title('Stage composition')
        ax[1].set_ylabel('Dancers')
        ax[0].set_xlabel('Choreography')
        ax[1].set_xlabel('Choreography')

        ax[0].set_yticklabels([f'Costume {i+1}' for i in range(matrix.shape[0])])
        ax[2].set_xlabel('Stress indicator')
        ax[0].set_ylabel('Costumes')
        ax[0].set_title(f'Solution visualization - total stress {cumulative_count[-1]}')
        ax[3].axis('off')
        ax[2].set_ylim(0)
        plt.tight_layout()
        os.makedirs('visualization',exist_ok=True)
        plt.savefig('visualization/'+self.filepath.split(os.sep)[-1].split('.')[0]+'.png')
        plt.show()

    def save_solution(self,sol:Solution):
        """
            Saves the solution
        """
        os.makedirs('solutions',exist_ok=True)
        filename = 'solutions/'+self.filepath.split(os.sep)[-1]

        # Write the array to the file
        np.savetxt(filename, sol.raw, fmt='%d', delimiter=',')
