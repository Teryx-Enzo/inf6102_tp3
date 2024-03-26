import argparse
import solver_naive
import solver_advanced
import time
from utils import Instance
from utils import *



if __name__ == '__main__':
    grade = 0
    for instanceId,optimum,upperbound,maxgrade in zip(['A','B','C','D'],[14,25,137,250],[19,40,213,401],[3,2,2,2]):
        try:
            inst = Instance(f'./instances/{instanceId}.txt')
            try:
                sol = inst.solution_from_file(f'./solutions/{instanceId}.txt')
                cost,validity = inst.objective_score(sol), inst.is_valid(sol)
                grade += (maxgrade-maxgrade*(cost-optimum)/(upperbound-optimum))*int(validity)
                print(f'{instanceId} : ',round((maxgrade-maxgrade*(cost-optimum)/(upperbound-optimum))*int(validity),maxgrade),f'/{maxgrade}')
            except FileNotFoundError as e:
                print(f'{instanceId} : ',0,f'/{maxgrade} (file ./solutions/{instanceId}.txt not found)')
                grade+=0
        except FileNotFoundError as e:
            print(f'{instanceId} : ',0,f'/{maxgrade} (file ./instances/{instanceId}.txt not found)')
            grade+=0
    print(f'Total ',round(grade,2),f'/9')
