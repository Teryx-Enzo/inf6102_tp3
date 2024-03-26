import argparse
import solver_naive
import solver_advanced
import time
from utils import Instance


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--infile', type=str, default='./instances/datA1')
    parser.add_argument('--outdir', type=str, default='solutions')
    parser.add_argument('--viz', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    instance = Instance(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: last dance")
    print("[INFO] input file: %s" % instance.filepath)
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(instance)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(instance)
    else:
        raise Exception("This agent does not exist")


    solving_time = round((time.time() - start_time) / 60,2)

    # You can disable the display if you do not want to generate the visualization
    if args.viz:
        instance.plot_solution(solution)
    
    instance.save_solution(solution)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print(f"[INFO] Is solution valid ? : {instance.is_valid(solution)}")
    print(f"[INFO] Overall stress index : {instance.objective_score(solution)}")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("***********************************************************")
