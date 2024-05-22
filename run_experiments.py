from new_functions import *
from utils import *
import os
import networkx as nx
import argparse
import json
import scipy
from tqdm import tqdm
import re
import time


def run_experiment(args):

    if args.problem=="bipartite" or args.type=="bipartite":
        args.type="bipartite"
        args.problem="bipartite"

    if args.problem=="topological_sorting" or args.type=="dag":
        args.type="dag"
        args.problem="topological_sorting"

    print(args)

    directory_path = 'graphs/'+args.type+'/'+args.size
    q_directory_path = 'graphs_questions/'+args.type+'/'+args.size
    results_path = 'exp_results/'+args.problem+'/'+args.adj+'/'+args.type+'/'+args.size+'/'+args.model+'/'+args.method
    experiment_path = 'experiments/'+args.problem+'/'+args.adj+'/'+args.type+'/'+args.size+'/'+args.model
    # Example of passing additional arguments: assume the adjacency list is delimited by commas

    no_graphs = len(os.listdir(directory_path))
    found_correct = 0
    total = 0

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    not_solved = []

    files = os.listdir(directory_path)
    for i in tqdm(range(len(files))):
        filename = str(i)+".txt"

        graph_file_path = os.path.join(directory_path, filename)

        if args.type=="dag":
            graph = nx.read_adjlist(graph_file_path, comments='#', create_using=nx.DiGraph)
        else:
            graph = nx.read_adjlist(graph_file_path, comments='#')

        question_file_path = os.path.join(q_directory_path, filename)
        f = open(question_file_path, 'r')
        question_str = f.read()
        f.close()

        question_json = json.loads(question_str)

        # if args.adj=="adj":
        #     question = question['adj'][args.problem]
        # else:
        if args.problem in question_json['edgelist']:
            question = question_json['edgelist'][args.problem]
        else:
            continue
        # import pdb; pdb.set_trace()

        # best_prompt = " Keep only the first list and output how many numbers it has."
        # question = question.strip("What is the number of nodes of G?")+prompt
        prompt = ""
        connected_nodes_prompt = ""
        connectivity_prompt = ""
        shortest_path_prompt = ""
        node_degree_prompt = ""
        cot_prompt = ""

        if args.method=="none" or 'default' in args.method or args.method=="cot" or args.method=="bag":

            if 'default' in args.method:
                name = args.method.replace("default","")
                f = open("pseudocodes/"+args.problem+name+".txt", "r")
                prompt = f.read()
                # lines = f.readlines()
                f.close()

                if isinstance(question, list):
                    for i in range(len(question)):
                        # question[i] = question[i]+" "+prompt
                        question[i] = prompt+ " "+question[i]
                else:
                    question = prompt + " " + question

            if args.problem=="node_count":
                question += " Output the result like that: The number of nodes is ..."

            if args.problem=="node_degree":
                node_degree_prompt = " Output the result like that: The degree of the node is ..."
                for i, q in enumerate(question):
                    question[i] += node_degree_prompt

            if args.problem=="edge_count":
                question += " Output the result like that: The number of edges is ..."

            if args.problem=="mst" or args.problem=="topological_sorting":
                question += " Output the result as a list."

            if args.problem=="bipartite":
                question += ' Output the result like that: "Yes, the graph G is bipartite." or "No, the graph G is not bipartite."'

            if args.problem=="cycle_check":
                question += ' Output the result like that: "There is a cycle" or "There is no cycle."'

            if args.problem=="connected_components_count":
                question += ' Output the result like that: The number of connected components is...'

            if args.problem=="connected_nodes":
                connected_nodes_prompt = " Output the result as a list."
                for i, q in enumerate(question):
                    question[i] += connected_nodes_prompt

            if args.problem=="shortest_path":
                shortest_path_prompt = " Output the result like that: the shortest path length is..."
                for i, q in enumerate(question):
                    question[i] += shortest_path_prompt

            if args.problem=="connectivity":
                connectivity_prompt = " Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'"
                for i, q in enumerate(question):
                    question[i] += connectivity_prompt

            if args.method=="cot" or args.method=="bag":
                if args.method=="cot":
                    cot_prompt = "Let's think step by step."
                elif args.method=="bag":
                    cot_prompt = "Let's construct a graph with the nodes and edges first."
                if isinstance(question, list):
                    for i in range(len(question)):
                        question[i] = question[i]+" "+cot_prompt
                else:
                    question = question + " " + cot_prompt

        elif args.method=="simplify_dif":
            question = question.strip("what is the number of nodes")
            prompt = " Keep only the first list and output how many numbers it has."
            # prompt = " Count how many numbers in the first sublist"
        elif args.method=="simplify_more":
            question = question.strip("what is the number of nodes")
            prompt = " Keep only the first list and count how many 0 and 1 it has."


        elif args.method=="alg" or 'pseudo' in args.method:

            if args.method=="alg":
                f = open("pseudocodes/"+args.problem+".txt", "r")
                prompt = f.read()
                # lines = f.readlines()
                f.close()

            if 'pseudo' in args.method:
                name = args.method.replace('pseudo','')
                f = open("pseudocodes/"+args.problem+"_"+name+"pseudo.txt", "r")
                prompt = f.read()
                # lines = f.readlines()
                f.close()

            if args.problem=="node_count":
                # question = question.strip("What is the number of nodes of G?")
                question += " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of nodes is ..."
            if args.problem=="node_degree":
                node_degree_prompt = " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The degree of the node is ..."
                for i in range(len(question)):
                    question[i] += " " + node_degree_prompt
            if args.problem=="edge_count":
                # question = question.strip("What is the number of edges of G?")
                question += " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of edges is ..."
            if args.problem=="cycle_check":
                # question = question.strip("Is there a cycle in G")
                question += ' Follow the provided pseudocode step-by-step and show all steps. Output the result like that: "There is a cycle" or "There is no cycle."'
            if args.problem=="connected_components_count":
                # question = question.strip("Is there a cycle in G")
                question += ' Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of connected components is ...'
            if args.problem=="mst":
                # question = question.strip("What is the minimum spanning tree of G?")
                question += " Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."
            if args.problem=="topological_sorting":
                # question = question.strip("What is the topological sorting of G?")
                question += " Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."
            if args.problem=="bipartite":
                # question = question.strip("Is graph G bipartite or not?")
                question += ' Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: "Yes, the graph G is bipartite." or "No, the graph G is not bipartite."'
            if args.problem=="shortest_path":
                shortest_path_prompt = " Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: the shortest path length is ..."
                for i in range(len(question)):
                    question[i] += shortest_path_prompt
            if args.problem=="connectivity":
                connectivity_prompt = "Follow the provided pseudocode step-by-step and show all steps. Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'"
                for i in range(len(question)):
                    question[i] += " " + connectivity_prompt
            if args.problem=="connected_nodes":
                connected_nodes_prompt = "Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."
                for i in range(len(question)):
                    question[i] += " " + connected_nodes_prompt

            if isinstance(question, list):
                for i in range(len(question)):
                    # question[i] = question[i]+" "+prompt
                    question[i] = prompt+ " "+question[i]
            else:
                question = prompt + " " + question
            # question = question + " " + prompt

        # time.sleep(0.5)
        questions = []
        answers = []
        answer = ""
        # import pdb; pdb.set_trace()

        result_path = os.path.join(results_path, filename)
        # if os.path.exists(result_path):
        #     f = open(result_path, "r")
        #     answer = f.read()
        #     f.close()
        # else:

        # this is for node_degree, connectivity, connected_nodes, edge_existence, shortest_path
        if isinstance(question, list):
            for q in question:
                questions.append(q)
                answer = get_openai_response(q, args)
                answers.append(answer)
        else:
            answer = get_openai_response(question, args)
            # import pdb; pdb.set_trace()

        corrects = []
        correct = "False"
        found_answers = []
        ground_truth_solutions = []

        edgelist = [(int(u), int(v)) for u, v in graph.edges()]

        if args.problem=="node_count":
            ground_truth_solution = node_count(edgelist)
            found_answer = get_number_answer(answer)
        elif args.problem=="edge_count":
            ground_truth_solution = edge_count(edgelist)
            found_answer = get_number_answer(answer)
        elif args.problem=="node_degree":
            for i, q in enumerate(questions):
                q = q.strip(cot_prompt)
                q = q.strip(node_degree_prompt)
                node = int(q.split()[-1].strip('"').strip("?").strip())
                ground_truth_solution = node_degree(edgelist, node)
                ground_truth_solutions.append(ground_truth_solution)
                found_answer = get_number_answer(answers[i])
                found_answers.append(found_answer)
                # import pdb; pdb.set_trace()
        elif args.problem=="connected_nodes":
            for i, q in enumerate(questions):
                q = q.strip(cot_prompt)
                q = q.strip(connected_nodes_prompt)
                # import pdb; pdb.set_trace()
                node = int(q.split()[-1].strip('"').strip("?").strip())
                found_answer, ground_truth_solution = connected_nodes_get_answer(answers[i], edgelist, node)
                ground_truth_solutions.append(ground_truth_solution)
                found_answers.append(found_answer)
        elif args.problem=="connected_components_count":
            ground_truth_solution = connected_components_count(edgelist)
            found_answer = get_number_answer(answer)
        elif args.problem=="cycle_check":
            ground_truth_solution = cycle_check(edgelist)
            found_answer = cycle_get_boolean_answer(answer)
        elif args.problem=="connectivity":
            for i, q in enumerate(questions):
                q = q.strip(cot_prompt)
                q = q.strip(connectivity_prompt).strip()
                source_node = int(q.split()[-1].strip('"').strip("?").strip())
                target_node = int(q.split()[-3].strip('"').strip("?").strip())
                ground_truth_solution = connectivity(edgelist, source_node, target_node)
                ground_truth_solutions.append(ground_truth_solution)
                found_answer = connectivity_get_boolean_answer(answers[i])
                found_answers.append(found_answer)
        elif args.problem=="bipartite":
            ground_truth_solution = is_bipartite(edgelist)
            found_answer = bipartite_get_boolean_answer(answer)
        elif args.problem=="shortest_path":
            for i, q in enumerate(questions):
                q = q.strip(cot_prompt)
                q = q.strip(shortest_path_prompt).strip()
                # import pdb; pdb.set_trace()
                source_node = int(q.split()[-1].strip('"').strip("?").strip())
                target_node = int(q.split()[-3].strip('"').strip("?").strip())
                found_answer = get_number_answer(answers[i])
                ground_truth_solution = shortest_path(edgelist, source_node, target_node)
                found_answers.append(found_answer)
                ground_truth_solutions.append(ground_truth_solution)
        elif args.problem=="mst":
            # ground_truth_solution = minimum_spanning_tree(edgelist)
            found_answer, ground_truth_solution = mst_get_answer(answer, edgelist)
        elif args.problem=="topological_sorting":
            # ground_truth_solution = topological_sorting(edgelist)
            found_answer, ground_truth_solution = topological_get_answer(answer, edgelist)


        if isinstance(question, list):

            f = open(result_path, "w")

            correct_file = "False"

            for i, q in enumerate(questions):
                found_answer = found_answers[i]
                ground_truth_solution = ground_truth_solutions[i]
                answer = answers[i]

                correct = "False"
                if found_answer==ground_truth_solution:
                    correct = "Correct"
                    found_correct +=1
                    print("Correct")
                    # import pdb; pdb.set_trace()
                else:
                    if filename not in not_solved:
                        not_solved.append(filename)
                total += 1

                f.write(correct+"\n\n")
                f.write("Ground truth: "+str(ground_truth_solution)+"\n")
                f.write("Found answer: "+str(found_answer)+"\n\n")
                f.write("Question: "+q+"\n\n")
                f.write("Answer: "+answer+"\n\n\--------------------\n\n")
            f.close()

        else:
            total += 1

            if found_answer==ground_truth_solution:
                correct = "Correct"
                print(correct)
                found_correct +=1
            else:
                not_solved.append(filename)

            result_path = os.path.join(results_path, filename)

            f = open(result_path, "w")
            f.write(correct+"\n\n")
            f.write("Ground truth: "+str(ground_truth_solution)+"\n")
            f.write("Found answer: "+str(found_answer)+"\n\n")
            f.write("Question: "+question+"\n\n")
            f.write("Answer: "+answer)
            f.close()

        # import pdb; pdb.set_trace()

    solved = "Solved correctly: "+str(found_correct)
    perc = "Percentage solved correctly (%): "+str(found_correct/total)
    total = "Total: "+str(total)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    f = open(experiment_path+'/'+args.method+".txt", "w")
    f.write(args.method+"\n")
    f.write(args.problem+"\n")
    f.write(solved+"\n"+perc+"\n"+total+"\n\n")
    if len(not_solved)>0:
        f.write("Not solved:\n")
        for ns in not_solved:
            f.write(ns+"\n")
    f.close()

    print(args)
    print(solved)
    print(perc)
    return perc

model_list = ["text-davinci-003", "code-davinci-002", "gpt-3.5-turbo", "gpt-4"]
parser = argparse.ArgumentParser(description="graph_reasoning")
parser.add_argument('--adj', type=str, default="edgelist", help='adjacency matrix in matrix or list (default: edgelist)')
parser.add_argument('--problem', type=str, default="node_count", help='problem to solve (default: count nodes)')
parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help='name of LM (default: text-davinci-003)')
parser.add_argument('--type', type=str, default="er", help='graph type (default: er)')
parser.add_argument('--size', type=str, default="small", help='graph size (default: small)')
parser.add_argument('--temperature', type=int, default=0, help='temperature (default: 0)')
parser.add_argument('--token', type=int, default=4000, help='max token (default: 4000)')
parser.add_argument('--method', type=str, default="none", help='method to experiment (default: none)')
parser.add_argument('-a', '--all', action='store_true', help='run all experiments (default: run single experiment)')

args = parser.parse_args()

# TODO: BIPARTITE, DAG

if args.all:
    problems = ["node_count", "edge_count", "node_degree", "connected_nodes", "connected_components_count", "cycle_check",
                "shortest_path", "minimum_spanning_tree", "topological_sorting", "bipartite"]
    
    sizes = ["small", "medium", "large"]
    # types = ["er", "ba", "path", "complete", "sbm", "sfn", "star"]
    types = ["er"]

    methods = ["none", "default_1_shot", "cot", "alg", "1_shot_pseudo"]

    args.model = 'gpt-3.5-turbo'

    for problem in problems:
        args.problem = problem
        for size in sizes:
            args.size = size
            for t in types:
                args.type = t
                for method in methods:
                    args.method = method
                    experiment_path = 'experiments/'+args.problem+'/'+args.adj+'/'+args.type+'/'+args.size+'/'+args.model
                    path = experiment_path+'/'+args.method+".txt"
                    if not os.path.isfile(path):
                        perc = run_experiment(args)
else:
    run_experiment(args)
