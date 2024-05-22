from new_functions import *
import os
import networkx as nx
import argparse
import json
import scipy
import openai
from tqdm import tqdm
import re
import inspect
import time

client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def extract_solution(answer):
    found_answer = find_number_between_phrases(answer)
    if found_answer is None:
        # print("No number found between the specified phrases. Attempting: ")
        found_answer = find_number(answer)
    return found_answer

def get_number_answer(answer):
    found_answer = find_number(answer)
    return found_answer

def get_boolean_answer(answer, ground_truth_solution):
    if str(ground_truth_solution) in answer:
        return True
    return False

def topological_get_answer(answer, edgelist):
    found_answer = answer.split("\n")[-1]
    if "[" not in found_answer and "]" not in found_answer:
        if len(answer.split("\n"))>=2:
            found_answer = answer.split("\n")[-2]
    if "[" not in found_answer and "]" not in found_answer:
        if len(answer.split("\n"))>=3:
            found_answer = answer.split("\n")[-3]
    if "[" in found_answer and "]" in found_answer:
        found_answer = found_answer[found_answer.index("["):found_answer.index("]")+1].strip(".").strip()
        try:
            found_answer = eval(found_answer)
            if is_valid_topological_sort(edgelist, found_answer):
                return found_answer, found_answer
        except:
            pass
    return found_answer, []

def connected_nodes_get_answer(answer, edgelist, node):
    counter = 1
    found_answer = answer.split("\n")[-counter]
    while("[" not in found_answer and "]" not in found_answer):
        counter += 1
        if len(answer.split("\n"))>=counter:
            found_answer = answer.split("\n")[-counter]
        if counter==10:
            break

    if "[" in found_answer and "]" in found_answer:
        found_answer = found_answer[found_answer.index("["):found_answer.index("]")+1].strip(".").strip()
        ground_truth_solution = connected_nodes(edgelist, int(node))
        try:
            found_answer = eval(found_answer)
            if set(found_answer)==set(ground_truth_solution):
                return found_answer, found_answer
        except:
            pass
    return found_answer, []

def mst_get_answer(answer, edgelist):
    counter = 1
    found_answer = answer.split("\n")[-counter]
    while("[" not in found_answer and "]" not in found_answer):
        counter += 1
        if len(answer.split("\n"))>=counter:
            found_answer = answer.split("\n")[-counter].strip(".").strip("`")
        if counter==10:
            break

    if "[" in found_answer and "]" in found_answer:
        found_answer = found_answer[found_answer.index("["):found_answer.index("]")+1].strip(".").strip("`")

        try:
            found_answer = eval(found_answer)
            # print(found_answer)
            if is_valid_minimum_spanning_tree(edgelist, found_answer):
                return found_answer, found_answer
        except:
            pass
    return found_answer, []

def bipartite_get_boolean_answer(answer):
    # answer = answer.lower()
    anser = answer.split("\n")[-1]
    if 'Yes, the graph G is bipartite.' in answer or 'Yes' in answer or "is indeed bipartite" in answer or ", the graph G is bipartite" in answer:
        return True
    if "No, the graph G is not bipartite." in answer or 'is not' in answer or "isn't" in answer or ", the graph G is not bipartite" in answer or ", the graph G is not a bipartite" in answer:
        return False
    return None

def connectivity_get_boolean_answer(answer):
    # answer = answer.lower()
    if 'Yes the two nodes are connected.' in answer or 'Yes' in answer:
        return True
    if 'No the two nodes are not connected.' in answer or 'No' in answer:
        return False
    return None

def cycle_get_boolean_answer(answer):
    # answer = answer.lower()
    # last_elems = answer.split()[-15:]
    # last_str = " ".join(last_elems)
    # if "There is a cycle" in answer or "the graph G does contain a cycle" in answer or 'Yes, there is a cycle' in answer or 'The graph has a cycle.' in answer or "does have a cycle" in answer:
    #     return True
    # if "There is no cycle" in answer or 'The graph does not have a cycle.' in answer or "is no" in last_str or "no cycle" in last_str or "isn't a cycle" in last_str or "is acyclic" in last_str or "absence of cycle" in last_str or "the graph does not have a cycle" in last_str or "did not find any cycle" in last_str:
    #     return False
    # if "whether there is a cycle" in last_str or "has a cycle or not" in last_str or "if there is a cycle" in last_str:
    #     return None
    answer = answer.split("\n")[-1]
    if "There is a cycle" in answer:
        return True
    if "There is no cycle" in answer:
        return False
    return None

def find_number(answer):
    answer_list = answer.split()[-10:]
    answer_number = 0
    # import pdb; pdb.set_trace()
    for i in range(1, len(answer_list)):
        word = answer_list[-i]
        word = word.strip().strip("'").strip('"').strip(".").strip("*").strip("'").strip('"')
        try:
            # Try to convert the word to an integer
            answer_number = int(word)
            return answer_number
        except ValueError:
            # If it's not an integer, just continue to the next word
            correct = "Cannot parse integer"
            pass
    return -2


def find_number_between_phrases(text):
    # Define a regular expression pattern
    # This pattern looks for 'the graph has', followed by any number of spaces (\s*),
    # digits (\d+), more spaces, and then 'nodes'.
    pattern = r"the graph G has\s*(\d+)\s*nodes"
    # pattern = r"the number of nodes in graph G is\s*(\d+)"

    # Search the text for the pattern
    match = re.search(pattern, text, re.IGNORECASE)

    # If a match is found, return the number as an integer
    if match:
        return int(match.group(1))

    # If no match is found, return None
    return None

@retry(wait=wait_random_exponential(min=1, max=100), stop=stop_after_attempt(500))
def get_openai_response(text, args):
    completion = client.chat.completions.create(
      model=args.model,
      messages=[
        {"role": "system", "content": "You are a helpful graph problem assistant."},
        {"role": "user", "content": text}
      ],
      temperature=0
      # seed=20
    )
    return(completion.choices[0].message.content)
