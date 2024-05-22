# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Random graph generation."""

import random
import os
import json
import networkx as nx
import numpy as np

from networkx.algorithms import bipartite


_NUMBER_OF_NODES_RANGE = {
    "small": np.arange(5, 11),
    "medium": np.arange(11, 21),
    "large": np.arange(21, 51),
}
_NUMBER_OF_COMMUNITIES_RANGE = {
    "small": np.arange(2, 4),
    "medium": np.arange(2, 8),
    "large": np.arange(2, 10),
}


def generate_graphs(
    number_of_graphs,
    graph_sizes,
    algorithm,
    directed,
    random_seed = 13,
    er_min_sparsity = 0.0,
    er_max_sparsity = 1.0,
):
  """Generating multiple graphs using the provided algorithms.

  Args:
    number_of_graphs: number of graphs to generate
    algorithm: the random graph generator algorithm
    directed: whether to generate directed or undirected graphs.
    random_seed: the random seed to generate graphs with.
    er_min_sparsity: minimum sparsity of er graphs.
    er_max_sparsity: maximum sparsity of er graphs.

  Returns:
    generated_graphs: a list of nx graphs.
  Raises:
    NotImplementedError: if the algorithm is not yet implemented.
  """

  random.seed(random_seed)
  np.random.seed(random_seed)

  generated_graphs = []
  random_state = np.random.RandomState(random_seed)
  if algorithm == "er":
    for i in range(number_of_graphs):
      n_nodes = 0
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      while n_nodes != number_of_nodes:
        sparsity = random.uniform(er_min_sparsity, er_max_sparsity)
        G = nx.erdos_renyi_graph(number_of_nodes, sparsity, seed=random_state, directed=directed)
        remove_disconnected_nodes(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes = G.number_of_nodes()
      generated_graphs.append(G)
  elif algorithm == "ba":
    for i in range(number_of_graphs):
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      m = random.randint(1, number_of_nodes - 1)
      n_nodes = 0
      while n_nodes != number_of_nodes:
        G = nx.barabasi_albert_graph(number_of_nodes, m, seed=random_state)
        remove_disconnected_nodes(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes = G.number_of_nodes()
      if directed:
        generated_graphs.append(randomize_directions(G))
      else:
        generated_graphs.append(G)
  elif algorithm == "sbm":
    for i in range(number_of_graphs):
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      number_of_communities = random.choice(_NUMBER_OF_COMMUNITIES_RANGE[graph_sizes])
      n_nodes = 0
      while n_nodes != number_of_nodes:
        # sizes forms number of nodes in communities.
        sizes = []
        for _ in range(number_of_communities - 1):
          sizes.append(
              random.randint(
                  1,
                  max(
                      1,
                      number_of_nodes - sum(sizes) - (number_of_communities - 1),
                  ),
              )
          )
        sizes.append(number_of_nodes - sum(sizes))

        # p forms probabilities of communities connecting each other.
        p = np.random.uniform(size=(number_of_communities, number_of_communities))
        if random.uniform(0, 1) < 0.5:
          p = np.maximum(p, p.transpose())
        else:
          p = np.minimum(p, p.transpose())

        G = nx.stochastic_block_model(sizes, p, seed=random_state, directed=directed)
        remove_disconnected_nodes(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes = G.number_of_nodes()

      # sbm graph generator automatically adds dictionary attributes.
      sbm_graph = remove_graph_data(G)
      generated_graphs.append(G)
  elif algorithm == "sfn":
    for i in range(number_of_graphs):
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      n_nodes = 0
      while n_nodes != number_of_nodes:
        G = nx.scale_free_graph(number_of_nodes, seed=random_state)
        remove_disconnected_nodes(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes = G.number_of_nodes()
      # sfn graphs are by defaukt directed.
      if not directed:
        generated_graphs.append(remove_directions(G))
      else:
        generated_graphs.append(G)
  elif algorithm == "complete":
    for i in range(number_of_graphs):
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      create_using = nx.DiGraph if directed else nx.Graph
      generated_graphs.append(nx.complete_graph(number_of_nodes, create_using=create_using))
  elif algorithm == "star":
    for i in range(number_of_graphs):
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      # number_of_nodes for star is the input + a center node.
      G = nx.star_graph(number_of_nodes - 1)
      if directed:
        generated_graphs.append(randomize_directions(G))
      else:
        generated_graphs.append(G)
  elif algorithm == "path":
    for i in range(number_of_graphs):
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      create_using = nx.DiGraph if directed else nx.Graph
      generated_graphs.append(nx.path_graph(number_of_nodes, create_using=create_using))
  elif algorithm == 'dag':
    for i in range(number_of_graphs):
      n_nodes = 0
      cc = 0
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      while n_nodes != number_of_nodes and cc != 1:
        sparsity = random.uniform(er_min_sparsity, er_max_sparsity)
        G = nx.erdos_renyi_graph(number_of_nodes, sparsity, seed=random_state, directed=directed)
        remove_disconnected_nodes(G)
        n_nodes = G.number_of_nodes()
        cc = nx.number_connected_components(G)
      G_dag = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
      assert nx.is_directed_acyclic_graph(G_dag)
      generated_graphs.append(G_dag)
  elif algorithm == "bipartite":
    for i in range(number_of_graphs//2):
      n = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])-1
      m = random.choice(range(1,np.max(_NUMBER_OF_NODES_RANGE[graph_sizes])-n+1))
      n_nodes = 0
      while n_nodes != n+m:
        p = random.uniform(er_min_sparsity, er_max_sparsity)
        G = bipartite.random_graph(n, m, p)
        remove_disconnected_nodes(G)
        n_nodes = G.number_of_nodes()

      generated_graphs.append(G)

      n_nodes = 0
      number_of_nodes = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes])
      while n_nodes != number_of_nodes:
        sparsity = random.uniform(er_min_sparsity, er_max_sparsity)
        G = nx.erdos_renyi_graph(number_of_nodes, sparsity, seed=random_state, directed=directed)
        remove_disconnected_nodes(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes = G.number_of_nodes()
      generated_graphs.append(G)
  else:
    raise NotImplementedError()
  return generated_graphs


def remove_graph_data(graph):
  # GraphML writer does not support dictionary data for nodes or graphs.
  for ind in range((graph.number_of_nodes())):
    graph.nodes[ind].pop("block", None)
  graph_data_keys = list(graph.graph.keys())
  for _, node in enumerate(graph_data_keys):
    graph.graph.pop(node, None)
  return graph

def remove_disconnected_nodes(graph):
  to_remove = []
  for node in graph.nodes():
    if graph.degree(node) == 0:
      to_remove.append(node)
  graph.remove_nodes_from(to_remove)


def randomize_directions(graph):
  # Converting the undirected graph to a directed graph.
  directed_graph = graph.to_directed()
  # For each edge, randomly choose a direction.
  edges = list(graph.edges())
  for u, v in edges:
    if random.random() < 0.5:
      directed_graph.remove_edge(u, v)
    else:
      directed_graph.remove_edge(v, u)

  return directed_graph


def remove_directions(graph):
  # Converting the direted graph to an undirected one by removing directions.
  undirected_graph = nx.Graph()
  undirected_graph.add_nodes_from(graph.nodes())
  # Add edges between nodes, ignoring directions.
  for u, v in graph.edges():
    undirected_graph.add_edge(u, v)

  return undirected_graph


def create_dataset():
  adj_start = 'Let G be a graph. The adjacency matrix of graph G is the following: '
  edgelist_start = 'Let G be a graph. The edgelist of graph G is the following: '

  node_count = '. What is the number of nodes of G?'
  edge_count = '. What is the number of edges of G?'
  cycle_check = '. Is there a cycle in G?'
  connected_components_count = '. What is the number of connected components of G?'
  node_degree = '. What is the degree of node '
  edge_existence = '. Is there an edge between nodes '
  connected_nodes = '. Which nodes are the neighbors of node '
  connectivity = '. Is there a path between nodes '
  shortest_path = '. What is the shortest path length between nodes  '
  bipartite = '. Is graph G bipartite or not?'
  topological_sorting = '. Graph G is a directed acyclic graph. What is the topological sorting of G?'
  mst = '. What is the minimum spanning tree of G?'

  Gs = dict()
  questions = dict()
  for algorithm in ['er','ba','sbm','sfn','complete','star','path','bipartite','dag']:
    print(algorithm)
    Gs[algorithm] = {'small': dict(),'medium': dict(),'large': dict()}
    questions[algorithm] = {'small': dict(),'medium': dict(),'large': dict()}
    for graph_sizes in ['small','medium','large']:
      Gs_generated = generate_graphs(100, graph_sizes, algorithm, False)
      questions[algorithm][graph_sizes] = dict()
      if algorithm != 'dag' and algorithm !='bipartite':
        for graph_idx, G in enumerate(Gs_generated):
          Gs[algorithm][graph_sizes][graph_idx] = G
          questions[algorithm][graph_sizes][graph_idx] = {'adj': dict(), 'edgelist': dict()}
          adj = str(nx.to_numpy_array(G).astype(int).tolist())
          edgelist = str(list(G.edges()))

          ## node count, edge count, cycle check, connected components count, mst
          questions[algorithm][graph_sizes][graph_idx]['adj']['node_count']  = adj_start + adj + node_count
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['node_count']  = edgelist_start + edgelist + node_count
          questions[algorithm][graph_sizes][graph_idx]['adj']['edge_count']  = adj_start + adj + edge_count
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['edge_count']  = edgelist_start + edgelist + edge_count
          questions[algorithm][graph_sizes][graph_idx]['adj']['cycle_check']  = adj_start + adj + cycle_check
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['cycle_check']  = edgelist_start + edgelist + cycle_check
          questions[algorithm][graph_sizes][graph_idx]['adj']['connected_components_count']  = adj_start + adj + connected_components_count
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['connected_components_count']  = edgelist_start + edgelist + connected_components_count
          questions[algorithm][graph_sizes][graph_idx]['adj']['mst']  = adj_start + adj + mst
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['mst']  = edgelist_start + edgelist + mst
          nodes = list(G.nodes())

          ## node degree
          nodes_sample = np.random.choice(nodes, size=5, replace=False)
          questions[algorithm][graph_sizes][graph_idx]['adj']['node_degree']  = []
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['node_degree'] = []
          for node in nodes_sample:
            questions[algorithm][graph_sizes][graph_idx]['adj']['node_degree'].append(adj_start + adj + node_degree + str(node) +'?')
            questions[algorithm][graph_sizes][graph_idx]['edgelist']['node_degree'].append(edgelist_start + edgelist + node_degree + str(node) +'?')

          ## edge existence
          edges = list(G.edges())
          min_pairs = min(len(edges), (len(nodes)*(len(nodes)-1))//2-len(edges), 5)
          questions[algorithm][graph_sizes][graph_idx]['adj']['edge_existence'] = []
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['edge_existence'] = []
          idx = np.random.choice(range(len(edges)), size=min_pairs, replace=False)
          for i in range(idx.size):
            questions[algorithm][graph_sizes][graph_idx]['adj']['edge_existence'].append(adj_start + adj + edge_existence + str(edges[idx[i]][0]) + ' and ' + str(edges[idx[i]][1]) +'?')
            questions[algorithm][graph_sizes][graph_idx]['edgelist']['edge_existence'].append(edgelist_start + edgelist + edge_existence + str(edges[idx[i]][0]) + ' and ' + str(edges[idx[i]][1]) +'?')
          non_edges = list()
          for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
              if (i,j) not in edges:
                non_edges.append((i,j))
          idx = np.random.choice(range(len(non_edges)), size=min_pairs, replace=False)
          for i in range(idx.size):
            questions[algorithm][graph_sizes][graph_idx]['adj']['edge_existence'].append(adj_start + adj + edge_existence + str(non_edges[idx[i]][0]) + ' and ' + str(non_edges[idx[i]][1]) +'?')
            questions[algorithm][graph_sizes][graph_idx]['edgelist']['edge_existence'].append(edgelist_start + edgelist + edge_existence + str(non_edges[idx[i]][0]) + ' and ' + str(non_edges[idx[i]][1]) +'?')

          ## connected nodes (i.e., neighbors)
          neighbors = {}
          nodes_sample = np.random.choice(nodes, size=5, replace=False)
          questions[algorithm][graph_sizes][graph_idx]['adj']['connected_nodes'] = []
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['connected_nodes'] = []
          for node in nodes_sample:
            questions[algorithm][graph_sizes][graph_idx]['adj']['connected_nodes'].append(adj_start + adj + connected_nodes + str(node) +'?')
            questions[algorithm][graph_sizes][graph_idx]['edgelist']['connected_nodes'].append(edgelist_start + edgelist + connected_nodes + str(node) +'?')

          ## connectivity
          connected_pairs = []
          disconnected_pairs = []
          CCs = list(nx.connected_components(G))
          CCs = [list(CC) for CC in CCs]
          for j in range(len(CCs)):
            for k in range(j, len(CCs)):
              if j == k:
                for l in range(len(CCs[j])):
                  for m in range(l+1,len(CCs[k])):
                    connected_pairs.append((CCs[j][l],CCs[k][m]))
              else:
                for l in range(len(CCs[j])):
                  for m in range(len(CCs[k])):
                    disconnected_pairs.append((CCs[j][l],CCs[k][m]))

          if len(disconnected_pairs) > 0:
            min_len = min(len(connected_pairs),len(disconnected_pairs),5)
            questions[algorithm][graph_sizes][graph_idx]['adj']['connectivity'] = []
            questions[algorithm][graph_sizes][graph_idx]['edgelist']['connectivity'] = []
            idx = np.random.choice(range(len(connected_pairs)), size=min_len, replace=False)
            for i in range(idx.size):
              questions[algorithm][graph_sizes][graph_idx]['adj']['connectivity'].append(adj_start + adj + connectivity + str(connected_pairs[idx[i]][0]) + ' and ' + str(connected_pairs[idx[i]][1]) +'?')
              questions[algorithm][graph_sizes][graph_idx]['edgelist']['connectivity'].append(edgelist_start + edgelist + connectivity + str(connected_pairs[idx[i]][0]) + ' and ' + str(connected_pairs[idx[i]][1]) +'?')
            idx = np.random.choice(range(len(disconnected_pairs)), size=min_len, replace=False)
            for i in range(idx.size):
              questions[algorithm][graph_sizes][graph_idx]['adj']['connectivity'].append(adj_start + adj + connectivity + str(disconnected_pairs[idx[i]][0]) + ' and ' + str(disconnected_pairs[idx[i]][1]) +'?')
              questions[algorithm][graph_sizes][graph_idx]['edgelist']['connectivity'].append(edgelist_start + edgelist + connectivity + str(disconnected_pairs[idx[i]][0]) + ' and ' + str(disconnected_pairs[idx[i]][1]) +'?')

          # shortest path
          sps = dict(nx.all_pairs_shortest_path_length(G))
          pairs = list()
          for n1 in sps:
            for n2 in sps[n1]:
              if n1 != n2:
                pairs.append((n1,n2))
          questions[algorithm][graph_sizes][graph_idx]['adj']['shortest_path'] = []
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['shortest_path'] = []
          idx = np.random.choice(range(len(pairs)), size=5, replace=False)
          for i in range(idx.size):
            questions[algorithm][graph_sizes][graph_idx]['adj']['shortest_path'].append(adj_start + adj + shortest_path + str(pairs[idx[i]][0]) + ' and ' + str(pairs[idx[i]][1]) +'?')
            questions[algorithm][graph_sizes][graph_idx]['edgelist']['shortest_path'].append(edgelist_start + edgelist + shortest_path + str(pairs[idx[i]][0]) + ' and ' + str(pairs[idx[i]][1]) +'?')
      elif algorithm == 'dag':
        for graph_idx, G in enumerate(Gs_generated):
          Gs[algorithm][graph_sizes][graph_idx] = G
          questions[algorithm][graph_sizes][graph_idx] = {'adj': dict(), 'edgelist': dict()}
          adj = str(nx.to_numpy_array(G).astype(int).tolist())
          edgelist = str(list(G.edges()))

          ## topological sorting
          questions[algorithm][graph_sizes][graph_idx]['adj']['topological_sorting']  = adj_start + adj + topological_sorting
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['topological_sorting']  = edgelist_start + edgelist + topological_sorting
      elif algorithm == 'bipartite':
        for graph_idx, G in enumerate(Gs_generated):
          Gs[algorithm][graph_sizes][graph_idx] = G
          questions[algorithm][graph_sizes][graph_idx] = {'adj': dict(), 'edgelist': dict()}
          adj = str(nx.to_numpy_array(G).astype(int).tolist())
          edgelist = str(list(G.edges()))

          ## topological sorting
          questions[algorithm][graph_sizes][graph_idx]['adj']['bipartite']  = adj_start + adj + bipartite
          questions[algorithm][graph_sizes][graph_idx]['edgelist']['bipartite']  = edgelist_start + edgelist + bipartite
  return Gs, questions

Gs, questions = create_dataset()

for G in Gs.keys():
    for g_type in Gs[G].keys():
        graph_directory = "graphs/"+G+"/"+g_type
        question_directory = "graphs_questions/"+G+"/"+g_type

        if not os.path.exists(graph_directory):
            os.makedirs(graph_directory)
        if not os.path.exists(question_directory):
            os.makedirs(question_directory)

        for graph_id in Gs[G][g_type].keys():
            graph = Gs[G][g_type][graph_id]
            path = graph_directory+"/"+str(graph_id)+".txt"
            nx.write_adjlist(graph, path)

            question = questions[G][g_type][graph_id]
            path = question_directory+"/"+str(graph_id)+".txt"
            with open(path, "w") as file:
                json.dump(question, file)
