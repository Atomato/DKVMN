# %%
import networkx as nx
import matplotlib.pyplot as plt

prob_ij = {}
sum_prob_j = {}
with open("conditional_prob.txt", "r") as f:
    for j in range(1, 51):
        prob_ij_list = []
        for i in range(1, 51):
            if i == j: continue

            prob_ij[(i, j)] = float((f.readline()))
            prob_ij_list.append(prob_ij[(i, j)])
        
        sum_prob_j[j] = sum(prob_ij_list)

# Calculate influence
G = nx.Graph()
for j in range(1, 51):
    for i in range(1, 51):
        if i == j: continue
        influence = (prob_ij[(i, j)] / sum_prob_j[j]) * 49

        if influence > 1.6:
            G.add_edge(i, j)
            

for node in range(1, 51):
    if node not in G.nodes:
        influence_list = {}
        for k in range(1, 51):
            if node == k: continue
            influence_list[(prob_ij[(node, k)] / sum_prob_j[k]) * 49] = k
            influence_list[(prob_ij[(k, node)] / sum_prob_j[node]) * 49] = k
        
        max_node = influence_list[max(influence_list.keys())]
        G.add_edge(node, max_node)

# Ground truth cluster labels
nodes_list = [
    [7, 10, 12, 17, 21, 26, 27, 30, 36, 38, 42],
    [11, 13, 18, 24, 25, 33, 39, 50],
    [2, 4, 8, 29, 34, 40, 41, 45, 47, 49],
    [1, 3, 5, 9, 15, 16, 37, 43, 44, 46, 48],
    [6, 14, 19, 20, 22, 23, 28, 31, 32, 35]
]
color_list = ["r", "m", "c", "y", "g"]
pos = nx.spring_layout(G)

options = {"node_size": 300, "alpha": 0.8, "font_size": 10}
for nodes, color in zip(nodes_list, color_list):
    nx.draw(G, pos, nodelist=nodes, node_color=color, with_labels=True, **options)
