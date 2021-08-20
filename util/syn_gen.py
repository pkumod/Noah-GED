import networkx as nx
import random
import os

data_path = '/home/yanglei/GraphEditDistance/Syn/test/'
if not os.path.exists(data_path):
    os.makedirs(data_path)
nodes = 100

for i in range(2000):
    G = nx.fast_gnp_random_graph(nodes,0.2)
    nx.write_gexf(G,data_path+str(i)+'_1.gexf')
    edge_num = nx.number_of_edges(G)
    rm_edge = random.randint(0,edge_num-1)
    edge_list = list(G.edges())
    rm_start = edge_list[rm_edge][0]
    rm_end = edge_list[rm_edge][1]
    G.remove_edge(rm_start, rm_end)
    add_start = random.randint(0,nodes-2)
    add_end = random.randint(add_start+1,nodes-1)
    while ((add_start,add_end) in G.edges()):
        add_start = random.randint(0, nodes-2)
        add_end = random.randint(add_start+1, nodes-1)
    G.add_edge(add_start,add_end)
    nx.write_gexf(G,data_path+str(i)+'_2.gexf')

    rm_edge = random.randint(0, edge_num - 1)
    edge_list = list(G.edges())
    while (edge_list[rm_edge][0] == add_start and edge_list[rm_edge][1] == add_end):
        rm_edge = random.randint(0, edge_num - 1)
    G.remove_edge(edge_list[rm_edge][0], edge_list[rm_edge][1])
    add_start = random.randint(0, nodes-2)
    add_end = random.randint(add_start + 1, nodes-1)
    while ((add_start, add_end) in G.edges() or (add_start == rm_start and add_end == rm_end)):
        add_start = random.randint(0, nodes-2)
        add_end = random.randint(add_start + 1, nodes-1)
    G.add_edge(add_start, add_end)
    nx.write_gexf(G,data_path+str(i)+'_3.gexf')
