import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import re

def plot(string, fig_name):
    if os.path.exists('temp.gexf'):
        if os.path.exists('temp1.gexf'):
            os.remove('temp.gexf')
            os.remove('temp1.gexf')
            name = 'temp.gexf'
        else:
            name = 'temp1.gexf'
    else:
        name = 'temp.gexf'
    file = open(name, 'w')
    file.write(string)
    file.close()
    try:
        G = nx.read_gexf(name)
    except:
        return 1

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6, forward=True)
    if name == 'temp.gexf':
        node_color = "#0076ae"
    else:
        node_color = "#ff7400"
    nx.draw_networkx(G, pos=nx.spring_layout(G), node_color=node_color)
    fig.savefig(fig_name)
    plt.close()

def plot_path(path):
    G1 = nx.read_gexf('temp.gexf')
    G2 = nx.read_gexf('temp1.gexf')
    edit_path = path

    G = nx.disjoint_union(G1,G2)
    G1_nodes = len(G1.nodes())
    G2_nodes = len(G2.nodes())

    pos = nx.spring_layout(G)  # positions for all nodes

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6, forward=True)
    # nodes
    # options = {"node_size": 500, "alpha": 0.8}
    nodelist1 = range(G1_nodes)
    nodelist2 = range(G1_nodes,G1_nodes+G2_nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist1, node_color="#0076ae")
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist2, node_color="#ff7400")

    # edges
    graph2graph = []
    for path in edit_path:
        for i in nodelist1:
            if G.nodes[i]['label'] == path[0]:
                source = i
        for j in nodelist2:
            if G.nodes[j]['label'] == path[1]:
                target = j
        graph2graph.append((source,target))
        G.add_edge(source,target)

    l = nx.draw_networkx_edges(G, pos, edgelist=graph2graph, edge_color='#9e63b5' ,  style='dashed', width=2.0, alpha=0.5)

    edgelist1 = G.subgraph(nodelist1).edges()
    edgelist2 = G.subgraph(nodelist2).edges()
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist1,
        width=2,
        alpha=0.5,
        edge_color="#0076ae",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist2,
        width=2,
        alpha=0.5,
        edge_color="#ff7400",
    )

    labels = {}
    if 'type' in G.nodes().keys():
        for i in G.nodes():
            labels[i] = G.nodes[i]['type']
    else:
        for i in G.nodes():
            labels[i] = G.nodes[i]['label']
    nx.draw_networkx_labels(G, pos, labels, font_size=16)

    fig.savefig('test.png')
    plt.close()