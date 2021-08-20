import json
import math
import networkx as nx
import numpy as np
from texttable import Texttable
from os.path import basename
import re
import pp
import subprocess
import time
import pickle
from tqdm import tqdm, trange
import xml.etree.ElementTree as ET
from munkres import Munkres

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def loadGXL(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    index = 0
    g = nx.Graph(filename=basename(filename), name=root[0].attrib['id'])
    dic = {}  # used to retrieve incident nodes of edges
    for node in root.iter('node'):
        dic[node.attrib['id']] = str(index)
        labels = {}
        for attr in node.iter('attr'):
            labels[attr.attrib['name']] = attr[0].text
        labels['label'] = node.attrib['id']
        g.add_node(str(index), **labels)
        index += 1

    for edge in root.iter('edge'):
        labels = {}
        for attr in edge.iter('attr'):
            labels[attr.attrib['name']] = attr[0].text
        g.add_edge(dic[edge.attrib['from']], dic[edge.attrib['to']], **labels)
    return g

def process_pair(pair,superLabel):
    """
    Generate training data with a pair of graphs.
    :param pair: A pair of graph directory.
    :return g_list: A list of input training data.
    """
    g_list = []
    g1 = nx.read_gexf(pair[0])
    g2 = nx.read_gexf(pair[1]) 
    g1_tmp = nx.Graph(g1) # set to unfrozen
    count = len(g1)
    g1_tmp.add_node(str(count),label=superLabel)
    for node in g1.nodes():
        g1_tmp.add_edge(str(count),node)

    count = len(g2)
    g2_tmp = nx.Graph(g2) # set to unfrozen
    g2_tmp.add_node(str(count),label=superLabel)
    for node in g2.nodes():
        g2_tmp.add_edge(str(count),node)

    flag = 0
    with open('AIDS700nef/ged.txt') as fin: 
        for line in fin:
            if flag == 1:
                pt = line.find(']')
                path = line[1:pt]
                path_list = re.findall(r'[0-9]|None+', path)
                cost = line[pt+3:-2]
                cost_list = re.findall(r'[0-9]+', cost)
                break
            if len(line.split('\t')) != 3:
                continue
            g1_name, g2_name, distance = line.split('\t')
            if g1_name == basename(pair[0]) and g2_name == basename(pair[1]):
                ged = int(distance)
                if ged != -1:
                    flag = 1
                else:
                    g_list.append((g1_tmp, g2_tmp, ged))
                    return g_list
    g_list.append((g1_tmp, g2_tmp, ged))
    g1_node = list(g1.nodes())  # generate subgraph pairs
    g2_node = list(g2.nodes())
    for k in range(len(cost_list)-1):
        if path_list[2*k] != 'None':
            g1_node.remove(path_list[2*k])
        if path_list[2*k+1] != 'None':
            g2_node.remove(path_list[2*k+1])
        if len(g1_node)<=1 or len(g2_node)<=1:
            break
        g1_base = g1.subgraph(g1_node)
        g2_base = g2.subgraph(g2_node)
        reorder_g1 = {val:str(index)  for index, val in enumerate(g1_node)}
        g1_tmp = nx.Graph()
        for (val,index) in reorder_g1.items():
            g1_tmp.add_node(index,label=val)
        count = len(g1_base)
        g1_tmp.add_node(str(count),label=superLabel)
        for (i,j) in list(g1_base.edges()):
            g1_tmp.add_edge(reorder_g1[i],reorder_g1[j])
        for node in reorder_g1.values():
            g1_tmp.add_edge(str(count),node)

        reorder_g2 = {val:str(index)  for index, val in enumerate(g2_node)}
        g2_tmp = nx.Graph()
        for (val,index) in reorder_g2.items():
            g2_tmp.add_node(index,label=val)
        count = len(g2_base)
        g2_tmp.add_node(str(count),label=superLabel)
        for (i,j) in list(g2_base.edges()):
            g2_tmp.add_edge(reorder_g2[i],reorder_g2[j])
        for node in reorder_g2.values():
            g2_tmp.add_edge(str(count),node)
        ged_tmp = ged - int(cost_list[k])
        g_list.append((g1_tmp,g2_tmp,ged_tmp))
    return g_list

def process_file_without_path(GED, trainingSet, dataSet):
    """
    Generate training data for all training graphs without edit path.
    :GED: GED loaded for training graphs.
    :trainingSet: A set of training graphs.
    :dataSet: Name of dataset to differ.
    :return g_list: A list of input training data.
    """

    g_list = []
    for i in tqdm(range(len(trainingSet))):
        real_ged = GED.loc[GED['graph1']==basename(trainingSet[i][0])]
        real_ged = real_ged[real_ged['graph2']==basename(trainingSet[i][1])]
        if real_ged.empty:
            continue
        else:
            ged = int(real_ged.iloc[0]['ged'])
        if dataSet.startswith('./GREC'):
            g1 = loadGXL(trainingSet[i][0])
            g2 = loadGXL(trainingSet[i][1])
        else:
            g1 = nx.read_gexf(trainingSet[i][0])
            g2 = nx.read_gexf(trainingSet[i][1]) 
        flag = 0  
        for node in g1.nodes(): 
            if int(node) >= len(g1) or int(g1.nodes[node]['label']) >= len(g1):
                flag = 1
        for node in g2.nodes():
            if int(node) >= len(g2) or int(g2.nodes[node]['label']) >= len(g2):
                flag = 1
        if flag == 1:
            continue
        g_list.append((g1,g2,ged)) 
    return g_list


def process_file(dataSet, trainingSet, superLabel):
    """
    Generate training data for all training graphs.
    :dataSet: The dataset directory.
    :trainingSet: A set of training graphs.
    :superLabel: The label for super node.
    :return g_list: A list of input training data.
    """
    g_list = []

    fin = open(dataSet+'ged.txt')
    lines = fin.readlines()
    fin.close()
    for i in tqdm(range(500)):
        if lines[i].startswith('['):
            if len(lines[i-1].split('\t')) != 3:
                continue
            g1_name, g2_name, distance = lines[i-1].split('\t')
            g1_abs_name = dataSet+'train/'+g1_name
            g2_abs_name = dataSet+'test/'+g2_name
            if [g1_abs_name,g2_abs_name] not in trainingSet:
                continue
            ged = int(distance)
            if ged == -1:
                continue
            if dataSet.startswith('./GREC'):
                g1 = loadGXL(g1_abs_name)
                g2 = loadGXL(g2_abs_name)
            else:
                g1 = nx.read_gexf(g1_abs_name)
                g2 = nx.read_gexf(g2_abs_name) 
            

            """
            Remove graphs in which ids are not step-by-step,
            because in scatter_add, the index must not be greater than dim.
            """
            flag = 0  
            for node in g1.nodes(): 
                if int(node) >= len(g1) or int(g1.nodes[node]['label']) >= int(superLabel):
                    flag = 1
            for node in g2.nodes():
                if int(node) >= len(g2) or int(g2.nodes[node]['label']) >= int(superLabel):
                    flag = 1
            if flag == 1:
                continue
            g1_tmp = nx.Graph(g1) # set to unfrozen
            count = len(g1)
            g1_tmp.add_node(str(count),label=superLabel)
            for node in g1.nodes():
                g1_tmp.add_edge(str(count),node)

            count = len(g2)
            g2_tmp = nx.Graph(g2) # set to unfrozen
            g2_tmp.add_node(str(count),label=superLabel)
            for node in g2.nodes():
                g2_tmp.add_edge(str(count),node)

            g_list.append((g1_tmp, g2_tmp, ged))
            #add subgraphs#
            pt = lines[i].find(']')
            path = lines[i][1:pt]
            path_list = re.findall(r'\d+|None', path)
            if pt+3 > len(lines[i]):
                continue
            cost = lines[i][pt+3:-2]
            cost_list = re.findall(r'\d+', cost)
            g1_node = list(g1.nodes())  # generate subgraph pairs
            g2_node = list(g2.nodes())
            g1_removal_node = []
            g2_removal_node = []
            for k in range(len(cost_list)-1):
                # if k == 0 and int(cost_list[k]) == 0:
                #     continue
                # if k > 0 and int(cost_list[k]) == int(cost_list[k-1]):
                #     continue
                if path_list[2*k] != 'None':
                    g1_removal_node.append(path_list[2*k])
                    g1_node.remove(path_list[2*k])
                if path_list[2*k+1] != 'None':
                    g2_removal_node.append(path_list[2*k])
                    g2_node.remove(path_list[2*k+1])
                if len(g1_node)<=1 or len(g2_node)<=1:
                    break
                '''
                Maintain edges for lower bounds
                '''
                # for node1 in g1_removal_node:
                #     for node2 in g1_removal_node:
                #         if node1 != node2:
                #             if (node1, node2) in g1.edges():
                #                 g1.remove_edge(node1,node2)
                #     if node1 in g1.nodes():
                #         flag = 0
                #         for node_n in g1.neighbors(node1):
                #             if node_n not in g1_removal_node:
                #                 flag = 1
                #         if flag == 0:
                #             g1.remove_node(node1)

                # for node1 in g2_removal_node:
                #     for node2 in g2_removal_node:
                #         if node1 != node2:
                #             if (node1, node2) in g2.edges():
                #                 g2.remove_edge(node1,node2)
                #     if node1 in g2.nodes():
                #         flag = 0
                #         for node_n in g2.neighbors(node1):
                #             if node_n not in g2_removal_node:
                #                 flag = 1
                #         if flag == 0:
                #             g2.remove_node(node1)
                # ged_tmp = ged - int(cost_list[k])
                # g_list.append((g1,g2,ged_tmp))
                '''
                For learning heuristic
                '''
                g1_base = g1.subgraph(g1_node)
                g2_base = g2.subgraph(g2_node)
                reorder_g1 = {val:str(index)  for index, val in enumerate(g1_node)}
                g1_tmp = nx.Graph()
                for (val,index) in reorder_g1.items():
                    g1_tmp.add_node(index,label=val)
                count = len(g1_base)
                g1_tmp.add_node(str(count),label=superLabel)
                for (i,j) in list(g1_base.edges()):
                    g1_tmp.add_edge(reorder_g1[i],reorder_g1[j])
                for node in reorder_g1.values():
                    g1_tmp.add_edge(str(count),node)

                reorder_g2 = {val:str(index)  for index, val in enumerate(g2_node)}
                g2_tmp = nx.Graph()
                for (val,index) in reorder_g2.items():
                    g2_tmp.add_node(index,label=val)
                count = len(g2_base)
                g2_tmp.add_node(str(count),label=superLabel)
                for (i,j) in list(g2_base.edges()):
                    g2_tmp.add_edge(reorder_g2[i],reorder_g2[j])
                for node in reorder_g2.values():
                    g2_tmp.add_edge(str(count),node)
                ged_tmp = ged - int(cost_list[k])
                g_list.append((g1_tmp,g2_tmp,ged_tmp))
    return g_list

def process_pair_test(pair,superLabel):
    """
    Generate testing data with a pair of graphs.
    :param pair: A pair of graph directory.
    :return g_list: A pair of testing graphs.
    """
    g_list = []
    """
    Remove graphs in which ids are not step-by-step,
    because in scatter_add, the index must not be greater than dim.
    """
    
    g1 = nx.read_gexf(pair[0])
    g2 = nx.read_gexf(pair[1]) 
    # g1 = loadGXL(pair[0])
    # g2 = loadGXL(pair[1])
    flag = 0  
    for node in g1.nodes(): 
        if int(node) >= len(g1):
            flag = 1
    for node in g2.nodes():
        if int(node) >= len(g2):
            flag = 1
    if flag == 1:
        return None,None
    g1_tmp = nx.Graph(g1) #set to unfrozen
    count = len(g1)
    g1_tmp.add_node(str(count),label=superLabel)
    for node in g1.nodes():
        g1_tmp.add_edge(str(count),node)

    count = len(g2)
    g2_tmp = nx.Graph(g2) #set to unfrozen
    g2_tmp.add_node(str(count),label=superLabel)
    for node in g2.nodes():
        g2_tmp.add_edge(str(count),node)
    # return g1_tmp, g2_tmp
    return g1,g2  # return for SimGNN and GMN
def process_file_node(dataSet, trainingSet, superLabel):
    """
    Generate testing data with a pair of graphs.
    :param pair: A pair of graph directory.
    :return g_list: A pair of testing graphs.
    """
    g_list = []
    fin = open(dataSet+'ged.txt')
    lines = fin.readlines()
    fin.close()
    for i in tqdm(range(len(lines))):
        if lines[i].startswith('['):
            if len(lines[i-1].split('\t')) != 3:
                    continue
            g1_name, g2_name, distance = lines[i-1].split('\t')
            g1_abs_name = dataSet+'train/'+g1_name
            g2_abs_name = dataSet+'test/'+g2_name
            if [g1_abs_name,g2_abs_name] not in trainingSet:
                continue
            ged = int(distance)
            if ged == -1:
                continue
            pt = lines[i].find(']')
            path = lines[i][1:pt]
            path_list = re.findall(r'\d+|None', path)
            g1 = nx.read_gexf(g1_abs_name)
            g2 = nx.read_gexf(g2_abs_name)
            g_list.append((g1,g2,ged,path_list))
    return g_list

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    # prediction = -math.log(prediction)
    # target = -math.log(target)
    score = (prediction-target)**2
    return score

def get_ged(g1,g2,ged,graph_pair):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    real_ged = ged.loc[ged['graph1']==basename(graph_pair[0])]
    real_ged = real_ged[real_ged['graph2']==basename(graph_pair[1])].iloc[0]
    ged = int(real_ged['ged'])
    return ged

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))
    
    return rank_corr_function(r_prediction, r_target).correlation

def calculate_prec_at_k(k, prediction, target):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[:k]
    best_k_target = target.argsort()[:k]
    
    return len(set(best_k_pred).intersection(set(best_k_target))) / k

def command(cmd, timeout=60): 
        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True) 
        t_beginning = time.time() 
        seconds_passed = 0 
        while True: 
            if p.poll() is not None: 
                break 
            seconds_passed = time.time() - t_beginning 
            if timeout and seconds_passed > timeout: 
                p.terminate() 
                return "-1\n".encode()
            time.sleep(1) 
        return p.stdout.read() 

def cal_ged(start,end,task,gpu,timeout=60):
    result = ""
    for i in range(start, end):
        cmd1 = 'CUDA_VISIBLE_DEVICES=' + gpu + ' python src/ged.py ' +  task[i][0] + ' ' + task[i][1] + ' Noah 10 0'
        output1 = command(cmd1,timeout).decode()
        cmd2 = 'python src/ged.py ' +  task[i][0] + ' ' + task[i][1] + ' LS 10 0'
        output2 = command(cmd2,timeout).decode()
        result = result + os.path.basename(task[i][0]) + ' ' + os.path.basename(task[i][1]) + ' ' + str(output1)[:-1] + ' ' + str(output2)[:-1] + '\n'
    return result

def gen_output(evaluate_set):
    jobs = []
    start = 0
    end = len(evaluate_set)
    ncpus = 16
    ngpus = 4
    timeout = 20
    step = int(end/ncpus)+1
    # print ('Evaluation: {} steps in {} cpus.'.format(step,ncpus))
    job_server = pp.Server()
    job_server.set_ncpus(ncpus)
    for i in range(0,ncpus):
        ss = int(i*step)
        ee = int(ss+step)
        gpu = str(i % ngpus)
        if ee > end:
            ee = end
        jobs.append(job_server.submit(cal_ged, (ss,ee,evaluate_set,gpu),(command,), modules=('subprocess','time',)))
    job_server.wait()
    results = ""
    for job in jobs:
        result = str(job())
        results += result
    return results

def unprocessed_cost(u, v, lower_bound, superLabel):
    # remove supernode
    supernode = None
    for node in u.nodes():
        if u.nodes[node]['label'] == superLabel:
            supernode = node
    if supernode:
        u.remove_node(supernode)

    if lower_bound == 'heuristic':
        # heuristic
        inter_node = set(u.nodes()).intersection(set(v.nodes()))
        cost = max(len(u.nodes()), len(v.nodes())) - len(inter_node)
        return cost
    elif lower_bound == 'LS':
        # Label set-based
        cross_node = 0
        u_label = []
        for node in u.nodes():
            u_label.append(u.nodes[node]['label'])
        u_label.sort()

        v_label = []
        for node in v.nodes():
            v_label.append(v.nodes[node]['label'])
        v_label.sort()

        i = 0
        j = 0
        while (i < len(u_label) and j < len(v_label)):
            if u_label[i] == v_label[j]:
                cross_node += 1
                i += 1
                j += 1
            elif u_label[i] < v_label[j]:
                i += 1
            else:
                j += 1

        node_cost = max(len(u.nodes()), len(v.nodes())) - cross_node
        edge_u = u.edges()
        edge_v = v.edges()
        inter_edge = set(edge_u).intersection(set(edge_v))
        edge_cost = max(len(edge_u), len(edge_v)) - min(len(edge_u), len(edge_v))
        cost = node_cost + edge_cost
        return cost

    elif lower_bound == 'BM':
    # Branch match-based
        cost = 0
        u_label = []
        for node in u.nodes():
            u_label.append(u.nodes[node]['label'])
        u_label.sort()

        v_label = []
        for node in v.nodes():
            v_label.append(v.nodes[node]['label'])
        v_label.sort()

        i = 0
        j = 0
        while (i < len(u_label) and j < len(v_label)):
            if u_label[i] == v_label[j] and u.edges(list(u.nodes())[i]) == v.edges(list(v.nodes())[j]):
                u_label.pop(i)
                v_label.pop(j)
                i += 1
                j += 1
            elif u_label[i] < v_label[j]:
                i += 1
            else:
                j += 1
        i = 0
        j = 0
        while (i < len(u_label) and j < len(v_label)):
            if u_label[i] == v_label[j]:
                cost += 0.5
                u_label.pop(i)
                v_label.pop(j)
                i += 1
                j += 1
            elif u_label[i] < v_label[j]:
                i += 1
            else:
                j += 1
        cost = cost + max(len(u_label), len(v_label))
        return cost
    else:
        #Star match-based
        #Initial stars from graph pair
        stars_u = []
        for node in u.nodes():
            node_list = []
            node_list.append(node)
            for k in u.neighbors(node):
                node_list.append(k)
            stars_u.append(node_list)
        
        stars_v = []
        for node in v.nodes():
            node_list = []
            node_list.append(node)
            for k in v.neighbors(node):
                node_list.append(k)
            stars_v.append(node_list)

        max_degree = 0
        for i in stars_u:
            if len(i) > max_degree:
                max_degree = len(i)

        for i in stars_v:
            if len(i) > max_degree:
                max_degree = len(i)
        # Initial cost matrix
        if len(stars_u) > len(stars_v):
            for i in range(len(stars_u)-len(stars_v)):
                stars_v.append(None)
        if len(stars_u) < len(stars_v):
            for i in range(len(stars_v)-len(stars_u)):
                stars_u.append(None)
        cost_matrix = []
        for star1 in stars_u:
            cost_tmp = []
            for star2 in stars_v:
                cost_tmp.append(star_cost(star1,star2))
            cost_matrix.append(cost_tmp)
        m = Munkres()
        indexes = m.compute(cost_matrix)
        cost = 0
        for row, column in indexes:
            value = cost_matrix[row][column]
            cost += value
        cost = cost / max(4,max_degree)
        return cost

def star_cost(p,q):
    cost = 0
    if p == None:
        cost += 2 * len(q) - 1
        return cost
    if q == None:
        cost += 2 * len(p) - 1
        return cost
    if p[0] != q[0]:
        cost += 1
    if len(p) > 1 and len(q) > 1:
        p[1:].sort()
        q[1:].sort()
        i = 1
        j = 1
        cross_node = 0
        while (i < len(p) and j < len(q)):
            if p[i] == q[j]:
                cross_node += 1
                i += 1
                j += 1
            elif p[i] < q[j]:
                i += 1
            else:
                j += 1
        cost = cost + max(len(p),len(q)) - 1 - cross_node
    cost += abs(len(q)-len(p))
    return cost
