import glob
import time
import math
import torch
import random
import subprocess
import numpy as np
import networkx as nx
import pandas as pd
import os
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv,GraphConv
from layers import AttentionModule, TenorNetworkModule, MatchingModule
from scipy.stats import spearmanr, kendalltau
from utils import process_pair, process_pair_test, process_file, process_file_node, calculate_loss, get_ged, calculate_ranking_correlation, calculate_prec_at_k
from utils import tab_printer
from parser import parameter_parser

class ReGNN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(ReGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.matching_1 = MatchingModule(self.args)
        self.matching_2 = MatchingModule(self.args)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        tmp_feature_1 = abstract_features_1
        tmp_feature_2 = abstract_features_2

        abstract_features_1 = torch.sub(tmp_feature_1,self.matching_2(tmp_feature_2))
        abstract_features_2 = torch.sub(tmp_feature_2,self.matching_1(tmp_feature_1))

        scores = torch.mm(abstract_features_1, torch.t(abstract_features_2))
        scores = torch.sigmoid(scores)

        return scores

class ReGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ReGNN(self.args, self.number_of_labels).to(self.device)
        
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers. Add a super node with the label of '10' for each graph.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.gexf")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.gexf")
        graphs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for graph in tqdm(graphs):
            data = nx.read_gexf(graph)
            for node in data.nodes():
                self.global_labels = self.global_labels.union(set(data.nodes[node]['label']))
        self.super_label = str(len(self.global_labels))
        self.global_labels.add(self.super_label)
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
        print (self.global_labels)
    
    def load_ged(self):
        """
        Loading GED from .txt file. Not useful for subgraphs.
        """
        self.ged_dir = glob.glob(self.args.dataset + "ged.txt")
        graph1 = []
        graph2 = []
        ged = []
        fin = open(self.ged_dir[0])
        for line in fin:
            if len(line.split('\t')) <= 1:
                continue
            g1, g2, distance = line.split('\t')
            graph1.append(g1)
            graph2.append(g2)
            ged.append(distance[:-1])
        fin.close()
        df = pd.DataFrame({'graph1':graph1, 'graph2':graph2, 'ged':ged})
        self.ged = df
    
    def load_ged_test(self):
        """
        Loading GED from .txt file and transfer to dataframe.
        """
        self.ged_dir = glob.glob(self.args.testing_graphs + "ged.txt")
        graph1 = []
        graph2 = []
        ged = []
        fin = open(self.ged_dir[0])
        for line in fin:
            if len(line.split('\t')) <= 1:
                continue
            if line.startswith('['):
                continue
            g1, g2, distance = line.split('\t')
            graph1.append(g1)
            graph2.append(g2)
            ged.append(distance[:-1])
        df = pd.DataFrame({'graph1':graph1, 'graph2':graph2, 'ged':ged})
        self.ged_test = df
    
    def create_data_set(self):
        self.training_graphs_set = []
        for i in range(len(self.training_graphs)):
            for j in range(len(self.testing_graphs)):
                if i != j:
                    self.training_graphs_set.append([self.training_graphs[i], self.testing_graphs[j]])

    def higher_bound(self, g1, g2):
        cost = max(len(g1.nodes()), len(g2.nodes())) + max(len(g1.edges()), len(g2.edges())) 
        return cost

    def transfer_to_torch(self, g1, g2, ged, path):
        """
        Transferring the data to torch and creating a hash table with the indices, features and target.
        :param data: NetworkX graph pairs and graph edit distance.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        
        edges_1 = [[edge[0],edge[1]] for edge in g1.edges()] + [[edge[1],edge[0]] for edge in g1.edges()]
        edges_2 = [[edge[0],edge[1]] for edge in g2.edges()] + [[edge[1],edge[0]] for edge in g2.edges()]
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)
        label_1 = [g1.nodes[node]['label'] for node in g1.nodes()]
        label_2 = [g2.nodes[node]['label'] for node in g2.nodes()]
        features_1 = torch.FloatTensor(np.array([[ 1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()] for node in label_1]))
        features_2 = torch.FloatTensor(np.array([[ 1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()] for node in label_2]))
        node_sub = np.zeros((len(label_1),len(label_2)),dtype=float)
        for i in range(len(path)):
            if (i%2 == 0) and path[i] != 'None' and path[i+1] != 'None':
                node_sub[label_1.index(path[i])][label_2.index(path[i+1])] = 1.0
        node_sub = torch.FloatTensor(node_sub)
        # print (node_sub)
        new_data["edge_index_1"] = edges_1.cuda()
        new_data["edge_index_2"] = edges_2.cuda()
        new_data["features_1"] = features_1.cuda()
        new_data["features_2"] = features_2.cuda()
        new_data["node_sub"] = node_sub.cuda()

        normalized_ged = ged / self.higher_bound(g1,g2)
        target = torch.from_numpy(np.array(normalized_ged)).float()
        # target = torch.from_numpy(np.exp(-normalized_ged).reshape(1,1)).view(-1).float()
        new_data["target"] =  target.cuda()
        return new_data

    def transfer_to_torch_test(self, g1, g2, ged, graph_pair):
        """
        Transferring the data to torch and creating a hash table with the indices, features and target.
        :param data: NetworkX graph pairs, graph edit distance dictionary and graph locations.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        
        edges_1 = [[edge[0],edge[1]] for edge in g1.edges()] + [[edge[1],edge[0]] for edge in g1.edges()]
        edges_2 = [[edge[0],edge[1]] for edge in g2.edges()] + [[edge[1],edge[0]] for edge in g2.edges()]
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)
        label_1 = [g1.nodes[node]['label'] for node in g1.nodes()]
        label_2 = [g2.nodes[node]['label'] for node in g2.nodes()]
        features_1 = torch.FloatTensor(np.array([[ 1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()] for node in label_1]))
        features_2 = torch.FloatTensor(np.array([[ 1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()] for node in label_2]))
        new_data["edge_index_1"] = edges_1.cuda()
        new_data["edge_index_2"] = edges_2.cuda()
        new_data["features_1"] = features_1.cuda()
        new_data["features_2"] = features_2.cuda()

        real_ged = ged.loc[ged['graph1']==os.path.basename(graph_pair[0])]
        real_ged = real_ged[real_ged['graph2']==os.path.basename(graph_pair[1])].iloc[0]
        normalized_ged = int(real_ged['ged']) / self.higher_bound(g1,g2)
        target = torch.from_numpy(np.array(normalized_ged)).float()
        # target = torch.from_numpy(np.exp(-normalized_ged).reshape(1,1)).view(-1).float()
        new_data["target"] =  target.cuda()
        return new_data

    def cal_loss(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of training data set.
        :return loss: Loss on the batch. 
        """
        self.optimizer.zero_grad()
        losses = 0
        for data in batch:
            target = data["node_sub"]
            prediction = self.model(data)
            loss_fn = torch.nn.MSELoss(reduction='mean')
            losses = losses + loss_fn(target, prediction)
        losses.backward(retain_graph = True)
        self.optimizer.step()
        loss = losses.item()
        return loss
    
    def preprocess_data(self):
        """
        Preprocessing training data before train. To improve util% of GPU.
        """
        print("Data preprocessing.")
        self.create_data_set()
        self.training_data_set = []
        for graph_pair in tqdm(self.training_graphs_set):
            g_list = process_pair(graph_pair, self.super_label)
            for (g1,g2,ged) in g_list:
                data = self.transfer_to_torch(g1,g2,ged)
                self.training_data_set.append(data)

    def preprocess_data_file(self):
        """
        Preprocessing training data before train. To improve its speed.
        """
        print("Data preprocessing.")
        self.create_data_set()
        self.training_data_set = []
        g_list = process_file_node(self.args.dataset, self.training_graphs_set, self.super_label)
        for (g1,g2,ged,path) in g_list:
            data = self.transfer_to_torch(g1,g2,ged,path)
            self.training_data_set.append(data)

    def create_batches(self):
        """
        Creating batches from the training data list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_data_set)       
        batches = [self.training_data_set[graph:graph+self.args.batch_size] for graph in range(0, len(self.training_data_set), self.args.batch_size)]
        return batches



    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        if os.path.exists('/home/yanglei/GraphEditDistance/remodel.pkl'):
            self.model = torch.load('/home/yanglei/GraphEditDistance/remodel.pkl')
            print("Model Loading from .pkl\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        self.preprocess_data_file()
        epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        # self.load_ged()
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                loss_score = self.cal_loss(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
        torch.save(self.model,'/home/yanglei/GraphEditDistance/remodel.pkl')
        print("\nModel saved.\n")

    def score(self):
        """
        Scoring on the test set.
        """
        print("\nModel evaluation.\n")
        self.model = torch.load('/home/yanglei/GraphEditDistance/model.pkl')
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        self.testing_graphs_set = []
        self.load_ged_test()
        for i in range(len(self.testing_graphs)):
            for j in range(len(self.testing_graphs)):
                if i != j:
                    self.testing_graphs_set.append([self.testing_graphs[i], self.testing_graphs[j]])
        batch_size = len(self.testing_graphs_set)
        target_list = []
        pre_list = []
        pre_with_a_list = []
        l1_list = []
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []

        fout = open('output.txt','w')

        for graph_pair in tqdm(self.testing_graphs_set):
            g1, g2 = process_pair_test(graph_pair, self.super_label)
            data = self.transfer_to_torch_test(g1,g2,self.ged_test,graph_pair)
            self.ground_truth.append(get_ged(g1,g2,self.ged_test,graph_pair) / self.higher_bound(g1,g2))
            target = data["target"]
            prediction = self.model(data)
            target_list.append(target)
            pre_list.append(prediction.detach())
            if target != -1:
                cmd = 'python src/ged.py ' +  graph_pair[0] + ' ' + graph_pair[1] + ' SimGNN 10'
                result = subprocess.getoutput(cmd)
                cmd = 'python src/ged.py ' +  graph_pair[0] + ' ' + graph_pair[1] + ' LS 10'
                result1 = subprocess.getoutput(cmd)
                fout.write(os.path.basename(graph_pair[0]) + ' ' + os.path.basename(graph_pair[1]) + ' '
                + str(target.item() * self.higher_bound(g1,g2)) + ' ' + str(prediction.item() * self.higher_bound(g1,g2))
                + ' ' + str(result) + ' ' + str(result1) + '\n')
                l1 = abs(target - prediction) * self.higher_bound(g1, g2)
                l1_list.append(l1.detach().cpu().numpy())
            self.scores.append(calculate_loss(prediction, target).detach().cpu().numpy())
        
        fout.close()
        count = 0
        while(count < len(self.testing_graphs_set)):
            target_batch = np.array(target_list[count:count+batch_size])
            prediction_batch = np.array(pre_list[count:count+batch_size])
            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_batch, target_batch))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_batch, target_batch))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_batch, target_batch))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_batch, target_batch))
            count += batch_size
        self.rho = np.mean(rho_list)
        self.tau = np.mean(tau_list)
        self.prec_at_10 = np.mean(prec_at_10_list)
        self.prec_at_20 = np.mean(prec_at_20_list)
        self.l1 = np.mean(l1_list)

        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error= np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error,5))+".")
        print("mse error: " +str(round(model_error,5))+".")
        print("l1 error: " +str(round(self.l1,5))+".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = ReGNNTrainer(args)
    trainer.fit()
    # trainer.score()
    
if __name__ == "__main__":
    main()