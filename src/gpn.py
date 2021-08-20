import glob
import time
import math
import torch
import random
import subprocess
import numpy as np
import networkx as nx
import pandas as pd
import pickle
import os
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv,GraphConv,GINConv
from torch_geometric.data import DataLoader, Batch
from layers import AttentionModule, TenorNetworkModule, MatchingModule
from scipy.stats import spearmanr, kendalltau
from utils import process_pair, process_pair_test, process_file, process_file_without_path, calculate_loss, get_ged, calculate_ranking_correlation, calculate_prec_at_k, loadGXL, gen_output, unprocessed_cost
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class GPN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GPN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.user_features = 6
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """

        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        elif self.args.beamsize == True:
            self.feature_count = self.args.filters_3 + self.user_features
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))
            
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))
            
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        
        self.matching_1 = MatchingModule(self.args)
        self.matching_2 = MatchingModule(self.args)

        self.attention = AttentionModule(self.args)
        # self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        # self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        # self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        # self.matching_1 = MatchingModule(self.args)
        # self.matching_2 = MatchingModule(self.args)
        # self.attention = AttentionModule(self.args)
        if self.args.beamsize == False:
            self.tensor_network = TenorNetworkModule(self.args)
            self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
            self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        
        # Elastic beam size
        else:
            self.fully_connect_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
            self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)


    # histogram for SimGNN
    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1,1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1,-1)
        return hist

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

        abstract_features_1 = torch.abs(abstract_features_1)
        abstract_features_2 = torch.abs(abstract_features_2)

        if self.args.histogram == True:
            hist =self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))
        
        pooled_features_1 =  self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        if self.args.beamsize == False:
            scores = self.tensor_network(pooled_features_1, pooled_features_2)
            scores = torch.t(scores)
        else:
            scores = torch.cat((torch.t(pooled_features_1),torch.t(pooled_features_1)),dim=1).view(1,-1)
            user_setting = [1,0,0,0,0,1]
            user_setting = user_setting.cuda()
            scores = torch.cat((scores,user_setting),dim=1).view(1,-1)      

        if self.args.histogram == True:
            scores = torch.cat((scores,hist),dim=1).view(1,-1)
            
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

class GPNTrainer(object):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        """
        Creating a model.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = GPN(self.args, self.number_of_labels).to(self.device)
        
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers. Add a super node for each graph.
        """
        print("\nEnumerating unique labels.\n")
        if os.path.exists(self.args.global_labels):
            if self.args.global_labels == 'GREC.pkl':
                self.training_graphs = glob.glob(self.args.training_graphs + "*.gxl")
                self.testing_graphs = glob.glob(self.args.testing_graphs + "*.gxl")
            else:
                self.training_graphs = glob.glob(self.args.training_graphs + "*.gexf")
                self.testing_graphs = glob.glob(self.args.testing_graphs + "*.gexf")

            global_labels_file = open(self.args.global_labels,'rb')
            self.global_labels = pickle.load(global_labels_file)
            self.super_label = str(len(self.global_labels)-1)
            self.number_of_labels = len(self.global_labels)
            print ("super label:",self.super_label)
            print (self.global_labels)
        else:
            if self.args.global_labels == 'GREC.pkl':
                self.training_graphs = glob.glob(self.args.training_graphs + "*.gxl")
                self.testing_graphs = glob.glob(self.args.testing_graphs + "*.gxl")
            else:
                self.training_graphs = glob.glob(self.args.training_graphs + "*.gexf")
                self.testing_graphs = glob.glob(self.args.testing_graphs + "*.gexf")
            graphs = self.training_graphs + self.testing_graphs
            self.global_labels = set()
            for graph in tqdm(graphs):
                data = nx.read_gexf(graph)
                # data = loadGXL(graph)
                for node in data.nodes():
                    self.global_labels.add(data.nodes[node]['label'])
            self.super_label = str(len(self.global_labels))
            print ("super label:",self.super_label)
            self.global_labels.add(self.super_label)
            self.global_labels = list(self.global_labels)
            self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
            self.number_of_labels = len(self.global_labels)
            global_labels_file = open(self.args.global_labels,'wb')
            pickle.dump(self.global_labels,global_labels_file)
            print (self.global_labels)
    
    def load_ged(self):
        """
        Loading GED from .txt file. Not useful for subgraphs.
        """
        self.ged_dir = glob.glob(self.args.training_graphs + "ged.txt")
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
        self.ged_dir = glob.glob(self.args.testing_graphs + "ged.txt") #learning ability: self.args.testing_graphs 
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
        df = pd.DataFrame({'graph1':graph1, 'graph2':graph2, 'ged':ged}) # There might have some bug in the order of graph1 and graph2
        self.ged_test = df
    
    def create_data_set(self):
        self.training_graphs_set = []
        if self.args.global_labels == 'Syn.pkl':
            for i in range(8000):
                graph1 = self.args.training_graphs+str(i)+'_1.gexf'
                graph2 = self.args.training_graphs+str(i)+'_2.gexf'
                graph3 = self.args.training_graphs+str(i)+'_3.gexf'
                self.training_graphs_set.append([graph1, graph2])
                self.training_graphs_set.append([graph1, graph3])
            random.shuffle(self.training_graphs_set)
        else:
            training_source = self.training_graphs.copy()
            random.shuffle(training_source)
            training_target = training_source.copy()
            random.shuffle(training_target)
            for i in range(len(self.training_graphs)):
                if training_source[i] != training_target[i]:
                    self.training_graphs_set.append([training_source[i], training_target[i]])

    def higher_bound(self, g1, g2):
        cost = max(len(g1.nodes()), len(g2.nodes())) + max(len(g1.edges()), len(g2.edges())) 
        return cost

    def transfer_to_torch(self, g1, g2, ged):
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
        new_data["edge_index_1"] = edges_1.cuda()
        new_data["edge_index_2"] = edges_2.cuda()
        new_data["features_1"] = features_1.cuda()
        new_data["features_2"] = features_2.cuda()

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
        real_ged = real_ged[real_ged['graph2']==os.path.basename(graph_pair[1])]
        if real_ged.empty:
            return None
        else:
            real_ged = real_ged.iloc[0]
            normalized_ged = int(real_ged['ged']) #/ self.higher_bound(g1,g2)
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
            target = data["target"]
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(target, prediction)
        # triplet loss
        # for i in range(int(len(batch)/2)):
        #     prediction1 = self.model(batch[2*i])
        #     target1 = self.model(batch[2*i])
        #     prediction2 = self.model(batch[2*i+1])
        #     target2 = self.model(batch[2*i+1])
        #     losses = losses + torch.nn.functional.mse_loss(target1, prediction1) + torch.nn.functional.mse_loss(target2, prediction2) + max(0,prediction1-prediction2+1)
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
        # g_list = process_file(self.args.training_graphs, self.training_graphs_set, self.super_label)
        # For training SimGNN and GMN
        self.load_ged()
        g_list = process_file_without_path(self.ged, self.training_graphs_set, self.args.training_graphs)
        for (g1,g2,ged) in g_list:
            data = self.transfer_to_torch(g1,g2,ged)
            self.training_data_set.append(data)

    def process_test_data(self):
        """
        Processing testing data for subgraph testing.
        """
        print("Preparing for subgraph testing.")
        self.testing_graphs_set = []
        for i in self.training_graphs:
            for j in self.testing_graphs:
                self.testing_graphs_set.append([i,j])
        self.testing_data_set = []
        g_list = process_file(self.args.dataset, self.testing_graphs_set, self.super_label)
        self.testing_data_set = g_list

    def process_syn_data(self):
        """
        Processing testing data for synthetic dataset testing.
        """
        print("Preparing for synthetic dataset testing.")
        self.testing_graphs_set = []
        for i in range(2000):
            graph1 = self.args.testing_graphs+str(i)+'_1.gexf'
            graph2 = self.args.testing_graphs+str(i)+'_2.gexf'
            graph3 = self.args.testing_graphs+str(i)+'_3.gexf'
            self.testing_graphs_set.append([graph1, graph2])
            self.testing_graphs_set.append([graph1, graph3])
        self.testing_data_set = []
        g_list = process_file_without_path(self.ged_test, self.testing_graphs_set, self.args.testing_graphs)
        self.testing_data_set = g_list

    def create_batches(self):
        """
        Creating batches from the training data list.
        :return batches: List of lists with batches.
        """
        # random.shuffle(self.training_data_set)  
        batches = [self.training_data_set[graph:graph+self.args.batch_size] for graph in range(0, len(self.training_data_set), self.args.batch_size)]
        return batches

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        if os.path.exists('/home/yanglei/GraphEditDistance/model.pkl'):
            self.model = torch.load('/home/yanglei/GraphEditDistance/model.pkl')
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
        torch.save(self.model,'/home/yanglei/GraphEditDistance/model.pkl')
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
        print ("Valid data length:", len(self.ged_test))
        for i in range(len(self.testing_graphs)):
            for j in range(len(self.testing_graphs)):
                if i != j:
                    self.testing_graphs_set.append([self.testing_graphs[i], self.testing_graphs[j]])
        target_list = []
        pre_list = []
        l1_list = []

        if self.args.combinatorial == True:
            # for graph_pair in tqdm(self.testing_graphs_set):
            #     g1, g2 = process_pair_test(graph_pair, self.super_label)
            #     if (g1,g2) == (None, None):
            #         continue
            #     data = self.transfer_to_torch_test(g1,g2,self.ged_test,graph_pair)
            #     if data == None:
            #         continue
            #     self.ground_truth.append(get_ged(g1,g2,self.ged_test,graph_pair) / self.higher_bound(g1,g2))
            #     target = data["target"]
            #     prediction = self.model(data)
            #     target_list.append(target)
            #     pre_list.append(prediction.detach())
            #     if target != -1:
            #         cmd = 'python src/ged.py ' +  graph_pair[0] + ' ' + graph_pair[1] + ' Noah 10'
            #         result = subprocess.getoutput(cmd)
            #         cmd = 'python src/ged.py ' +  graph_pair[0] + ' ' + graph_pair[1] + ' LS 10'
            #         result1 = subprocess.getoutput(cmd)
            #         fout.write(os.path.basename(graph_pair[0]) + ' ' + os.path.basename(graph_pair[1]) + ' '
            #         + str(target.item() * self.higher_bound(g1,g2)) + ' ' + str(prediction.item() * self.higher_bound(g1,g2))
            #         + ' ' + str(result) + ' ' + str(result1) + '\n')
            #         l1 = abs(target - prediction) * self.higher_bound(g1, g2)
            #         l1_list.append(l1.detach().cpu().numpy())
            #     self.scores.append(calculate_loss(prediction, target).detach().cpu().numpy())      
            start = time.time()
            ss = 0
            step = 5000
            ee = ss + step
            total = math.ceil(len(self.testing_graphs_set) / step)
            while (ss < len(self.testing_graphs_set)):
                results = gen_output(self.testing_graphs_set[ss:ee])
                fout = open('output_AIDS_new.txt','a')
                fout.write(results)
                fout.close()
                ss = ee
                ee = ss + step
                print (str(int(ss/step))+'/'+str(total)+' of all evaluation has been done.')
            end = time.time()
            print ('Used time for evaluation: %.2f seconds' % (end-start))
        else:

            # Evaluation for learning based methods.
            # rho_list = []
            # tau_list = []
            # prec_at_10_list = [] 
            # prec_at_20_list = []
            # l2_list = []
            # acc = 0
            # fea = 0
            # count = 0
            # t = tqdm(total=len(self.training_graphs)*len(self.testing_graphs))
            # for i in range(len(self.testing_graphs)):
            #     target_batch = np.empty(len(self.training_graphs))
            #     prediction_batch = np.empty(len(self.training_graphs))
            #     for j in range(len(self.training_graphs)):
            #         graph_pair = self.testing_graphs[i], self.training_graphs[j]
            #         g1, g2 = process_pair_test(graph_pair, self.super_label)
            #         if (g1,g2) == (None, None):
            #             continue
            #         data = self.transfer_to_torch_test(g1,g2,self.ged_test,graph_pair)
            #         if data == None:
            #             continue
            #         self.ground_truth.append(get_ged(g1,g2,self.ged_test,graph_pair) / self.higher_bound(g1,g2))
            #         target = data["target"]
            #         prediction = self.model(data)
            #         target_batch[j] = target
            #         prediction_batch[j] = prediction.detach()
            #         if target > 0:
            #             l1 = abs(target - prediction) * self.higher_bound(g1, g2)
            #             l1_list.append(l1.detach().cpu().numpy())
            #             l2 = ((target - prediction) * self.higher_bound(g1, g2))**2
            #             l2_list.append(l2.detach().cpu().numpy())
            #             if (round(prediction.item()*self.higher_bound(g1, g2)) == round(target.item()*self.higher_bound(g1, g2))):
            #                 acc += 1
            #             if (round(prediction.item()*self.higher_bound(g1, g2)) >= round(target.item()*self.higher_bound(g1, g2))):
            #                 fea += 1
            #             count += 1
            #         self.scores.append(calculate_loss(prediction, target).detach().cpu().numpy())
            #     rho_list.append(calculate_ranking_correlation(spearmanr, prediction_batch, target_batch))
            #     tau_list.append(calculate_ranking_correlation(kendalltau, prediction_batch, target_batch))
            #     prec_at_10_list.append(calculate_prec_at_k(10, prediction_batch, target_batch))
            #     prec_at_20_list.append(calculate_prec_at_k(20, prediction_batch, target_batch))

            #     t.update(len(self.training_graphs))
            # self.rho = np.mean(rho_list)
            # self.tau = np.mean(tau_list)
            # self.prec_at_10 = np.mean(prec_at_10_list)
            # self.prec_at_20 = np.mean(prec_at_20_list)
            # self.mae = np.mean(l1_list)
            # self.mse = np.mean(l2_list)
            # print (round(acc/count,5), round(fea/count,5))
            # self.print_evaluation()
            # exit(0)

        # norm_ged_mean = np.mean(self.ground_truth)
        # base_error= np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        # model_error = np.mean(self.scores)
        # self.l1 = np.mean(l1_list)
        # print("Baseline error: " +str(round(base_error,5))+".")
        # print("mse error: " +str(round(model_error,5))+".")
        # print("l1 error: " +str(round(self.l1,5))+".")
        
            fin = open('output_AIDS_new.txt','r')
            graph1 = []
            graph2 = []
            ged_truth = []
            ged_combine = []
            ged_beamsearch = []
            count = 0
            acc_comb = 0
            acc_beam = 0
            lines = fin.readlines()
            for line in tqdm(lines):
                g1 = line.split(' ')[0]
                g2 = line.split(' ')[1]
                ged_comb = line.split(' ')[2]
                ged_beam = line.split(' ')[3][:-1]
                real_ged = self.ged_test.loc[self.ged_test['graph1']==g1]
                real_ged = real_ged[real_ged['graph2']==g2]
                if real_ged.empty:
                    ged = -1
                else:
                    ged = int(real_ged.iloc[0]['ged'])
                if int(ged) < 0 or int(ged_comb) < 0 or int(ged_beam) < 0:
                    continue
                graph1.append(g1)
                graph2.append(g2)
                ged_truth.append(int(ged))
                ged_combine.append(int(ged_comb))
                ged_beamsearch.append(int(ged_beam))
                count += 1
                if int(ged) >= int(ged_beam):
                    acc_beam += 1
                if int(ged) >= int(ged_comb):
                    acc_comb += 1
            fin.close()
            print (round(acc_beam/count,5), round(acc_comb/count,5))
            df = pd.DataFrame({'graph1':graph1, 'graph2':graph2, 'ged_truth':ged_truth, 
            'ged_combine':ged_combine, 'ged_beam':ged_beamsearch })
            df['mae_combine'] = df.apply(lambda row: abs(row['ged_truth']-row['ged_combine']), axis=1)
            df['mse_combine'] = df.apply(lambda row: (row['ged_truth']-row['ged_combine'])**2, axis=1)
            df['mae_beam'] = df.apply(lambda row: abs(row['ged_truth']-row['ged_beam']), axis=1)
            df['mse_beam'] = df.apply(lambda row: (row['ged_truth']-row['ged_beam'])**2, axis=1)


            evaluation_list = ['combine','beam']
            for i in range(2):
                print ('\nEvaluation for '+evaluation_list[i]+':')
                rho_list = []
                tau_list = []
                prec_at_10_list = [] 
                prec_at_20_list = []
                df_size = []
                for index in range(len(self.testing_graphs)):
                    training_graph = os.path.basename(self.testing_graphs[index])
                    df_tmp = df.loc[df['graph1']==training_graph]
                    if df_tmp.empty:
                        continue
                    df_size.append(len(df_tmp))
                    rho_list.append(calculate_ranking_correlation(spearmanr, df_tmp['ged_'+evaluation_list[i]], df_tmp['ged_truth']))
                    tau_list.append(calculate_ranking_correlation(kendalltau, df_tmp['ged_'+evaluation_list[i]], df_tmp['ged_truth']))
                    if (len(df_tmp)>10):
                        prec_at_10_list.append(calculate_prec_at_k(10, df_tmp['ged_'+evaluation_list[i]], df_tmp['ged_truth']))
                    else:
                        prec_at_10_list.append(1.0)
                    if (len(df_tmp)>20):    
                        prec_at_20_list.append(calculate_prec_at_k(20, df_tmp['ged_'+evaluation_list[i]], df_tmp['ged_truth']))
                    else:
                        prec_at_20_list.append(1.0)
                self.mae = np.mean(df['mae_'+evaluation_list[i]])
                self.mse = np.mean(df['mse_'+evaluation_list[i]])
                self.rho = np.nanmean(rho_list)
                self.tau = np.nanmean(tau_list)
                self.prec_at_10 = np.mean(prec_at_10_list)
                self.prec_at_20 = np.mean(prec_at_20_list)
                self.print_evaluation()

    def score2(self):
        """
        Scoring on the subgraph test set.
        """
        print("\nModel evaluation.\n")
        l1_list = []
        l2_list = []
        acc = 0
        fea = 0
        count = 0
        self.process_test_data()
        
        '''
        Learning-based methods
        '''
        self.model = torch.load('/home/yanglei/GraphEditDistance/model_IMDB.pkl')
        self.model.eval()
        for i in tqdm(range(len(self.testing_data_set))):
            (g1,g2,ged) = self.testing_data_set[i]
            data = self.transfer_to_torch(g1,g2,ged)
            target = data["target"]
            prediction = self.model(data)
            if target > 0:
                l1 = abs(target - prediction) * self.higher_bound(g1, g2)
                l1_list.append(l1.detach().cpu().numpy())
                l2 = ((target - prediction) * self.higher_bound(g1, g2))**2
                l2_list.append(l2.detach().cpu().numpy())
                if (round(prediction.item()*self.higher_bound(g1, g2)) == round(target.item()*self.higher_bound(g1, g2))):
                    acc += 1
                if (round(prediction.item()*self.higher_bound(g1, g2)) >= round(target.item()*self.higher_bound(g1, g2))):
                    fea += 1
                count += 1
        self.mae = np.mean(l1_list)
        self.mse = np.mean(l2_list)
        print("accuracy: " + str(round(acc/count, 5)) + ".")
        print("feasibility: " + str(round(fea/count, 5)) + ".")
        print("mae: " + str(round(self.mae, 5)) + ".")
        print("mse: " + str(round(self.mse, 5)) + ".")
        '''
        Alg-based methods
        '''
        methods = ['LS','heuristic','BM','SM']
        for k in range(len(methods)):
            l1_list = []
            l2_list = []
            acc = 0
            fea = 0
            count = 0
            for i in tqdm(range(len(self.testing_data_set))):
                (g1,g2,ged) = self.testing_data_set[i]
                cost = unprocessed_cost(g1,g2,methods[k],self.super_label)
                l1_list.append(abs(cost-ged))
                l2_list.append((cost-ged)**2)
                if cost == ged:
                    acc += 1
                if cost >= ged:
                    fea += 1
                count += 1
            
            self.mae = np.mean(l1_list)
            self.mse = np.mean(l2_list)
            print("Evaluation for " + methods[k] + " lower bound:")
            print("accuracy: " + str(round(acc/count, 5)) + ".")
            print("feasibility: " + str(round(fea/count, 5)) + ".")
            print("mae: " + str(round(self.mae, 5)) + ".")
            print("mse: " + str(round(self.mse, 5)) + ".")
    
    def score3(self):
        """
        Scoring on the synthetic dataset.
        """
        print("\nModel evaluation.\n")
        l1_list = []
        l2_list = []
        pred_list = []
        acc = 0
        fea = 0
        acc2 = 0
        count = 0
        self.load_ged_test()
        self.process_syn_data()
        
        '''
        Learning-based methods
        '''
        self.model = torch.load('/home/yanglei/GraphEditDistance/model.pkl')
        self.model.eval()
        for i in tqdm(range(len(self.testing_data_set))):
            (g1,g2,ged) = self.testing_data_set[i]
            data = self.transfer_to_torch(g1,g2,ged)
            target = data["target"]
            prediction = self.model(data)
            if target > 0:
                l1 = abs(target - prediction* self.higher_bound(g1, g2)) * self.higher_bound(g1, g2)
                l1_list.append(l1.detach().cpu().numpy())
                l2 = ((target - prediction* self.higher_bound(g1, g2)) * self.higher_bound(g1, g2))**2
                l2_list.append(l2.detach().cpu().numpy())
                pred = prediction * self.higher_bound(g1, g2)*self.higher_bound(g1, g2)
                pred_list.append(pred.detach().cpu())
                if (round(prediction.item()*self.higher_bound(g1, g2)) == round(target.item()*self.higher_bound(g1, g2))):
                    acc += 1
                if (round(prediction.item()*self.higher_bound(g1, g2)) >= round(target.item()*self.higher_bound(g1, g2))):
                    fea += 1
                count += 1
        self.mae = np.mean(l1_list)
        self.mse = np.mean(l2_list)
        print (pred_list)
        for i in range(1000):
            if pred_list[2*i] < pred_list[2*i+1]:
                acc2 += 1
        print("accuracy: " + str(round(acc/count, 5)) + ".")
        print("feasibility: " + str(round(fea/count, 5)) + ".")
        print("mae: " + str(round(self.mae, 5)) + ".")
        print("mse: " + str(round(self.mse, 5)) + ".")
        print("triplet accuracy: " + str(round(acc2/count, 5)) + ".")

    def print_evaluation(self):
        """
        Printing the error rates.
        Baseline is the simple mean as the prediction.
        """
        print("mae: " + str(round(self.mae, 5)) + ".")
        print("mse: " + str(round(self.mse, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")

