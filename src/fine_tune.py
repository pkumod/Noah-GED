import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor, embedding
from collections import OrderedDict 
from gpn import GPN
import networkx as nx
import numpy as np
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class finetune_GPN(GPN):
    def __init__(self, args, number_labels):
        super(finetune_GPN, self).__init__(args, number_labels)
        self.args = args
        self.number_of_labels = number_labels
        global_labels_file = open(self.args.global_labels,'rb')
        self.global_labels = pickle.load(global_labels_file)
        self.super_label = str(len(self.global_labels))

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
            scores = torch.cat((pooled_features_1,pooled_features_1),dim=1).view(1,-1)
            user_setting = [1,0,0,0,0,1]
            user_setting = user_setting.cuda()
            scores = torch.cat((scores,user_setting),dim=1).view(1,-1)      

        if self.args.histogram == True:
            scores = torch.cat((scores,hist),dim=1).view(1,-1)
            
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return pooled_features_1,pooled_features_2

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

    def process_data(self):
        X = []
        y = []
        fin = open(self.args.dataset + 'start.txt')
        lines = fin.readlines()
        fin.close()
        
        for i in tqdm(range(len(lines))):
            g1_name, g2_name, label = lines[i].split('\t')
            g1 = nx.read_gexf(self.args.training_graphs + g1_name)
            g2 = nx.read_gexf(self.args.testing_graphs + g2_name)
            """
            Remove graphs in which ids are not step-by-step,
            because in scatter_add, the index must not be greater than dim.
            """
            flag = 0  
            for node in g1.nodes(): 
                if int(node) >= len(g1) or int(g1.nodes[node]['label']) >= int(self.super_label):
                    flag = 1
            for node in g2.nodes():
                if int(node) >= len(g2) or int(g2.nodes[node]['label']) >= int(self.super_label):
                    flag = 1
            if flag == 1:
                continue
            data = self.transfer_to_torch(g1,g2,12)
            embedding1, embedding2 = self.model.forward(data)
            embedding = torch.cat((torch.t(embedding1),torch.t(embedding2)),dim=1).view(1,-1).detach().cpu().numpy().tolist()[0]
            X.append(embedding)
            y.append(int(label))
        return X,y

    def finetune(self):
        """
        Finetuning the start node.
        """
        self.device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print ("\nCurrent device is {}".format(self.device))
        self.model = finetune_GPN(self.args, self.number_of_labels).to(self.device)
        print("\nModel finetuning.\n")
        if os.path.exists('/home/yanglei/GraphEditDistance/model.pkl'):
            model_load = torch.load('/home/yanglei/GraphEditDistance/model.pkl')
            print("Model Loading from .pkl\n")
        optimizer = torch.optim.Adam(model_load.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        model_load_dict = model_load.state_dict()
        self.model_dict = self.model.state_dict()
        self.model_dict.update(model_load_dict)
        self.model.load_state_dict(model_load_dict)
        # g1 = nx.read_gexf('2692.gexf')
        # g2 = nx.read_gexf('2693.gexf')
        # data = self.transfer_to_torch(g1,g2,12)
        # embedding1, embedding2 = self.model.forward(data)
        # embedding = torch.cat((torch.t(embedding1),torch.t(embedding2)),dim=1).view(1,-1)
        # print(embedding)
        torch.save(self.model,'/home/yanglei/GraphEditDistance/pretrained_model.pkl')
        X, y = self.process_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(16, 4), random_state=1, verbose=1)
        clf.fit(X_train, y_train)
        with open('start_node_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
