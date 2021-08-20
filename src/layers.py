import torch

class AttentionModule(torch.nn.Module):
    """
    Attention Module for Graph Level Embedding.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3)) 
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN/GIN.
        :return representation: A graph level representation vector. 
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding,transformed_global.view(-1,1)))
        representation = torch.mm(torch.t(embedding),sigmoid_scores)
        return representation

class MatchingModule(torch.nn.Module):
    """
    Graph-to-graph Module to gather cross-graph information.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MatchingModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3)) 
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN/GIN.
        :return representation: A graph level representation vector. 
        """
        global_context = torch.sum(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        return transformed_global  

class TenorNetworkModule(torch.nn.Module):
    """
    Tensor Network module for similarity vector calculation.
    """
    def __init__(self,args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2*self.args.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_3,-1)).view(self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring  + self.bias)
        return scores

# TODO
# class DiffPool(torch.nn.Module):
#     def __init__(self, args, num_nodes = 10, num_layers = 4, hidden = 16, ratio=0.25):
#         super(DiffPool, self).__init__()
        
#         self.args = args
#         num_features = self.args.filters_3
        
#         self.att = DenseAttentionModule(self.args)
        
#         num_nodes = ceil(ratio * num_nodes)
#         self.embed_block1 = Block(num_features, hidden, hidden)
#         self.pool_block1 = Block(num_features, hidden, num_nodes)

#         self.embed_blocks = torch.nn.ModuleList()
#         self.pool_blocks = torch.nn.ModuleList()
#         for i in range((num_layers // 2) - 1):
#             num_nodes = ceil(ratio * num_nodes)
#             self.embed_blocks.append(Block(hidden, hidden, hidden))
#             self.pool_blocks.append(Block(hidden, hidden, num_nodes))
#         self.jump = JumpingKnowledge(mode='cat')
#         self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
#         self.lin2 = Linear(hidden, num_features)