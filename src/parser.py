import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run.")

    parser.add_argument("--dataset",
                        nargs = "?",
                        default = "./AIDS700nef/",
	                help = "Folder of dataset, including training and testing graphs.")

    parser.add_argument("--training-graphs",
                        nargs = "?",
                        default = "./AIDS700nef/train/",
	                help = "Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs = "?",
                        default = "./AIDS700nef/test/",
	                help = "Folder with testing graph pair jsons.")

    parser.add_argument("--global-labels",
                        nargs = "?",
                        default = "AIDS.pkl",
                    help = "Global labels in last training process.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 5,
	                help = "Number of training epochs. Default is 5.")

    parser.add_argument("--filters-1",
                        type = int,
                        default = 128,
	                help = "Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--filters-2",
                        type = int,
                        default = 64,
	                help = "Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--filters-3",
                        type = int,
                        default = 32,
	                help = "Filters (neurons) in 3rd convolution. Default is 32.")

    parser.add_argument("--tensor-neurons",
                        type = int,
                        default = 8,
	                help = "Neurons in tensor network layer. Default is 8.")

    parser.add_argument("--bottle-neck-neurons",
                        type = int,
                        default = 4,
	                help = "Bottle neck layer neurons. Default is 4.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 128,
	                help = "Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type = int,
                        default = 16,
	                help = "Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.001,
	                help = "Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 5*10**-4,
	                help = "Adam weight decay. Default is 5*10^-4.")
    """
    Histogram used in SimGNN. Default not use.
    """
    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true") 
    """
    Switch evaluation.
    """
    parser.add_argument("--combinatorial",
                        dest="combinatorial",
                        action="store_true")

    parser.add_argument("--gnn-operator",
                        nargs = "?",
                        default = "gcn",
	                help = "Type of GNN-Operator. Default is gcn")

    parser.add_argument("--beamsize",
                        dest="beamsize",
                        action="store_true")

    parser.set_defaults(histogram=False)
    parser.set_defaults(combinatorial=False)
    parser.set_defaults(beamsize=False)

    return parser.parse_args()
