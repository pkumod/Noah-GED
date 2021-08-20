from utils import tab_printer
from gpn import GPNTrainer
from parser import parameter_parser
from fine_tune import finetune_GPN
import networkx as nx

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a GPN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = GPNTrainer(args)
    # trainer.fit()
    """
    Scoring on the prediction and learning ability.
    """
    trainer.score()
    """
    Scoring on the subgraph test set.
    """
    # trainer.score2()
    """
    Scoring on the generalization ability.
    """
    # trainer.score3()
    """
    Finetuning for downstream tasks.
    """
    # model = finetune_GPN(args, trainer.number_of_labels)
    # model.finetune()
if __name__ == "__main__":
    main()
