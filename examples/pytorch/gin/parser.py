"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        # dataset
        self.parser.add_argument(
            '--dataset', type=str, default="MUTAG",
            choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI'],
            help='name of dataset (default: Graph obtained from OB Representation of each house)')
        self.parser.add_argument(
            '--batch_size', type=int, default=256,
            help='batch size for training and validation (default: 32)')
        self.parser.add_argument(
            '--fold_idx', type=int, default=0,
            help='the index(<10) of fold in 10-fold validation.')
        self.parser.add_argument(
            '--filename', type=str, default="",
            help='output file')

        # device
        self.parser.add_argument(
            '--disable-cuda', action='store_true',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=0,
            help='which gpu device to use (default: 0)')

        # net
        self.parser.add_argument(
            '--num_layers', type=int, default=3,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=2,
            help='number of MLP layers(default: 2). 1 means linear model.')
        self.parser.add_argument(
            '--hidden_dim', type=int, default=64,
            help='number of hidden units (default: 64)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--epochs', type=int, default=500,
            help='number of epochs to train (default: 350)')
        self.parser.add_argument(
            '--lr', type=float, default=0.001,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0.5,
            help='final layer dropout (default: 0.5)')
        self.parser.add_argument(
            '--save_embeddings', type=bool, default=False,
            help='Save the graph final layer embeddings using Shuffle = False')
        self.parser.add_argument(
            '--nb_classes', type=int, default=15,
            help='Num of output classes')
        self.parser.add_argument(
            '--input_features', type=int, default=4,
            help='Input graph features')
        self.parser.add_argument(
            '--split_ratio', type=float, default=0.2,
            help='Split Ratio of val data ratio from train data')
        self.parser.add_argument(
            '--num_workers', type=int, default=10,
            help='Split Ratio of val data ratio from train data')


        # self.parser.add_argument(
        #     '--house_start_end_dict', type=dict, default=  {'houseB': (33784, 35224),
        #                                                     'houseC': (51052, 51123), 'houseA': (77539, 78960),
        #                                                     'ordonezA': (114626, 115919)}, help='Input graph features
        self.parser.add_argument(
                '--house_start_end_dict', type=dict, default= {'ordonezB': (767, 892), 'houseB': (4039, 4058),
                                                                'houseC': (4636, 4645), 'houseA': (7254, 7267),
        'ordonezA': (8429, 8453)}, help='Input graph features')

        # done
        self.args = self.parser.parse_args()
