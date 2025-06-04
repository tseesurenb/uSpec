'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing

args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This will be /home/madmin/uSpec

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['uspec']

# Aligned naming convention
config['train_u_batch_size'] = args.train_u_batch_size  # Training batch size
config['eval_u_batch_size'] = args.eval_u_batch_size    # Evaluation batch size
config['dataset'] = args.dataset
config['lr'] = args.lr
config['decay'] = args.decay
config['n_eigen'] = args.n_eigen
config['epochs'] = args.epochs
config['filter'] = args.filter
config['filter_order'] = args.filter_order
config['verbose'] = args.verbose
config['val_ratio'] = args.val_ratio
config['patience'] = args.patience
config['min_delta'] = args.min_delta
config['n_epoch_eval'] = args.n_epoch_eval
config['m_type'] = args.m_type

# Backward compatibility (optional)
config['u_batch'] = args.train_u_batch_size  # Alias for old code
config['test_u_batch_size'] = args.eval_u_batch_size  # Alias for old code

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


TRAIN_epochs = args.epochs
topks = eval(args.topks)


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗   ██╗███████╗██████╗ ███████╗ ██████╗
██║   ██║██╔════╝██╔══██╗██╔════╝██╔════╝
██║   ██║███████╗██████╔╝█████╗  ██║     
██║   ██║╚════██║██╔═══╝ ██╔══╝  ██║     
╚██████╔╝███████║██║     ███████╗╚██████╗
 ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝
                                         
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
