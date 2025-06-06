'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Updated with DySimGCF-style parameters for model_double

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing

args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'ml-100k']
all_models  = ['uspec']

# Basic training parameters
config['train_u_batch_size'] = args.train_u_batch_size
config['eval_u_batch_size'] = args.eval_u_batch_size
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

# Filter design options
config['filter_design'] = args.filter_design
config['init_filter'] = args.init_filter
config['run_convergence_test'] = args.run_convergence_test

# DySimGCF-style parameters (for model_double)
config['u_sim'] = args.u_sim
config['i_sim'] = args.i_sim
config['u_K'] = args.u_K
config['i_K'] = args.i_K
config['self_loop'] = args.self_loop

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