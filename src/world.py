'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import sys
import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This will be /home/madmin/uSpec

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['cf', 'uspec']
config['multicore'] = args.multicore
config['dataset'] = args.dataset
config['test_u_batch_size'] = args.testbatch
config['lr'] = args.lr
config['decay'] = args.decay
config['i_K'] = args.i_K
config['u_K'] = args.u_K
config['n_eigen'] = args.n_eigen
config['s_temp'] = args.s_temp
config['epochs']= args.epochs
config['filter']= args.filter
config['filter_order'] = 4  # Polynomial order for spectral filters
config['spectral_reg'] = 1e-5  # Regularization for filter coefficients
config['verbose']= args.verbose
config['A_split'] = False
config['A_n_fold'] = args.a_fold
config['n_epoch_eval'] = args.n_epoch_eval
config['m_type'] = args.m_type

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
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
