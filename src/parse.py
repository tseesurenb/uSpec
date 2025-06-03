'''
Created on June 3, 2025
Pytorch Implementation of uSpec in
Batsuuri. Tse et al. uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go uinversal Spectral model for CF")

    parser.add_argument('--lr', type=float,default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument('--testbatch', type=int,default=100, help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--topks', nargs='?',default="[20]", help="@k test list")
    parser.add_argument('--i_K', type=int,default=400)
    parser.add_argument('--u_K', type=int,default=400)
    parser.add_argument('--sim_K', type=int,default=40)
    parser.add_argument('--s_temp', type=float,default=1.0)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--n_eigen', type=int,default=100)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [cf, uspec]')
    parser.add_argument('--filter', type=str, default='u', help='u, i, and ui')
    parser.add_argument('--verbose', type=int, default=0, help='0 for silent, 1 for verbose')
    
    return parser.parse_args()
