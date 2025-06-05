'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced argument parser with filter design options

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Spectral model for CF")

    # Basic training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-2, help="the weight decay for l2 normalizaton")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='batch size for training users, -1 for full dataset')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="batch size for evaluation users (memory management)")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset and evaluation
    parser.add_argument('--dataset', type=str, default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of training data to use for validation (0.1 = 10%)')
    
    # Model architecture
    parser.add_argument('--n_eigen', type=int, default=500)
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [uspec]')
    parser.add_argument('--m_type', type=str, default='single', help='single or double similarity')
    parser.add_argument('--filter', type=str, default='u', help='u, i, or ui')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order for spectral filters')
    
    # NEW: Filter design options
    parser.add_argument('--filter_design', type=str, default='basis', 
                       choices=['original', 'basis', 'adaptive', 'neural'],
                       help='Filter design: original (polynomial), basis (combination), adaptive (eigenvalue-dependent), neural (MLP)')
    parser.add_argument('--init_filter', type=str, default='smooth',
                       help='Initial filter pattern from filters.py (e.g., smooth, golden_036, butterworth)')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='0 for silent, 1 for verbose')
    parser.add_argument('--run_convergence_test', action='store_true', 
                       help='Run comprehensive convergence test across designs and initializations')
    
    return parser.parse_args()

# Add validation for filter design combinations
def validate_args(args):
    """Validate argument combinations"""
    
    # Check if init_filter exists (you might want to validate against filters.py)
    valid_init_filters = [
        'smooth', 'golden_036', 'butterworth', 'gaussian', 'bessel', 'conservative',
        'chebyshev', 'aggressive', 'elliptic_lp', 'hamming_lp'
    ]
    
    if args.init_filter not in valid_init_filters:
        print(f"Warning: init_filter '{args.init_filter}' may not exist in filters.py")
        print(f"Valid options include: {valid_init_filters[:5]}...")
    
    # Adjust epochs for convergence test
    if args.run_convergence_test and args.epochs < 30:
        print(f"Note: Convergence test recommended with epochs >= 30, setting to 30")
        args.epochs = 30
    
    # Recommend settings for different filter designs
    if args.filter_design == 'basis':
        print(f"📋 Using BASIS filter design - recommended for best convergence")
    elif args.filter_design == 'original':
        print(f"🔧 Using ORIGINAL filter design - may need higher epochs for convergence")
    elif args.filter_design == 'adaptive':
        print(f"🎯 Using ADAPTIVE filter design - experimental eigenvalue-dependent filtering")
    elif args.filter_design == 'neural':
        print(f"🧠 Using NEURAL filter design - experimental neural network filtering")
    
    return args

# if __name__ == "__main__":
#     args = parse_args()
#     args = validate_args(args)
#     print("Parsed arguments:", args)