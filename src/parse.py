'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced argument parser with DySimGCF-style parameters for model_double

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
    parser.add_argument('--epochs', type=int, default=200)
    
    # Dataset and evaluation
    parser.add_argument('--dataset', type=str, default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of training data to use for validation (0.1 = 10%)')
    
    # Model architecture
    parser.add_argument('--n_eigen', type=int, default=30)
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [uspec]')
    parser.add_argument('--m_type', type=str, default='single', help='single or double similarity')
    parser.add_argument('--filter', type=str, default='u', help='u, i, or ui')
    parser.add_argument('--filter_order', type=int, default=5, help='polynomial order for spectral filters')
    
    # Filter design options
    parser.add_argument('--filter_design', type=str, default='enhanced_basis', 
                       choices=['original', 'basis', 'enhanced_basis', 'adaptive_golden', 'adaptive', 'neural', 'deep', 'multiscale', 'ensemble'],
                       help='Filter design: original (polynomial), basis (combination), enhanced_basis (performance-optimized), adaptive_golden (golden ratio variants), adaptive (eigenvalue-dependent), neural (MLP), deep (deep neural network), multiscale (multi-scale frequency bands), ensemble (ensemble of filter types)')
    parser.add_argument('--init_filter', type=str, default='butterworth',
                       help='Initial filter pattern from filters.py (e.g., smooth, golden_036, butterworth)')
    
    # DySimGCF-style parameters (for model_double)
    parser.add_argument('--u_sim', type=str, default='cos', help='User similarity: cos (cosine) or jac (jaccard)')
    parser.add_argument('--i_sim', type=str, default='cos', help='Item similarity: cos (cosine) or jac (jaccard)')
    parser.add_argument('--u_K', type=int, default=50, help='Top-K users for user similarity (DySimGCF default)')
    parser.add_argument('--i_K', type=int, default=20, help='Top-K items for item similarity (DySimGCF default)')
    parser.add_argument('--self_loop', action='store_true', help='Include self-loops in similarity matrices')
    
    # Training control
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
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
    """Validate argument combinations with DySimGCF parameters"""
    
    # Check if init_filter exists
    valid_init_filters = [
        # Core smoothing filters
        'smooth', 'butterworth', 'gaussian', 'bessel', 'conservative',
        # Golden ratio variants
        'golden_034', 'golden_036', 'golden_348', 'golden_352',
        'soft_golden_ratio', 'golden_ratio_balanced', 
        'golden_optimized_1', 'golden_optimized_2', 'golden_optimized_3',
        # Oscillatory patterns
        'oscillatory_soft', 'oscillatory_soft_v2', 'oscillatory_soft_v3',
        # Fine-tuned coefficients
        'soft_tuned_351', 'soft_tuned_352', 'soft_tuned_353',
        # Mathematical patterns
        'fibonacci_soft', 'euler_soft', 'natural_harmony',
        # Multi-band and adaptive
        'multi_band', 'multi_band_balanced', 'wiener_like', 'adaptive_smooth',
        # Baselines
        'identity', 'exponential_decay'
    ]
    
    if args.init_filter not in valid_init_filters:
        print(f"Warning: init_filter '{args.init_filter}' may not exist in filters.py")
        print(f"Valid options include: {valid_init_filters[:8]}...")
    
    # Validate similarity types
    if args.u_sim not in ['cos', 'jac']:
        print(f"Warning: u_sim '{args.u_sim}' should be 'cos' or 'jac'")
    
    if args.i_sim not in ['cos', 'jac']:
        print(f"Warning: i_sim '{args.i_sim}' should be 'cos' or 'jac'")
    
    # DySimGCF-style parameter recommendations
    if args.m_type == 'double':
        print(f"🔄 DOUBLE MODE (DySimGCF-Style):")
        print(f"   User similarity: {args.u_sim} (K={args.u_K})")
        print(f"   Item similarity: {args.i_sim} (K={args.i_K})")
        print(f"   Self-loops: {args.self_loop}")
        
        # Dataset-specific recommendations for DySimGCF parameters
        if args.dataset == 'ml-100k':
            if args.u_K != 80 or args.i_K != 10:
                print(f"💡 TIP: For {args.dataset}, DySimGCF uses u_K=80, i_K=10")
        elif args.dataset == 'yelp2018':
            if args.u_K != 50 or args.i_K != 20:
                print(f"💡 TIP: For {args.dataset}, DySimGCF uses u_K=50, i_K=20")
        elif args.dataset == 'amazon_book':
            if args.u_K != 40 or args.i_K != 5:
                print(f"💡 TIP: For {args.dataset}, DySimGCF uses u_K=40, i_K=5")
    else:
        print(f"🔄 SINGLE MODE: Standard similarity approach")
    
    # Adjust epochs for convergence test
    if args.run_convergence_test and args.epochs < 30:
        print(f"Note: Convergence test recommended with epochs >= 30, setting to 30")
        args.epochs = 30
    
    # Filter design recommendations
    design_info = {
        'original': "🔧 ORIGINAL - Direct polynomial learning, may need higher epochs",
        'basis': "📋 BASIS - Recommended for best convergence and performance balance",
        'enhanced_basis': "⭐ ENHANCED BASIS - Optimized for maximum performance",
        'adaptive_golden': "🏆 ADAPTIVE GOLDEN - Golden ratio optimization",
        'adaptive': "🎯 ADAPTIVE - Eigenvalue-dependent experimental filtering",
        'neural': "🧠 NEURAL - Small neural network filtering (~561 params)",
        'deep': "🚀 DEEP - Deep neural network filtering (~1000+ params)",
        'multiscale': "🌈 MULTISCALE - Multi-scale frequency bands (~500+ params)",
        'ensemble': "🎭 ENSEMBLE - Ensemble of filter types (~2000+ params)"
    }
    
    if args.filter_design in design_info:
        print(design_info[args.filter_design])
    
    # Parameter capacity warnings and recommendations
    high_capacity_designs = ['deep', 'multiscale', 'ensemble']
    if args.filter_design in high_capacity_designs:
        print(f"⚠️  HIGH-CAPACITY FILTER: {args.filter_design}")
        print(f"   - Training may take longer")
        print(f"   - Consider reducing epochs if needed")
        print(f"   - Recommended for larger datasets")
        
        # Adjust batch sizes for high-capacity filters
        if args.train_u_batch_size > 1000:
            print(f"   - Reducing train batch size for memory efficiency")
            args.train_u_batch_size = min(args.train_u_batch_size, 800)
        
        if args.eval_u_batch_size > 500:
            print(f"   - Reducing eval batch size for memory efficiency")
            args.eval_u_batch_size = min(args.eval_u_batch_size, 300)
    
    # Dataset-specific recommendations
    if args.dataset in ['amazon-book', 'yelp2018'] and args.filter_design in ['original', 'adaptive']:
        print(f"💡 TIP: For {args.dataset}, consider 'enhanced_basis' or 'ensemble' for better performance")
    
    elif args.dataset == 'ml-100k' and args.filter_design == 'ensemble':
        print(f"💡 TIP: For small dataset {args.dataset}, 'basis' or 'enhanced_basis' may be more suitable")
    
    # Learning rate recommendations for high-capacity filters
    if args.filter_design in high_capacity_designs and args.lr > 0.001:
        print(f"💡 TIP: Consider lower learning rate (0.0005-0.001) for high-capacity filters")
    
    return args

def get_filter_design_info():
    """Get information about available filter designs"""
    return {
        'original': {
            'params': '~14',
            'description': 'Direct polynomial coefficient learning',
            'best_for': 'Quick baselines, understanding basic behavior'
        },
        'basis': {
            'params': '~32', 
            'description': 'Learnable combination of proven filter patterns',
            'best_for': 'Balanced performance and efficiency'
        },
        'enhanced_basis': {
            'params': '~52',
            'description': 'Enhanced basis with performance optimizations', 
            'best_for': 'High performance with moderate complexity'
        },
        'adaptive_golden': {
            'params': '~28',
            'description': 'Adaptive variations of golden ratio patterns',
            'best_for': 'Golden ratio optimization and fine-tuning'
        },
        'adaptive': {
            'params': '~15',
            'description': 'Eigenvalue-dependent frequency band adaptation',
            'best_for': 'Experimental adaptive filtering'
        },
        'neural': {
            'params': '~561',
            'description': 'Small neural network for spectral response',
            'best_for': 'Neural approach with moderate parameters'
        },
        'deep': {
            'params': '~1000+',
            'description': 'Deep neural network with multiple hidden layers',
            'best_for': 'Maximum neural expressiveness'
        },
        'multiscale': {
            'params': '~500+', 
            'description': 'Multi-scale frequency bands with modulation network',
            'best_for': 'Complex frequency-dependent patterns'
        },
        'ensemble': {
            'params': '~2000+',
            'description': 'Ensemble of classical, deep, and multiscale filters',
            'best_for': 'Maximum performance on large datasets'
        }
    }

def print_filter_design_help():
    """Print detailed help about filter designs"""
    print("\n" + "="*80)
    print("🎛️  FILTER DESIGN OPTIONS")
    print("="*80)
    
    designs = get_filter_design_info()
    
    for design, info in designs.items():
        print(f"\n{design.upper():15} ({info['params']:8} params)")
        print(f"  Description: {info['description']}")
        print(f"  Best for:    {info['best_for']}")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS:")
    print("  🚀 Start with: 'basis' or 'enhanced_basis'")
    print("  🎯 For efficiency: 'original' or 'adaptive_golden'")  
    print("  🏆 For max performance: 'ensemble' (large datasets)")
    print("  🧪 For experimentation: 'deep' or 'multiscale'")
    print("="*80)
    
    print("\n" + "="*80)
    print("🔄 MODEL TYPES:")
    print("  SINGLE: Standard similarity (like LightGCN)")
    print("  DOUBLE: DySimGCF-style user-user + item-item similarities")
    print("="*80)