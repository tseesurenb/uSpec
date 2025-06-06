#!/usr/bin/env python3
'''
Advanced Hyperparameter Search for High-Capacity Filter Designs
Focuses on neural, deep, multiscale, and ensemble filters with sophisticated optimization

Usage:
python hyperparam_search.py [--dataset DATASET] [--filter_design DESIGN] [--strategy STRATEGY] [--budget BUDGET]
'''

import os
import sys
import argparse
import time
import warnings
import subprocess
import json
import pickle
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np
import random
from itertools import product
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

class HyperparameterSearcher:
    """Advanced hyperparameter search for high-capacity filter designs"""
    
    def __init__(self, dataset: str, filter_design: str, strategy: str = 'bayesian'):
        self.dataset = dataset
        self.filter_design = filter_design
        self.strategy = strategy
        self.results_history = []
        self.best_result = None
        
        # Define search spaces for high-capacity filters
        self.search_spaces = self._get_search_spaces()
        
    def _get_search_spaces(self) -> Dict[str, Dict]:
        """Define hyperparameter search spaces for each filter design"""
        
        base_space = {
            'lr': [0.0005, 0.001, 0.002, 0.005, 0.01],
            'decay': [0.001, 0.005, 0.01, 0.02, 0.05],
            'filter_order': [4, 5, 6, 7, 8],
            'patience': [5, 8, 10, 12],
        }
        
        dataset_eigen_ranges = {
            'ml-100k': [50, 75, 100, 125, 150],
            'ml-1m': [100, 150, 200, 250, 300],
            'lastfm': [80, 120, 150, 200, 250],
            'gowalla': [150, 200, 250, 300, 400],
            'yelp2018': [200, 300, 400, 500, 600],
            'amazon-book': [300, 400, 500, 600, 800]
        }
        
        dataset_epoch_ranges = {
            'ml-100k': [30, 40, 50, 60],
            'ml-1m': [25, 35, 45, 55],
            'lastfm': [25, 35, 45, 55],
            'gowalla': [20, 30, 40, 50],
            'yelp2018': [15, 25, 35, 45],
            'amazon-book': [15, 20, 30, 40]
        }
        
        dataset_epoch_ranges = {
            'ml-100k': [30, 40, 50, 60],
            'ml-1m': [25, 35, 45, 55],
            'lastfm': [25, 35, 45, 55],
            'yelp2018': [15, 25, 35, 45],
            'amazon-book': [15, 20, 30, 40]
        }
        
        base_space['n_eigen'] = dataset_eigen_ranges.get(self.dataset, dataset_eigen_ranges['ml-100k'])
        base_space['epochs'] = dataset_epoch_ranges.get(self.dataset, dataset_epoch_ranges['ml-100k'])
        
        # Filter-specific extensions
        filter_specific = {
            'neural': {
                **base_space,
                'hidden_dims': ['[64, 32]', '[128, 64]', '[256, 128]', '[128, 64, 32]', '[256, 128, 64]'],
                'dropout_rate': [0.0, 0.1, 0.2, 0.3],
                'activation': ['relu', 'tanh', 'elu'],
                'batch_norm': [True, False],
                'residual_connections': [True, False]
            },
            'deep': {
                **base_space,
                'hidden_dims': ['[128, 64, 32]', '[256, 128, 64]', '[512, 256, 128]', 
                               '[256, 128, 64, 32]', '[512, 256, 128, 64]'],
                'dropout_rate': [0.1, 0.2, 0.3, 0.4],
                'activation': ['relu', 'elu', 'leaky_relu'],
                'batch_norm': [True],
                'residual_connections': [True, False],
                'depth_scaling': [1.0, 1.2, 1.5],
                'layer_decay': [0.9, 0.95, 1.0]
            },
            'multiscale': {
                **base_space,
                'scales': ['[1, 2, 4]', '[1, 2, 3, 6]', '[1, 2, 4, 8]', '[1, 3, 6, 12]'],
                'scale_weights': ['uniform', 'learned', 'attention'],
                'fusion_method': ['concat', 'add', 'attention'],
                'per_scale_dim': [32, 64, 128],
                'attention_heads': [2, 4, 8]
            },
            'ensemble': {
                **base_space,
                'n_models': [3, 5, 7],
                'diversity_loss': [0.0, 0.01, 0.05, 0.1],
                'ensemble_method': ['average', 'learned_weights', 'attention'],
                'model_types': ['mixed', 'homogeneous'],
                'bootstrap_ratio': [0.8, 0.9, 1.0],
                'feature_sampling': [0.8, 0.9, 1.0]
            },
            'enhanced_basis': {
                **base_space,
                'basis_expansion': [2, 3, 4, 5],
                'regularization_strength': [0.001, 0.01, 0.1],
                'orthogonal_constraint': [True, False],
                'adaptive_weights': [True, False]
            }
        }
        
        return filter_specific.get(self.filter_design, base_space)
    
    def _get_high_performance_inits(self) -> List[str]:
        """Get initialization patterns known to work well with high-capacity filters"""
        return [
            'soft_golden_ratio', 'golden_036', 'golden_optimized_1', 'golden_optimized_2',
            'smooth', 'butterworth', 'soft_tuned_351', 'oscillatory_soft_v2',
            'natural_harmony', 'multi_band_balanced'
        ]
    
    def random_search(self, budget: int) -> List[Dict]:
        """Random hyperparameter search"""
        print(f"🎲 Starting random search with budget {budget}")
        
        configs = []
        search_space = self.search_spaces
        
        for i in range(budget):
            config = {}
            for param, values in search_space.items():
                if isinstance(values, list):
                    config[param] = random.choice(values)
                else:
                    config[param] = values
            
            # Add initialization pattern
            config['init_filter'] = random.choice(self._get_high_performance_inits())
            configs.append(config)
        
        return configs
    
    def grid_search(self, budget: int) -> List[Dict]:
        """Grid search (limited by budget)"""
        print(f"🔍 Starting grid search with budget {budget}")
        
        search_space = self.search_spaces
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[param] if isinstance(search_space[param], list) 
                       else [search_space[param]] for param in param_names]
        
        all_combinations = list(product(*param_values))
        
        # Sample if too many combinations
        if len(all_combinations) > budget:
            all_combinations = random.sample(all_combinations, budget)
        
        configs = []
        init_patterns = self._get_high_performance_inits()
        
        for combo in all_combinations[:budget]:
            config = dict(zip(param_names, combo))
            config['init_filter'] = random.choice(init_patterns)
            configs.append(config)
        
        return configs
    
    def bayesian_search(self, budget: int) -> List[Dict]:
        """Simplified Bayesian optimization using past results"""
        print(f"🧠 Starting Bayesian search with budget {budget}")
        
        configs = []
        
        # Start with some random exploration
        exploration_budget = min(budget // 3, 10)
        configs.extend(self.random_search(exploration_budget))
        
        # Run initial configs to get some data
        for config in configs:
            result = self._evaluate_config(config)
            if result is not None:
                self.results_history.append((config, result))
        
        # Exploitation phase based on promising areas
        remaining_budget = budget - len(configs)
        
        if self.results_history:
            # Find best configurations
            self.results_history.sort(key=lambda x: x[1], reverse=True)
            top_configs = [x[0] for x in self.results_history[:3]]
            
            # Generate variations around best configs
            for _ in range(remaining_budget):
                if top_configs:
                    base_config = random.choice(top_configs).copy()
                    # Add noise to create variations
                    config = self._mutate_config(base_config)
                    configs.append(config)
        
        return configs[exploration_budget:]  # Return only the exploitation configs
    
    def _mutate_config(self, base_config: Dict) -> Dict:
        """Create a mutated version of a configuration"""
        config = base_config.copy()
        search_space = self.search_spaces
        
        # Mutate 1-3 parameters
        n_mutations = random.randint(1, min(3, len(search_space)))
        params_to_mutate = random.sample(list(search_space.keys()), n_mutations)
        
        for param in params_to_mutate:
            if param in search_space and isinstance(search_space[param], list):
                config[param] = random.choice(search_space[param])
        
        # Occasionally change initialization
        if random.random() < 0.3:
            config['init_filter'] = random.choice(self._get_high_performance_inits())
        
        return config
    
    def _evaluate_config(self, config: Dict) -> Optional[float]:
        """Evaluate a single configuration"""
        cmd = self._build_command(config)
        
        try:
            print(f"    Testing config...", end=" ", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            
            if result.returncode != 0:
                print("❌")
                return None
            
            # Parse NDCG
            ndcg = self._parse_ndcg(result.stdout)
            if ndcg is not None:
                print(f"✅ {ndcg:.6f}")
                return ndcg
            else:
                print("❌")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏰")
            return None
        except Exception:
            print("💥")
            return None
    
    def _build_command(self, config: Dict) -> List[str]:
        """Build command line for experiment"""
        cmd = [
            sys.executable, "main.py",
            "--dataset", self.dataset,
            "--filter_design", self.filter_design,
            "--init_filter", config.get('init_filter', 'golden_036'),
            "--filter", "ui",
            "--seed", "2025",
            "--verbose", "0"
        ]
        
        # Add standard parameters that exist in your parse.py
        standard_params = ['lr', 'decay', 'filter_order', 'n_eigen', 'epochs', 'patience']
        for param in standard_params:
            if param in config:
                cmd.extend([f"--{param}", str(config[param])])
        
        # Add dataset-specific batch sizes
        dataset_batch_sizes = {
            'ml-100k': ('500', '200'),
            'ml-1m': ('1000', '400'),
            'lastfm': ('800', '300'),
            'gowalla': ('1200', '400'),
            'yelp2018': ('1500', '500'),
            'amazon-book': ('2000', '600')
        }
        
        train_batch, eval_batch = dataset_batch_sizes.get(self.dataset, ('500', '200'))
        cmd.extend(['--train_u_batch_size', train_batch, '--eval_u_batch_size', eval_batch])
        
        return cmd
    
    def _parse_ndcg(self, output: str) -> Optional[float]:
        """Parse NDCG@20 from output"""
        for line in output.split('\n'):
            if "Final Test Results:" in line and "NDCG@20=" in line:
                try:
                    ndcg_part = line.split("NDCG@20=")[1]
                    return float(ndcg_part.split(",")[0])
                except (IndexError, ValueError):
                    continue
        return None
    
    def search(self, budget: int) -> Dict:
        """Main search function"""
        print(f"🚀 Hyperparameter Search: {self.filter_design.upper()} on {self.dataset.upper()}")
        print(f"📊 Strategy: {self.strategy.upper()}, Budget: {budget}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Generate configurations based on strategy
        if self.strategy == 'random':
            configs = self.random_search(budget)
        elif self.strategy == 'grid':
            configs = self.grid_search(budget)
        elif self.strategy == 'bayesian':
            configs = self.bayesian_search(budget)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # For non-Bayesian methods, evaluate all configs
        if self.strategy != 'bayesian':
            for i, config in enumerate(configs, 1):
                print(f"\n[{i}/{len(configs)}] Config {i}")
                result = self._evaluate_config(config)
                if result is not None:
                    self.results_history.append((config, result))
        
        # Analyze results
        if not self.results_history:
            print("❌ No successful experiments!")
            return {}
        
        self.results_history.sort(key=lambda x: x[1], reverse=True)
        self.best_result = self.results_history[0]
        
        total_time = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print(f"🏆 HYPERPARAMETER SEARCH RESULTS")
        print("=" * 80)
        
        print(f"\n🥇 BEST CONFIGURATION (NDCG@20: {self.best_result[1]:.6f}):")
        best_config = self.best_result[0]
        for key, value in sorted(best_config.items()):
            print(f"  {key:<20}: {value}")
        
        print(f"\n📊 TOP 5 CONFIGURATIONS:")
        for i, (config, ndcg) in enumerate(self.results_history[:5], 1):
            print(f"{i}. NDCG@20: {ndcg:.6f}")
            key_params = ['lr', 'decay', 'n_eigen', 'epochs', 'init_filter']
            for param in key_params:
                if param in config:
                    print(f"   {param}: {config[param]}")
            print()
        
        # Parameter importance analysis
        self._analyze_parameter_importance()
        
        print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"✅ Completed {len(self.results_history)} successful experiments")
        
        return {
            'best_config': self.best_result[0],
            'best_ndcg': self.best_result[1],
            'all_results': self.results_history,
            'total_time': total_time
        }
    
    def _analyze_parameter_importance(self):
        """Analyze which parameters have the most impact"""
        print(f"\n🔍 PARAMETER IMPORTANCE ANALYSIS:")
        
        if len(self.results_history) < 5:
            print("  Not enough data for analysis")
            return
        
        # Get top 25% and bottom 25% results
        n_results = len(self.results_history)
        top_25_percent = self.results_history[:max(1, n_results//4)]
        bottom_25_percent = self.results_history[-max(1, n_results//4):]
        
        # Analyze each parameter
        all_params = set()
        for config, _ in self.results_history:
            all_params.update(config.keys())
        
        param_impact = {}
        
        for param in all_params:
            top_values = [config.get(param) for config, _ in top_25_percent if param in config]
            bottom_values = [config.get(param) for config, _ in bottom_25_percent if param in config]
            
            if top_values and bottom_values:
                # For categorical parameters, find most common values
                if isinstance(top_values[0], str):
                    top_common = max(set(top_values), key=top_values.count) if top_values else None
                    bottom_common = max(set(bottom_values), key=bottom_values.count) if bottom_values else None
                    if top_common != bottom_common:
                        param_impact[param] = f"Top: {top_common}, Bottom: {bottom_common}"
                else:
                    # For numerical parameters, compare means
                    top_mean = np.mean([x for x in top_values if x is not None])
                    bottom_mean = np.mean([x for x in bottom_values if x is not None])
                    if abs(top_mean - bottom_mean) > 0.1 * top_mean:
                        param_impact[param] = f"Top: {top_mean:.3f}, Bottom: {bottom_mean:.3f}"
        
        for param, impact in sorted(param_impact.items()):
            print(f"  {param:<20}: {impact}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save search results to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"hyperparam_results_{self.filter_design}_{self.dataset}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'filter_design': self.filter_design,
                'dataset': self.dataset,
                'strategy': self.strategy,
                'results_history': self.results_history,
                'best_result': self.best_result
            }, f)
        
        print(f"💾 Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Hyperparameter Search")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='Dataset to test on')
    parser.add_argument('--filter_design', type=str, default='neural',
                       choices=['neural', 'deep', 'multiscale', 'ensemble', 'enhanced_basis'],
                       help='High-capacity filter design to optimize')
    parser.add_argument('--strategy', type=str, default='bayesian',
                       choices=['random', 'grid', 'bayesian'],
                       help='Search strategy')
    parser.add_argument('--budget', type=int, default=50,
                       help='Number of configurations to try')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Validate high-capacity filter
    high_capacity_filters = ['neural', 'deep', 'multiscale', 'ensemble', 'enhanced_basis']
    if args.filter_design not in high_capacity_filters:
        print(f"❌ {args.filter_design} is not a high-capacity filter.")
        print(f"Available high-capacity filters: {', '.join(high_capacity_filters)}")
        return 1
    
    searcher = HyperparameterSearcher(args.dataset, args.filter_design, args.strategy)
    results = searcher.search(args.budget)
    
    if args.save and results:
        searcher.save_results()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

# ============================================================================
# HIGH-CAPACITY FILTER HYPERPARAMETER SEARCH - Usage Guide
# ============================================================================

# Quick Start Examples:
# python hyperparam_search.py --filter_design neural --dataset ml-100k --budget 30
# python hyperparam_search.py --filter_design deep --dataset ml-1m --strategy bayesian --budget 50
# GOWALLA: Location-based social network dataset
# python hyperparam_search.py --filter_design enhanced_basis --dataset gowalla --budget 25  
# python hyperparam_search.py --filter_design neural --dataset gowalla --budget 35
# python hyperparam_search.py --filter_design deep --dataset gowalla --budget 45

# YELP2018: Large, sparse commercial dataset
# python hyperparam_search.py --filter_design enhanced_basis --dataset yelp2018 --budget 30
# python hyperparam_search.py --filter_design deep --dataset yelp2018 --budget 50
# python hyperparam_search.py --filter_design ensemble --dataset yelp2018 --budget 80

# AMAZON-BOOK: Very large, extremely sparse e-commerce dataset  
# python hyperparam_search.py --filter_design deep --dataset amazon-book --budget 40
# python hyperparam_search.py --filter_design ensemble --dataset amazon-book --budget 100
# python hyperparam_search.py --filter_design multiscale --dataset amazon-book --budget 60

# ============================================================================
# High-Capacity Filter Definitions:
# ============================================================================

# NEURAL (~561 parameters):
# - Multi-layer perceptron with configurable architecture
# - Key hyperparameters: hidden_dims, dropout_rate, activation, batch_norm
# - Good for: Medium-sized datasets, needs careful regularization

# DEEP (~1000+ parameters):
# - Deep neural network with residual connections
# - Key hyperparameters: depth, layer_decay, residual_connections
# - Good for: Large datasets, complex patterns

# MULTISCALE (~500+ parameters):
# - Multi-resolution filter processing
# - Key hyperparameters: scales, fusion_method, attention_heads
# - Good for: Capturing different temporal patterns

# ENSEMBLE (~2000+ parameters):
# - Multiple models combined with learned weights
# - Key hyperparameters: n_models, diversity_loss, ensemble_method
# - Good for: Maximum performance, large datasets only

# ENHANCED_BASIS (~52 parameters):
# - Enhanced traditional approach (included as baseline)
# - Key hyperparameters: basis_expansion, regularization_strength
# - Good for: Balanced performance/complexity

# ============================================================================
# Search Strategy Recommendations:
# ============================================================================

# BAYESIAN (Recommended for expensive experiments):
# - Intelligently explores promising regions
# - Best for: Limited budget (20-50 experiments)
# - Balances exploration vs exploitation

# RANDOM (Good baseline):
# - Simple random sampling
# - Best for: Initial exploration, understanding parameter sensitivity
# - Easy to parallelize

# GRID (Comprehensive but expensive):
# - Systematic exploration
# - Best for: Small parameter spaces, thorough analysis
# - Can be very expensive

# ============================================================================
# Budget Recommendations by Filter Type:
# ============================================================================

# NEURAL: 30-50 experiments
# - Focus on: hidden_dims, dropout_rate, lr, decay
# - Critical: regularization (dropout, batch_norm)

# DEEP: 50-100 experiments  
# - Focus on: architecture depth, residual connections, layer_decay
# - Critical: avoiding overfitting

# MULTISCALE: 40-80 experiments
# - Focus on: scale combinations, fusion_method, attention_heads
# - Critical: scale selection and combination

# ENSEMBLE: 100+ experiments
# - Focus on: n_models, diversity_loss, ensemble_method
# - Critical: model diversity and combination weights

# ============================================================================
# Expected Hyperparameter Insights:
# ============================================================================

# Learning Rate (lr):
# - Neural/Deep: 0.001-0.002 often optimal
# - Ensemble: Lower rates (0.0005-0.001) for stability
# - Multiscale: Medium rates (0.001-0.005) for convergence

# Regularization:
# - Neural: dropout_rate 0.1-0.2, batch_norm=True
# - Deep: dropout_rate 0.2-0.3, residual_connections=True
# - Ensemble: diversity_loss 0.01-0.05

# Architecture:
# - Neural: [128, 64] or [256, 128] often sufficient
# - Deep: 3-4 layers with gradual dimension reduction
# - Multiscale: scales=[1,2,4] good baseline, attention fusion

# Dataset-Specific:
# - ml-100k: Smaller architectures, more regularization
# - ml-1m: Medium architectures, balanced regularization  
# - lastfm: Larger architectures, less regularization

# ============================================================================
# Interpreting Results:
# ============================================================================

# Success Indicators:
# - NDCG@20 > 0.390 (ml-100k): Excellent performance
# - NDCG@20 > 0.350 (ml-1m): Good performance for large dataset
# - Consistent results across runs: Stable hyperparameters

# Red Flags:
# - High variance in results: Unstable hyperparameters
# - Poor performance despite high capacity: Overfitting
# - Training timeouts: Architecture too complex

# Parameter Importance Analysis:
# - Parameters with large top/bottom differences are critical
# - Focus future searches on these parameters
# - Stable parameters can be fixed in subsequent searches

# ============================================================================
# Advanced Usage Patterns:
# ============================================================================

# Multi-Stage Search:
# 1. Random search (budget=30) for broad exploration
# 2. Bayesian search (budget=50) around promising regions  
# 3. Grid search (budget=20) for fine-tuning

# Cross-Dataset Validation:
# 1. Optimize on ml-100k (fast feedback)
# 2. Validate on ml-1m (larger scale)
# 3. Test on lastfm (different domain)

# Architecture Evolution:
# 1. Start with neural (baseline high-capacity)
# 2. Move to deep (if dataset is large enough)
# 3. Try multiscale (for temporal patterns)
# 4. Use ensemble (only for final maximum performance)

# ============================================================================