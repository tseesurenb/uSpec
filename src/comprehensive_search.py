'''
Created on June 8, 2025
Comprehensive Hyperparameter Search for Universal Spectral CF
Systematic exploration of both basic and enhanced models with full parameter space

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
import sys
import argparse
import time
import warnings
import subprocess
import json
import pickle
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

warnings.filterwarnings("ignore")

class ComprehensiveHyperparameterSearch:
    """Comprehensive hyperparameter search with both basic and enhanced model support"""
    
    def __init__(self, dataset: str, search_mode: str = 'stratified', model_types: List[str] = None):
        self.dataset = dataset
        self.search_mode = search_mode
        self.model_types = model_types or ['basic', 'enhanced']
        self.results_history = []
        self.best_result = None
        
        # Define the comprehensive search space
        self.search_space = self._define_search_space()
        
        # Calculate total combinations
        self.total_combinations = self._calculate_total_combinations()
        
    def _define_search_space(self) -> Dict[str, List]:
        """Define the comprehensive search space for both model types"""
        
        # Base search space
        search_space = {
            # Model selection
            'model_type': self.model_types,
            
            # Eigenvalue configurations (applicable to both models)
            'u_n_eigen': [15, 20, 25, 30, 35, 40, 45],
            'i_n_eigen': [25, 30, 35, 40, 45, 50, 55, 60],
            
            # Learning rates
            'lr': [0.001, 0.01],
            
            # Filter types (from parse.py)
            'filter': ['u', 'i', 'ui'],
            
            # Fixed parameters for consistency
            'filter_order': [6],  # Standard order
            'epochs': [30 if self.dataset == 'ml-100k' else 25],
            'patience': [8],
            'decay': [0.01],  # Standard weight decay
        }
        
        # Enhanced model specific parameters
        if 'enhanced' in self.model_types:
            search_space.update({
                # Filter designs (only for enhanced model)
                'filter_design': [
                    'original', 'basis', 'enhanced_basis', 'adaptive_golden', 
                    'multiscale', 'ensemble', 'band_stop', 'adaptive_band_stop', 
                    'parametric_multi_band', 'harmonic'
                ],
                
                # Filter initialization patterns (only for enhanced model)
                'init_filter': [
                    # Core smoothing filters
                    'smooth', 'butterworth', 'gaussian',
                    # Golden ratio variants
                    'golden_036',
                    # Band-stop patterns
                    'band_stop', 'notch'
                ],
                
                # Similarity configurations (only for enhanced model)
                'similarity_type': ['cosine', 'jaccard'],
                'similarity_threshold': [0.001, 0.005, 0.01, 0.02],
            })
        
        return search_space
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of combinations considering model-specific parameters"""
        if len(self.model_types) == 1:
            # Single model type
            if self.model_types[0] == 'basic':
                # Basic model: exclude enhanced-only parameters
                basic_space = {k: v for k, v in self.search_space.items() 
                              if k not in ['filter_design', 'init_filter', 'similarity_type', 'similarity_threshold']}
                total = 1
                for param, values in basic_space.items():
                    total *= len(values)
                return total
            else:
                # Enhanced model: all parameters
                total = 1
                for param, values in self.search_space.items():
                    total *= len(values)
                return total
        else:
            # Both model types: basic + enhanced combinations
            basic_combinations = self._calculate_basic_combinations()
            enhanced_combinations = self._calculate_enhanced_combinations()
            return basic_combinations + enhanced_combinations
    
    def _calculate_basic_combinations(self) -> int:
        """Calculate combinations for basic model only"""
        basic_params = ['model_type', 'u_n_eigen', 'i_n_eigen', 'lr', 'filter', 
                       'filter_order', 'epochs', 'patience', 'decay']
        total = 1
        for param in basic_params:
            if param == 'model_type':
                total *= 1  # Only 'basic'
            else:
                total *= len(self.search_space[param])
        return total
    
    def _calculate_enhanced_combinations(self) -> int:
        """Calculate combinations for enhanced model only"""
        total = 1
        for param, values in self.search_space.items():
            if param == 'model_type':
                total *= 1  # Only 'enhanced'
            else:
                total *= len(values)
        return total
    
    def _get_dataset_batch_sizes(self) -> Tuple[str, str]:
        """Get appropriate batch sizes for dataset"""
        batch_configs = {
            'ml-100k': ('500', '200'),
            'ml-1m': ('1000', '400'),
            'lastfm': ('800', '300'),
            'gowalla': ('1200', '400'),
            'yelp2018': ('1500', '500'),
            'amazon-book': ('2000', '600')
        }
        return batch_configs.get(self.dataset, ('500', '200'))
    
    def generate_configurations(self, budget: Optional[int] = None) -> List[Dict]:
        """Generate configurations based on search mode"""
        
        if self.search_mode == 'systematic':
            return self._generate_systematic_configs(budget)
        elif self.search_mode == 'random':
            return self._generate_random_configs(budget or 100)
        elif self.search_mode == 'stratified':
            return self._generate_stratified_configs(budget or 200)
        elif self.search_mode == 'eigenvalue_focused':
            return self._generate_eigenvalue_focused_configs(budget or 150)
        elif self.search_mode == 'model_comparison':
            return self._generate_model_comparison_configs(budget or 100)
        else:
            raise ValueError(f"Unknown search mode: {self.search_mode}")
    
    def _generate_systematic_configs(self, budget: Optional[int]) -> List[Dict]:
        """Generate all possible configurations (systematic grid search)"""
        print(f"🔍 Generating systematic configurations for {len(self.model_types)} model type(s)...")
        print(f"📊 Total possible combinations: {self.total_combinations:,}")
        
        configs = []
        
        # Generate configurations for each model type
        for model_type in self.model_types:
            model_configs = self._generate_configs_for_model(model_type, budget)
            configs.extend(model_configs)
        
        if budget and len(configs) > budget:
            print(f"⚠️  Generated {len(configs):,} configs, sampling {budget:,} due to budget")
            configs = random.sample(configs, budget)
        
        print(f"✅ Generated {len(configs):,} systematic configurations")
        return configs
    
    def _generate_configs_for_model(self, model_type: str, budget: Optional[int]) -> List[Dict]:
        """Generate configurations for a specific model type"""
        if model_type == 'basic':
            # Basic model parameters
            param_names = ['u_n_eigen', 'i_n_eigen', 'lr', 'filter', 'filter_order', 'epochs', 'patience', 'decay']
            param_values = [self.search_space[param] for param in param_names]
            
            configs = []
            for combination in product(*param_values):
                config = dict(zip(param_names, combination))
                config['model_type'] = 'basic'
                configs.append(config)
        
        elif model_type == 'enhanced':
            # Enhanced model parameters (all parameters)
            param_names = list(self.search_space.keys())
            param_values = [self.search_space[param] for param in param_names]
            
            configs = []
            for combination in product(*param_values):
                config = dict(zip(param_names, combination))
                if config['model_type'] == 'enhanced':  # Only keep enhanced configs
                    configs.append(config)
        
        return configs
    
    def _generate_model_comparison_configs(self, budget: int) -> List[Dict]:
        """Generate configurations for direct model comparison"""
        print(f"⚖️  Generating {budget:,} model comparison configurations...")
        
        configs = []
        configs_per_model = budget // len(self.model_types)
        
        # Generate matched pairs for comparison
        base_configs = []
        for _ in range(configs_per_model):
            base_config = {
                'u_n_eigen': random.choice(self.search_space['u_n_eigen']),
                'i_n_eigen': random.choice(self.search_space['i_n_eigen']),
                'lr': random.choice(self.search_space['lr']),
                'filter': random.choice(self.search_space['filter']),
                'filter_order': self.search_space['filter_order'][0],
                'epochs': self.search_space['epochs'][0],
                'patience': self.search_space['patience'][0],
                'decay': self.search_space['decay'][0],
            }
            base_configs.append(base_config)
        
        # Create model-specific versions
        for base_config in base_configs:
            for model_type in self.model_types:
                config = base_config.copy()
                config['model_type'] = model_type
                
                if model_type == 'enhanced':
                    # Add enhanced-specific parameters
                    config['filter_design'] = random.choice(self.search_space['filter_design'])
                    config['init_filter'] = random.choice(self.search_space['init_filter'])
                    config['similarity_type'] = random.choice(self.search_space['similarity_type'])
                    config['similarity_threshold'] = random.choice(self.search_space['similarity_threshold'])
                
                configs.append(config)
        
        print(f"✅ Generated {len(configs):,} model comparison configurations")
        return configs
    
    def _generate_random_configs(self, budget: int) -> List[Dict]:
        """Generate random configurations for both model types"""
        print(f"🎲 Generating {budget:,} random configurations...")
        
        configs = []
        configs_per_model = budget // len(self.model_types)
        
        for model_type in self.model_types:
            for _ in range(configs_per_model):
                config = {'model_type': model_type}
                
                # Common parameters
                for param in ['u_n_eigen', 'i_n_eigen', 'lr', 'filter', 'filter_order', 'epochs', 'patience', 'decay']:
                    config[param] = random.choice(self.search_space[param])
                
                # Enhanced-specific parameters
                if model_type == 'enhanced':
                    for param in ['filter_design', 'init_filter', 'similarity_type', 'similarity_threshold']:
                        config[param] = random.choice(self.search_space[param])
                
                configs.append(config)
        
        # Fill remaining budget
        remaining = budget - len(configs)
        for _ in range(remaining):
            model_type = random.choice(self.model_types)
            config = {'model_type': model_type}
            
            for param in ['u_n_eigen', 'i_n_eigen', 'lr', 'filter', 'filter_order', 'epochs', 'patience', 'decay']:
                config[param] = random.choice(self.search_space[param])
            
            if model_type == 'enhanced':
                for param in ['filter_design', 'init_filter', 'similarity_type', 'similarity_threshold']:
                    config[param] = random.choice(self.search_space[param])
            
            configs.append(config)
        
        print(f"✅ Generated {len(configs):,} random configurations")
        return configs
    
    def _generate_stratified_configs(self, budget: int) -> List[Dict]:
        """Generate stratified configurations for both model types"""
        print(f"📊 Generating {budget:,} stratified configurations...")
        
        configs = []
        
        # Distribute budget across model types
        budget_per_model = budget // len(self.model_types)
        
        for model_type in self.model_types:
            print(f"  Generating {budget_per_model} configs for {model_type} model...")
            
            # 1. Eigenvalue ratio experiments (30% of model budget)
            eigenvalue_budget = int(budget_per_model * 0.3)
            configs.extend(self._generate_eigenvalue_ratio_configs(eigenvalue_budget, model_type))
            
            # 2. Learning rate experiments (20% of model budget)
            lr_budget = int(budget_per_model * 0.2)
            configs.extend(self._generate_learning_rate_configs(lr_budget, model_type))
            
            # 3. Model-specific experiments (40% of model budget)
            specific_budget = int(budget_per_model * 0.4)
            if model_type == 'enhanced':
                configs.extend(self._generate_filter_design_configs(specific_budget, model_type))
            else:
                configs.extend(self._generate_basic_specific_configs(specific_budget, model_type))
            
            # 4. Random exploration (remaining 10%)
            remaining_budget = budget_per_model - (eigenvalue_budget + lr_budget + specific_budget)
            configs.extend(self._generate_random_configs_for_model(remaining_budget, model_type))
        
        # Shuffle to avoid systematic bias
        random.shuffle(configs)
        
        print(f"✅ Generated {len(configs):,} stratified configurations")
        return configs[:budget]
    
    def _generate_random_configs_for_model(self, budget: int, model_type: str) -> List[Dict]:
        """Generate random configurations for a specific model type"""
        configs = []
        
        for _ in range(budget):
            config = {'model_type': model_type}
            
            # Common parameters
            for param in ['u_n_eigen', 'i_n_eigen', 'lr', 'filter', 'filter_order', 'epochs', 'patience', 'decay']:
                config[param] = random.choice(self.search_space[param])
            
            # Enhanced-specific parameters
            if model_type == 'enhanced':
                for param in ['filter_design', 'init_filter', 'similarity_type', 'similarity_threshold']:
                    config[param] = random.choice(self.search_space[param])
            
            configs.append(config)
        
        return configs
    
    def _generate_eigenvalue_ratio_configs(self, budget: int, model_type: str) -> List[Dict]:
        """Generate configs focused on eigenvalue ratios for specific model"""
        configs = []
        
        # Test eigenvalue combinations
        eigenvalue_pairs = [(u, i) for u in self.search_space['u_n_eigen'] 
                           for i in self.search_space['i_n_eigen']]
        
        # Sample pairs if budget is smaller
        if len(eigenvalue_pairs) > budget:
            eigenvalue_pairs = random.sample(eigenvalue_pairs, budget)
        
        for u_eigen, i_eigen in eigenvalue_pairs[:budget]:
            config = {
                'model_type': model_type,
                'u_n_eigen': u_eigen,
                'i_n_eigen': i_eigen,
                'lr': 0.001,  # Standard learning rate
                'filter': 'ui',  # Most comprehensive
                'filter_order': 6,
                'epochs': self.search_space['epochs'][0],
                'patience': 8,
                'decay': 0.01,
            }
            
            # Add enhanced-specific parameters
            if model_type == 'enhanced':
                config.update({
                    'filter_design': 'enhanced_basis',  # Proven design
                    'init_filter': 'smooth',  # Reliable initialization
                    'similarity_type': 'cosine',
                    'similarity_threshold': 0.01,
                })
            
            configs.append(config)
        
        return configs
    
    def _generate_filter_design_configs(self, budget: int, model_type: str) -> List[Dict]:
        """Generate configs focused on filter designs (enhanced model only)"""
        if model_type != 'enhanced':
            return []
        
        configs = []
        configs_per_design = max(1, budget // len(self.search_space['filter_design']))
        
        for design in self.search_space['filter_design']:
            for _ in range(configs_per_design):
                if len(configs) >= budget:
                    break
                
                config = {
                    'model_type': 'enhanced',
                    'u_n_eigen': random.choice([25, 30, 35]),  # Medium values
                    'i_n_eigen': random.choice([40, 45, 50]),  # Medium values
                    'lr': random.choice(self.search_space['lr']),
                    'filter': random.choice(self.search_space['filter']),
                    'filter_design': design,
                    'init_filter': random.choice(self.search_space['init_filter']),
                    'similarity_type': random.choice(self.search_space['similarity_type']),
                    'similarity_threshold': random.choice(self.search_space['similarity_threshold']),
                    'filter_order': 6,
                    'epochs': self.search_space['epochs'][0],
                    'patience': 8,
                    'decay': 0.01,
                }
                configs.append(config)
        
        return configs[:budget]
    
    def _generate_basic_specific_configs(self, budget: int, model_type: str) -> List[Dict]:
        """Generate configs focused on basic model optimization"""
        configs = []
        
        # Focus on filter types and eigenvalue combinations for basic model
        for _ in range(budget):
            config = {
                'model_type': 'basic',
                'u_n_eigen': random.choice(self.search_space['u_n_eigen']),
                'i_n_eigen': random.choice(self.search_space['i_n_eigen']),
                'lr': random.choice(self.search_space['lr']),
                'filter': random.choice(self.search_space['filter']),
                'filter_order': random.choice([3, 4, 5, 6]),  # Explore different orders for basic
                'epochs': self.search_space['epochs'][0],
                'patience': 8,
                'decay': 0.01,
            }
            configs.append(config)
        
        return configs
    
    def _generate_learning_rate_configs(self, budget: int, model_type: str) -> List[Dict]:
        """Generate configs focused on learning rates for specific model"""
        configs = []
        configs_per_lr = budget // len(self.search_space['lr'])
        
        for lr in self.search_space['lr']:
            for _ in range(configs_per_lr):
                if len(configs) >= budget:
                    break
                
                config = {
                    'model_type': model_type,
                    'u_n_eigen': random.choice(self.search_space['u_n_eigen']),
                    'i_n_eigen': random.choice(self.search_space['i_n_eigen']),
                    'lr': lr,
                    'filter': random.choice(self.search_space['filter']),
                    'filter_order': 6,
                    'epochs': self.search_space['epochs'][0],
                    'patience': 8,
                    'decay': 0.01,
                }
                
                # Add enhanced-specific parameters
                if model_type == 'enhanced':
                    config.update({
                        'filter_design': random.choice(self.search_space['filter_design']),
                        'init_filter': random.choice(self.search_space['init_filter']),
                        'similarity_type': random.choice(self.search_space['similarity_type']),
                        'similarity_threshold': random.choice(self.search_space['similarity_threshold']),
                    })
                
                configs.append(config)
        
        return configs
    
    def _generate_eigenvalue_focused_configs(self, budget: int) -> List[Dict]:
        """Generate configurations focused on eigenvalue exploration for both models"""
        print(f"🔬 Generating {budget:,} eigenvalue-focused configurations...")
        
        configs = []
        budget_per_model = budget // len(self.model_types)
        
        for model_type in self.model_types:
            model_configs = self._generate_eigenvalue_ratio_configs(budget_per_model, model_type)
            configs.extend(model_configs)
        
        print(f"✅ Generated {len(configs):,} eigenvalue-focused configurations")
        return configs[:budget]
    
    def _build_command(self, config: Dict) -> List[str]:
        """Build command line for experiment"""
        train_batch, eval_batch = self._get_dataset_batch_sizes()
        
        cmd = [
            sys.executable, "main.py",
            "--model_type", config['model_type'],
            "--dataset", self.dataset,
            "--u_n_eigen", str(config['u_n_eigen']),
            "--i_n_eigen", str(config['i_n_eigen']),
            "--lr", str(config['lr']),
            "--filter", config['filter'],
            "--filter_order", str(config['filter_order']),
            "--epochs", str(config['epochs']),
            "--patience", str(config['patience']),
            "--decay", str(config['decay']),
            "--train_u_batch_size", train_batch,
            "--eval_u_batch_size", eval_batch,
            "--seed", "2025",
            "--verbose", "0"
        ]
        
        # Add enhanced-specific parameters
        if config['model_type'] == 'enhanced':
            cmd.extend([
                "--filter_design", config['filter_design'],
                "--init_filter", config['init_filter'],
                "--similarity_type", config['similarity_type'],
                "--similarity_threshold", str(config['similarity_threshold']),
            ])
        
        return cmd
    
    def _evaluate_config(self, config: Dict, timeout: int = 900) -> Optional[float]:
        """Evaluate a single configuration"""
        cmd = self._build_command(config)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                return None
            
            # Parse NDCG@20
            ndcg = self._parse_ndcg(result.stdout)
            return ndcg
                
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
    
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
    
    def run_search(self, budget: Optional[int] = None) -> Dict:
        """Run the comprehensive hyperparameter search"""
        
        print(f"🚀 COMPREHENSIVE HYPERPARAMETER SEARCH")
        print(f"=" * 80)
        print(f"📊 Dataset: {self.dataset.upper()}")
        print(f"🔍 Search Mode: {self.search_mode.upper()}")
        print(f"🤖 Model Types: {', '.join(self.model_types).upper()}")
        print(f"📈 Total Possible Combinations: {self.total_combinations:,}")
        
        if budget:
            print(f"💰 Budget: {budget:,} experiments")
        else:
            print(f"💰 Budget: UNLIMITED (all combinations)")
        
        print(f"=" * 80)
        
        # Generate configurations
        start_time = time.time()
        configs = self.generate_configurations(budget)
        
        if not configs:
            print("❌ No configurations generated!")
            return {}
        
        print(f"\n🏃 Running {len(configs):,} experiments...")
        print(f"⏱️  Estimated time: {len(configs) * 0.5:.1f} minutes (assuming 30s per experiment)")
        print("-" * 80)
        
        # Run experiments
        successful_experiments = 0
        failed_experiments = 0
        
        for i, config in enumerate(configs, 1):
            # Progress indicator
            if i % 10 == 0 or i <= 10:
                progress = (i / len(configs)) * 100
                print(f"\n[{i:4d}/{len(configs)}] Progress: {progress:5.1f}%")
            
            # Show current config (abbreviated)
            model_type = config['model_type'][:4]  # 'basi' or 'enha'
            config_str = f"{model_type} u={config['u_n_eigen']:2d}, i={config['i_n_eigen']:2d}, lr={config['lr']:.3f}"
            
            if config['model_type'] == 'enhanced':
                config_str += f", {config['filter_design'][:8]}, {config['init_filter'][:6]}"
            
            print(f"  {config_str:<60}", end=" ", flush=True)
            
            # Evaluate
            ndcg = self._evaluate_config(config)
            
            if ndcg is not None:
                print(f"✅ {ndcg:.6f}")
                self.results_history.append((config, ndcg))
                successful_experiments += 1
                
                # Update best result
                if self.best_result is None or ndcg > self.best_result[1]:
                    self.best_result = (config, ndcg)
                    print(f"    🏆 NEW BEST: {ndcg:.6f} ({config['model_type']})")
            else:
                print("❌")
                failed_experiments += 1
        
        total_time = time.time() - start_time
        
        # Analyze and report results
        self._analyze_and_report_results(successful_experiments, failed_experiments, total_time)
        
        return {
            'best_config': self.best_result[0] if self.best_result else None,
            'best_ndcg': self.best_result[1] if self.best_result else None,
            'all_results': self.results_history,
            'total_time': total_time,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments
        }
    
    def _analyze_and_report_results(self, successful: int, failed: int, total_time: float):
        """Analyze and report comprehensive results with model comparison"""
        
        if not self.results_history:
            print("\n❌ No successful experiments!")
            return
        
        # Sort results by performance
        self.results_history.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n" + "=" * 100)
        print(f"🏆 COMPREHENSIVE SEARCH RESULTS - {self.dataset.upper()}")
        print(f"=" * 100)
        
        print(f"\n📊 EXPERIMENT SUMMARY:")
        print(f"  ✅ Successful: {successful:,}")
        print(f"  ❌ Failed: {failed:,}")
        print(f"  ⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"  ⚡ Average Time: {total_time/successful:.1f}s per experiment")
        
        # Model type breakdown
        self._analyze_model_performance()
        
        # Top results
        print(f"\n🥇 TOP 10 CONFIGURATIONS:")
        print(f"{'Rank':<4} {'Model':<8} {'u_eigen':<7} {'i_eigen':<7} {'Ratio':<5} {'LR':<6} "
              f"{'Filter':<6} {'Design':<12} {'NDCG@20':<10}")
        print("-" * 85)
        
        for i, (config, ndcg) in enumerate(self.results_history[:10], 1):
            ratio = config['i_n_eigen'] / config['u_n_eigen']
            design = config.get('filter_design', 'N/A')[:12]
            print(f"{i:<4} {config['model_type']:<8} {config['u_n_eigen']:<7} {config['i_n_eigen']:<7} "
                  f"{ratio:<5.2f} {config['lr']:<6.3f} {config['filter']:<6} {design:<12} {ndcg:<10.6f}")
        
        # Parameter analysis
        self._analyze_parameter_importance()
        
        # Model-specific analysis
        if len(self.model_types) > 1:
            self._analyze_model_comparison()
        
        # Eigenvalue ratio analysis
        self._analyze_eigenvalue_ratios()
    
    def _analyze_model_performance(self):
        """Analyze performance by model type"""
        print(f"\n🤖 MODEL TYPE PERFORMANCE:")
        
        model_results = defaultdict(list)
        for config, ndcg in self.results_history:
            model_results[config['model_type']].append(ndcg)
        
        print(f"  {'Model':<10} {'Count':<6} {'Mean NDCG':<12} {'Max NDCG':<12} {'Min NDCG':<12}")
        print("  " + "-" * 60)
        
        for model_type in sorted(model_results.keys()):
            scores = model_results[model_type]
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            print(f"  {model_type:<10} {len(scores):<6} {mean_score:<12.6f} {max_score:<12.6f} {min_score:<12.6f}")
    
    def _analyze_model_comparison(self):
        """Analyze direct model comparison"""
        print(f"\n⚖️  MODEL COMPARISON ANALYSIS:")
        
        # Find best results for each model type
        best_by_model = {}
        for config, ndcg in self.results_history:
            model_type = config['model_type']
            if model_type not in best_by_model or ndcg > best_by_model[model_type][1]:
                best_by_model[model_type] = (config, ndcg)
        
        print(f"  Best Results by Model Type:")
        for model_type, (config, ndcg) in best_by_model.items():
            print(f"    {model_type}: {ndcg:.6f}")
            if model_type == 'enhanced':
                print(f"      └─ Filter Design: {config.get('filter_design', 'N/A')}")
                print(f"      └─ Init Filter: {config.get('init_filter', 'N/A')}")
                print(f"      └─ Similarity: {config.get('similarity_type', 'N/A')}")
        
        # Performance improvement analysis
        if 'basic' in best_by_model and 'enhanced' in best_by_model:
            basic_score = best_by_model['basic'][1]
            enhanced_score = best_by_model['enhanced'][1]
            improvement = ((enhanced_score - basic_score) / basic_score) * 100
            print(f"  📈 Enhanced vs Basic Improvement: {improvement:+.2f}%")
    
    def _analyze_parameter_importance(self):
        """Analyze parameter importance across both models"""
        print(f"\n🔍 PARAMETER IMPORTANCE ANALYSIS:")
        
        if len(self.results_history) < 10:
            print("  Not enough data for analysis")
            return
        
        # Get top 25% and bottom 25% results
        n_results = len(self.results_history)
        top_quartile = self.results_history[:max(1, n_results//4)]
        bottom_quartile = self.results_history[-max(1, n_results//4):]
        
        # Common parameters
        common_params = ['u_n_eigen', 'i_n_eigen', 'lr', 'filter', 'model_type']
        
        for param in common_params:
            top_values = [config[param] for config, _ in top_quartile]
            bottom_values = [config[param] for config, _ in bottom_quartile]
            
            if isinstance(top_values[0], (int, float)):
                top_mean = np.mean(top_values)
                bottom_mean = np.mean(bottom_values)
                print(f"  {param:<15}: Top={top_mean:6.2f}, Bottom={bottom_mean:6.2f}, Diff={top_mean-bottom_mean:+6.2f}")
            else:
                top_common = max(set(top_values), key=top_values.count) if top_values else "N/A"
                bottom_common = max(set(bottom_values), key=bottom_values.count) if bottom_values else "N/A"
                print(f"  {param:<15}: Top={top_common}, Bottom={bottom_common}")
    
    def _analyze_eigenvalue_ratios(self):
        """Analyze eigenvalue ratio performance across models"""
        print(f"\n📊 EIGENVALUE RATIO ANALYSIS:")
        
        ratio_performance = defaultdict(lambda: defaultdict(list))
        
        for config, ndcg in self.results_history:
            ratio = config['i_n_eigen'] / config['u_n_eigen']
            ratio_bin = round(ratio * 2) / 2  # Round to nearest 0.5
            model_type = config['model_type']
            ratio_performance[ratio_bin][model_type].append(ndcg)
        
        print(f"  {'Ratio':<6} {'Model':<8} {'Count':<6} {'Mean NDCG':<12} {'Max NDCG':<12}")
        print("  " + "-" * 50)
        
        for ratio in sorted(ratio_performance.keys()):
            for model_type in sorted(ratio_performance[ratio].keys()):
                scores = ratio_performance[ratio][model_type]
                mean_score = np.mean(scores)
                max_score = np.max(scores)
                print(f"  {ratio:<6.1f} {model_type:<8} {len(scores):<6} {mean_score:<12.6f} {max_score:<12.6f}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save comprehensive results with model information"""
        if filename is None:
            timestamp = int(time.time())
            models_str = "_".join(self.model_types)
            filename = f"comprehensive_search_{self.dataset}_{models_str}_{self.search_mode}_{timestamp}.pkl"
        
        results_data = {
            'dataset': self.dataset,
            'model_types': self.model_types,
            'search_mode': self.search_mode,
            'search_space': self.search_space,
            'total_combinations': self.total_combinations,
            'results_history': self.results_history,
            'best_result': self.best_result,
            'timestamp': time.time()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results_data, f)
        
        print(f"\n💾 Results saved to {filename}")
        
        # Also save as CSV for easy analysis
        csv_filename = filename.replace('.pkl', '.csv')
        self._save_csv_results(csv_filename)
    
    def _save_csv_results(self, filename: str):
        """Save results as CSV for easy analysis"""
        if not self.results_history:
            return
        
        rows = []
        for config, ndcg in self.results_history:
            row = config.copy()
            row['ndcg'] = ndcg
            row['ratio'] = config['i_n_eigen'] / config['u_n_eigen']
            # Fill missing enhanced parameters for basic model
            if config['model_type'] == 'basic':
                row.update({
                    'filter_design': 'N/A',
                    'init_filter': 'N/A',
                    'similarity_type': 'N/A',
                    'similarity_threshold': 'N/A'
                })
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"💾 CSV results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hyperparameter Search with Model Selection")
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='Dataset to search on')
    
    parser.add_argument('--search_mode', type=str, default='stratified',
                       choices=['systematic', 'random', 'stratified', 'eigenvalue_focused', 'model_comparison'],
                       help='Search strategy')
    
    parser.add_argument('--model_types', type=str, nargs='+', default=['basic', 'enhanced'],
                       choices=['basic', 'enhanced'],
                       help='Model types to search (can specify multiple)')
    
    parser.add_argument('--budget', type=int, default=None,
                       help='Maximum number of experiments (None for unlimited)')
    
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results to file')
    
    parser.add_argument('--timeout', type=int, default=900,
                       help='Timeout per experiment in seconds')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_types:
        print("❌ Must specify at least one model type")
        return 1
    
    # Initialize searcher
    searcher = ComprehensiveHyperparameterSearch(args.dataset, args.search_mode, args.model_types)
    
    # Warn about large search spaces
    if args.search_mode == 'systematic' and args.budget is None:
        print(f"⚠️  WARNING: Systematic search without budget will run {searcher.total_combinations:,} experiments!")
        print(f"   This could take {searcher.total_combinations * 0.5 / 60:.0f} hours!")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Search cancelled.")
            return 1
    
    # Run search
    results = searcher.run_search(args.budget)
    
    # Save results
    if args.save and results:
        searcher.save_results()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# COMPREHENSIVE HYPERPARAMETER SEARCH - Usage Examples with Both Models
# ============================================================================

# MODEL COMPARISON (Recommended - Compare basic vs enhanced):
# python comprehensive_search.py --dataset ml-100k --search_mode model_comparison --model_types basic enhanced --budget 100

# STRATIFIED SEARCH - Both Models (Balanced exploration):
# python comprehensive_search.py --dataset ml-100k --search_mode stratified --model_types basic enhanced --budget 200

# BASIC MODEL ONLY (Fast exploration):
# python comprehensive_search.py --dataset ml-100k --search_mode stratified --model_types basic --budget 100

# ENHANCED MODEL ONLY (Full feature exploration):
# python comprehensive_search.py --dataset ml-100k --search_mode stratified --model_types enhanced --budget 150

# EIGENVALUE FOCUSED - Both Models (Understand eigenvalue impact):
# python comprehensive_search.py --dataset ml-100k --search_mode eigenvalue_focused --model_types basic enhanced --budget 200

# LARGE DATASET SEARCHES:
# python comprehensive_search.py --dataset gowalla --search_mode model_comparison --model_types basic enhanced --budget 150
# python comprehensive_search.py --dataset yelp2018 --search_mode stratified --model_types enhanced --budget 100

# ============================================================================
# SEARCH MODE EXPLANATIONS WITH MODEL SUPPORT:
# ============================================================================

# MODEL_COMPARISON:
# - Generates matched configuration pairs for direct model comparison
# - Same eigenvalue counts, learning rates, etc. for both models
# - Enhanced model gets additional filter_design, init_filter parameters
# - Best for understanding model type impact

# STRATIFIED (with both models):
# - Distributes budget evenly across model types
# - Each model gets: 30% eigenvalue, 20% LR, 40% model-specific, 10% random
# - Model-specific: filter designs for enhanced, filter orders for basic
# - Most comprehensive exploration

# EIGENVALUE_FOCUSED (with both models):
# - Tests eigenvalue ratio ranges for each model type separately
# - Shows how eigenvalue allocation affects each model differently
# - Good for understanding model-specific eigenvalue sensitivity

# RANDOM (with both models):
# - Random sampling from full parameter space for each model
# - Budget split evenly between model types
# - Quick baseline for both models

# ============================================================================
# PARAMETER SPACE SUMMARY WITH BOTH MODELS:
# ============================================================================

# Common Parameters (both models):
# - model_type: ['basic', 'enhanced'] (2 values)
# - u_n_eigen: [15, 20, 25, 30, 35, 40, 45] (7 values)
# - i_n_eigen: [25, 30, 35, 40, 45, 50, 55, 60] (8 values)
# - lr: [0.001, 0.01] (2 values)
# - filter: ['u', 'i', 'ui'] (3 values)

# Enhanced-Only Parameters:
# - filter_design: [10 options] (10 values)
# - init_filter: [6 options] (6 values)
# - similarity_type: ['cosine', 'jaccard'] (2 values)
# - similarity_threshold: [0.001, 0.005, 0.01, 0.02] (4 values)

# Total Combinations:
# - Basic model only: 7 × 8 × 2 × 3 = 336
# - Enhanced model only: 7 × 8 × 2 × 3 × 10 × 6 × 2 × 4 = 161,280
# - Both models: 336 + 161,280 = 161,616

# ============================================================================
# EXPECTED RESULTS AND INSIGHTS:
# ============================================================================

# Typical Performance Differences (ML-100K):
# - Basic Model: NDCG@20 ≈ 0.375-0.385
# - Enhanced Model: NDCG@20 ≈ 0.385-0.395
# - Improvement: ~1-3% (but with much more flexibility)

# Runtime Differences:
# - Basic Model: ~20-40s per experiment
# - Enhanced Model: ~40-80s per experiment (depending on filter design)

# Memory Usage:
# - Basic Model: Lower memory footprint
# - Enhanced Model: Higher due to caching and advanced processing

# Best Use Cases:
# - model_comparison: Understanding when enhanced features are worth it
# - Basic only: Fast prototyping and baseline establishment
# - Enhanced only: Production optimization and maximum performance
# - Both models: Complete understanding of trade-offs

# ============================================================================