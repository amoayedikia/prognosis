#!/usr/bin/env python3
"""
PROGRESS Ablation Study - Using Actual PROGRESS Implementation

This script runs ablation study using the actual PROGRESS.py models
(TrajectoryParameterNetwork and DeepSurvivalNetwork) to properly
evaluate feature group contributions.

Usage:
    python progress_ablation_actual.py --data-dir ./dataset --output-dir ./ablation_output
"""

import sys
import os

# Add the directory containing PROGRESS.py to the path
sys.path.insert(0, '/mnt/user-data/uploads')
sys.path.insert(0, '/mnt/project')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Import from actual PROGRESS.py
from PROGRESS import (
    PROGRESSConfig,
    PROGRESSDataset,
    TrajectoryParameterNetwork,
    DeepSurvivalNetwork,
    TrajectoryLoss,
    CoxPartialLikelihoodLoss,
    PROGRESSMetrics,
    setup_logging
)

# =============================================================================
# FEATURE GROUP DEFINITIONS
# =============================================================================

@dataclass
class FeatureGroups:
    """Definition of feature groups for ablation study."""
    
    GROUPS = {
        'CSF': ['ABETA_harm', 'PTAU_harm', 'TTAU_harm'],
        'Ratios': ['PTAU_ABETA_ratio', 'TTAU_PTAU_ratio'],
        'Demographics': ['AGE_AT_BASELINE', 'SEX', 'EDUC'],
        'Cognition': ['baseline_MMSE', 'baseline_CDRSUM']
    }
    
    ALL_FEATURES = [
        'ABETA_harm', 'PTAU_harm', 'TTAU_harm',
        'PTAU_ABETA_ratio', 'TTAU_PTAU_ratio',
        'AGE_AT_BASELINE', 'SEX', 'EDUC',
        'baseline_MMSE', 'baseline_CDRSUM'
    ]
    
    GROUP_LABELS = {
        'CSF': 'CSF Biomarkers',
        'Ratios': 'Derived Ratios',
        'Demographics': 'Demographics',
        'Cognition': 'Baseline Cognition'
    }
    
    @classmethod
    def get_ablation_indices(cls, remove_group: str, feature_names: List[str]) -> List[int]:
        """Get indices of features to KEEP after removing a group."""
        features_to_remove = cls.GROUPS[remove_group]
        return [i for i, f in enumerate(feature_names) if f not in features_to_remove]


# =============================================================================
# ABLATED DATASET WRAPPER
# =============================================================================

class AblatedDataset(Dataset):
    """
    Wrapper around PROGRESSDataset that masks out specific feature groups.
    """
    
    def __init__(self, 
                 base_dataset: PROGRESSDataset,
                 keep_indices: List[int] = None):
        """
        Args:
            base_dataset: Original PROGRESSDataset
            keep_indices: Indices of features to keep (None = keep all)
        """
        self.base_dataset = base_dataset
        self.keep_indices = keep_indices
        
        # Get original data
        self.X_full = base_dataset.X.clone()
        self.Y_traj = base_dataset.Y_traj.clone()
        self.T = base_dataset.T.clone()
        self.E = base_dataset.E.clone()
        self.subjects = base_dataset.subjects
        
        # Apply feature mask
        if keep_indices is not None:
            self.X = self.X_full[:, keep_indices]
            self.feature_names = [base_dataset.feature_names[i] for i in keep_indices]
        else:
            self.X = self.X_full
            self.feature_names = base_dataset.feature_names
        
        self.input_dim = self.X.shape[1]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'trajectory_params': self.Y_traj[idx],
            'time': self.T[idx],
            'event': self.E[idx],
            'subject_id': self.subjects[idx]
        }


# =============================================================================
# PROGRESS TRAINER FOR ABLATION
# =============================================================================

class PROGRESSAblationTrainer:
    """
    Trainer that uses actual PROGRESS models for ablation study.
    """
    
    def __init__(self, config: PROGRESSConfig = None):
        self.config = config or PROGRESSConfig()
        self.device = self.config.get_device()
        
        self.traj_model = None
        self.surv_model = None
        
        self.traj_loss_fn = TrajectoryLoss(self.config.calibration_weight)
        self.surv_loss_fn = CoxPartialLikelihoodLoss(self.config.ranking_weight)
    
    def setup_models(self, input_dim: int):
        """Initialize PROGRESS models with correct input dimension."""
        self.traj_model = TrajectoryParameterNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.traj_hidden_dims,
            dropout=self.config.traj_dropout,
            num_attention_heads=self.config.traj_attention_heads,
            use_batch_norm=self.config.traj_use_batch_norm
        ).to(self.device)
        
        self.surv_model = DeepSurvivalNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.surv_hidden_dims,
            dropout=self.config.surv_dropout,
            use_batch_norm=self.config.surv_use_batch_norm
        ).to(self.device)
    
    def train_survival_model(self,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            num_epochs: int = 50) -> Dict:
        """Train survival model and return best validation loss."""
        
        optimizer = torch.optim.AdamW(
            self.surv_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(num_epochs):
            # Training
            self.surv_model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)
                
                optimizer.zero_grad()
                risk_scores = self.surv_model(features)
                loss_dict = self.surv_loss_fn(risk_scores, times, events)
                loss = loss_dict['total']
                
                if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.surv_model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.surv_model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    times = batch['time'].to(self.device)
                    events = batch['event'].to(self.device)
                    
                    risk_scores = self.surv_model(features)
                    loss_dict = self.surv_loss_fn(risk_scores, times, events)
                    
                    if not torch.isnan(loss_dict['total']):
                        val_losses.append(loss_dict['total'].item())
            
            val_loss = np.mean(val_losses) if val_losses else float('inf')
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.surv_model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                break
        
        # Restore best model
        if best_state is not None:
            self.surv_model.load_state_dict(best_state)
        
        return {'best_val_loss': best_val_loss}
    
    def evaluate_survival(self, 
                         test_loader: DataLoader,
                         horizons: List[float] = [2.0, 3.0, 5.0]) -> Dict:
        """Evaluate survival model on test data."""
        
        self.surv_model.eval()
        
        all_features = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in test_loader:
                all_features.append(batch['features'])
                all_times.append(batch['time'])
                all_events.append(batch['event'])
        
        features = torch.cat(all_features).to(self.device)
        times = torch.cat(all_times).numpy()
        events = torch.cat(all_events).numpy()
        
        # Get risk scores
        with torch.no_grad():
            risk_scores = self.surv_model(features).cpu().numpy().squeeze()
        
        results = {}
        
        # C-index
        results['c_index'] = PROGRESSMetrics.concordance_index(risk_scores, times, events)
        
        # Time-dependent AUC at each horizon
        for h in horizons:
            results[f'auc_{h}yr'] = PROGRESSMetrics.time_dependent_auc(
                risk_scores, times, events, h
            )
        
        return results


# =============================================================================
# MAIN ABLATION STUDY
# =============================================================================

class PROGRESSAblationStudy:
    """
    Ablation study using actual PROGRESS implementation.
    """
    
    def __init__(self,
                 data_dir: str,
                 output_dir: str = './ablation_output',
                 n_folds: int = 5,
                 num_epochs: int = 50,
                 random_state: int = 42):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.random_state = random_state
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'ablation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
    
    def load_data(self) -> PROGRESSDataset:
        """Load NACC integrated dataset."""
        
        integrated_path = os.path.join(self.data_dir, 'nacc_integrated_dataset.pkl')
        
        if not os.path.exists(integrated_path):
            raise FileNotFoundError(f"Dataset not found: {integrated_path}")
        
        self.logger.info(f"Loading data from: {integrated_path}")
        integrated_data = pd.read_pickle(integrated_path)
        self.logger.info(f"Loaded {len(integrated_data)} subjects")
        
        # Create PROGRESS dataset
        config = PROGRESSConfig()
        dataset = PROGRESSDataset(
            integrated_data=integrated_data,
            fit_scaler=True,
            config=config
        )
        
        self.logger.info(f"Dataset created: {len(dataset)} valid subjects")
        self.logger.info(f"Features: {dataset.feature_names}")
        
        return dataset
    
    def run_ablation(self, 
                     base_dataset: PROGRESSDataset,
                     horizons: List[float] = [2.0, 3.0, 5.0]) -> Dict:
        """
        Run complete ablation study.
        """
        
        self.logger.info("=" * 70)
        self.logger.info("PROGRESS ABLATION STUDY")
        self.logger.info("Using actual PROGRESS implementation")
        self.logger.info("=" * 70)
        
        feature_names = base_dataset.feature_names
        conditions = ['Full'] + list(FeatureGroups.GROUPS.keys())
        
        # Initialize results
        results = {
            'c_index': {cond: [] for cond in conditions},
            'auc': {h: {cond: [] for cond in conditions} for h in horizons}
        }
        
        # Get events for stratification
        events = base_dataset.E.numpy()
        n_samples = len(base_dataset)
        indices = np.arange(n_samples)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                            random_state=self.random_state)
        
        for fold, (train_val_idx, test_idx) in enumerate(cv.split(indices, events)):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"FOLD {fold + 1}/{self.n_folds}")
            self.logger.info(f"{'='*70}")
            
            # Split train_val into train and validation
            train_events = events[train_val_idx]
            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            train_idx, val_idx = next(cv_inner.split(train_val_idx, train_events))
            train_idx = train_val_idx[train_idx]
            val_idx = train_val_idx[val_idx]
            
            self.logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
            
            # Evaluate each ablation condition
            for condition in conditions:
                self.logger.info(f"\n--- Condition: {condition} ---")
                
                # Get feature indices for this condition
                if condition == 'Full':
                    keep_indices = None
                else:
                    keep_indices = FeatureGroups.get_ablation_indices(condition, feature_names)
                
                # Create ablated dataset
                ablated_dataset = AblatedDataset(base_dataset, keep_indices)
                input_dim = ablated_dataset.input_dim
                
                self.logger.info(f"  Features: {input_dim} (removed {condition if condition != 'Full' else 'none'})")
                
                # Create data loaders
                train_loader = DataLoader(
                    Subset(ablated_dataset, train_idx),
                    batch_size=32,
                    shuffle=True,
                    drop_last=True
                )
                val_loader = DataLoader(
                    Subset(ablated_dataset, val_idx),
                    batch_size=32,
                    shuffle=False
                )
                test_loader = DataLoader(
                    Subset(ablated_dataset, test_idx),
                    batch_size=32,
                    shuffle=False
                )
                
                # Train and evaluate PROGRESS survival model
                try:
                    config = PROGRESSConfig()
                    trainer = PROGRESSAblationTrainer(config)
                    trainer.setup_models(input_dim)
                    
                    # Train
                    train_result = trainer.train_survival_model(
                        train_loader, val_loader, 
                        num_epochs=self.num_epochs
                    )
                    
                    # Evaluate
                    eval_result = trainer.evaluate_survival(test_loader, horizons)
                    
                    # Store results
                    c_index = eval_result['c_index']
                    results['c_index'][condition].append(c_index)
                    
                    for h in horizons:
                        auc = eval_result[f'auc_{h}yr']
                        results['auc'][h][condition].append(auc)
                    
                    self.logger.info(f"  C-index: {c_index:.4f}")
                    for h in horizons:
                        self.logger.info(f"  AUC@{h}yr: {eval_result[f'auc_{h}yr']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"  Error: {e}")
                    results['c_index'][condition].append(np.nan)
                    for h in horizons:
                        results['auc'][h][condition].append(np.nan)
        
        # Aggregate results
        self.results = self._aggregate_results(results, horizons)
        
        return self.results
    
    def _aggregate_results(self, raw_results: Dict, horizons: List[float]) -> Dict:
        """Aggregate fold results."""
        
        conditions = ['Full'] + list(FeatureGroups.GROUPS.keys())
        
        aggregated = {
            'c_index': {},
            'auc': {h: {} for h in horizons},
            'contribution': {}
        }
        
        # Aggregate C-index
        for condition in conditions:
            scores = raw_results['c_index'][condition]
            valid_scores = [s for s in scores if not np.isnan(s)]
            
            aggregated['c_index'][condition] = {
                'mean': np.mean(valid_scores) if valid_scores else np.nan,
                'std': np.std(valid_scores) if valid_scores else np.nan,
                'scores': valid_scores
            }
        
        # Aggregate AUC
        for h in horizons:
            for condition in conditions:
                scores = raw_results['auc'][h][condition]
                valid_scores = [s for s in scores if not np.isnan(s)]
                
                aggregated['auc'][h][condition] = {
                    'mean': np.mean(valid_scores) if valid_scores else np.nan,
                    'std': np.std(valid_scores) if valid_scores else np.nan,
                    'scores': valid_scores
                }
        
        # Compute contributions
        full_mean = aggregated['c_index']['Full']['mean']
        full_scores = aggregated['c_index']['Full']['scores']
        
        for group in FeatureGroups.GROUPS.keys():
            ablated_mean = aggregated['c_index'][group]['mean']
            ablated_scores = aggregated['c_index'][group]['scores']
            
            if not np.isnan(full_mean) and not np.isnan(ablated_mean):
                abs_drop = full_mean - ablated_mean
                rel_drop = (abs_drop / full_mean) * 100 if full_mean > 0 else 0
                
                # Statistical test
                if len(full_scores) == len(ablated_scores) and len(full_scores) > 1:
                    t_stat, p_value = stats.ttest_rel(full_scores, ablated_scores)
                else:
                    t_stat, p_value = np.nan, np.nan
                
                aggregated['contribution'][group] = {
                    'absolute_drop': abs_drop,
                    'relative_drop': rel_drop,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False
                }
            else:
                aggregated['contribution'][group] = {
                    'absolute_drop': np.nan,
                    'relative_drop': np.nan,
                    't_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False
                }
        
        return aggregated
    
    def generate_summary(self) -> str:
        """Generate text summary of results."""
        
        lines = []
        lines.append("=" * 70)
        lines.append("PROGRESS ABLATION STUDY SUMMARY")
        lines.append("(Using actual PROGRESS implementation)")
        lines.append("=" * 70)
        
        # C-index results
        lines.append("\nC-index Results:")
        lines.append("-" * 50)
        for condition in ['Full'] + list(FeatureGroups.GROUPS.keys()):
            data = self.results['c_index'][condition]
            lines.append(f"  {condition:15s}: {data['mean']:.4f} ± {data['std']:.4f}")
        
        # Contributions
        lines.append("\nFeature Group Contributions:")
        lines.append("-" * 50)
        
        # Find most important
        contributions = self.results['contribution']
        sorted_groups = sorted(contributions.keys(), 
                              key=lambda g: contributions[g]['relative_drop'],
                              reverse=True)
        
        for group in sorted_groups:
            contrib = contributions[group]
            sig = "*" if contrib['significant'] else ""
            lines.append(f"  {FeatureGroups.GROUP_LABELS[group]:20s}: "
                        f"{contrib['relative_drop']:+.1f}% (p={contrib['p_value']:.4f}){sig}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table."""
        
        conditions = ['Full'] + list(FeatureGroups.GROUPS.keys())
        
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{PROGRESS Ablation Study: C-index performance when each feature group is removed.}")
        lines.append(r"\label{tab:progress_ablation}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Condition & C-index & $\Delta$ (\%) & p-value & Sig. \\")
        lines.append(r"\midrule")
        
        # Full model
        full_data = self.results['c_index']['Full']
        lines.append(f"Full & {full_data['mean']:.3f}$\\pm${full_data['std']:.3f} & - & - & - \\\\")
        lines.append(r"\midrule")
        
        # Ablated conditions
        for group in FeatureGroups.GROUPS.keys():
            data = self.results['c_index'][group]
            contrib = self.results['contribution'][group]
            
            sig = r"\checkmark" if contrib['significant'] else ""
            p_str = f"{contrib['p_value']:.3f}" if not np.isnan(contrib['p_value']) else "-"
            
            lines.append(f"$-${FeatureGroups.GROUP_LABELS[group]} & "
                        f"{data['mean']:.3f}$\\pm${data['std']:.3f} & "
                        f"{contrib['relative_drop']:+.1f}\\% & "
                        f"{p_str} & {sig} \\\\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        return "\n".join(lines)
    
    def save_results(self):
        """Save all results to files."""
        
        # Convert for JSON
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(x) for x in obj]
            return obj
        
        # Save JSON
        json_path = os.path.join(self.output_dir, 'progress_ablation_results.json')
        with open(json_path, 'w') as f:
            json.dump(to_serializable(self.results), f, indent=2)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'progress_ablation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self.generate_summary())
        
        # Save LaTeX
        latex_path = os.path.join(self.output_dir, 'progress_ablation_table.tex')
        with open(latex_path, 'w') as f:
            f.write(self.generate_latex_table())
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PROGRESS Ablation Study')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing nacc_integrated_dataset.pkl')
    parser.add_argument('--output-dir', type=str, default='./ablation_output',
                       help='Output directory')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per fold')
    
    args = parser.parse_args()
    
    # Run ablation study
    study = PROGRESSAblationStudy(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        num_epochs=args.epochs
    )
    
    # Load data
    dataset = study.load_data()
    
    # Run ablation
    results = study.run_ablation(dataset)
    
    # Print summary
    print(study.generate_summary())
    
    # Save results
    study.save_results()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
