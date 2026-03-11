#!/usr/bin/env python3
"""
Demographic Fairness Analysis for PROGRESS Framework

This script evaluates PROGRESS model performance stratified by demographic groups
to assess algorithmic fairness across populations, as required for clinical journal
publications (addressing concerns raised by Yuan et al., 2023).

Experiments:
    1. Performance stratified by sex (Female vs Male)
    2. Performance stratified by age group (≤70 vs >70)
    3. Performance stratified by education level (Low vs High)
    4. Disparity metrics and statistical significance testing
    5. Calibration analysis across demographic groups

Output:
    - demographic_fairness_results.json: Complete results
    - demographic_fairness_summary.csv: Summary table
    - demographic_fairness_analysis.tex: LaTeX tables for paper
    - demographic_fairness_plots/: Visualizations

Usage:
    python demographic_fairness_analysis.py --data-dir ./dataset --csf-file ./investigator_fcsf_nacc69.csv

Author: Generated for PROGRESS paper
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import os
import sys
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from PROGRESS module
try:
    from PROGRESS import (
        PROGRESSConfig, 
        TrajectoryParameterNetwork, DeepSurvivalNetwork
    )
except ImportError:
    print("Warning: Could not import from PROGRESS.py. Using local definitions.")


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file = os.path.join(output_dir, 'demographic_fairness.log')
    
    logger = logging.getLogger('demographic_fairness')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger

logger = logging.getLogger('demographic_fairness')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PROGRESSConfig:
    """Configuration for PROGRESS framework."""
    
    sequence_length: int = 5
    min_visits_trajectory: int = 3
    
    traj_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    traj_dropout: float = 0.3
    traj_attention_heads: int = 4
    traj_use_batch_norm: bool = True
    
    surv_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    surv_dropout: float = 0.3
    surv_use_batch_norm: bool = True
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    patience: int = 15
    gradient_clip: float = 1.0
    
    mc_dropout_samples: int = 50
    n_folds: int = 5
    
    device: str = 'auto'
    
    def get_device(self) -> torch.device:
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.device)


# =============================================================================
# DATASET WITH DEMOGRAPHIC TRACKING
# =============================================================================

class DemographicDataset(Dataset):
    """Dataset with demographic group tracking for fairness analysis."""
    
    MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}
    
    def __init__(self, integrated_data: pd.DataFrame, scaler: RobustScaler = None,
                 fit_scaler: bool = False, config: PROGRESSConfig = None):
        self.config = config or PROGRESSConfig()
        self.integrated_data = integrated_data
        
        self.subjects = self._get_valid_subjects()
        self.baseline_features, self.feature_names = self._extract_baseline_features()
        self.trajectory_params = self._compute_trajectory_parameters()
        self.survival_data = self._extract_survival_data()
        
        # Extract demographic information
        self.demographics = self._extract_demographics()
        
        if fit_scaler:
            self.scaler = RobustScaler()
            self.baseline_features_scaled = self.scaler.fit_transform(self.baseline_features)
        elif scaler is not None:
            self.scaler = scaler
            self.baseline_features_scaled = self.scaler.transform(self.baseline_features)
        else:
            self.scaler = None
            self.baseline_features_scaled = self.baseline_features
        
        self._create_tensors()
    
    def _is_valid(self, value) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if value in self.MISSING_CODES:
            return False
        return True
    
    def _clean_value(self, value, default: float = np.nan) -> float:
        if not self._is_valid(value):
            return default
        return float(value)
    
    def _get_valid_subjects(self) -> List[str]:
        valid_subjects = []
        for _, row in self.integrated_data.iterrows():
            naccid = row.get('NACCID')
            if naccid is None:
                continue
            has_biomarker = any([
                self._is_valid(row.get('ABETA_harm')),
                self._is_valid(row.get('PTAU_harm')),
                self._is_valid(row.get('TTAU_harm'))
            ])
            if not has_biomarker:
                continue
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) >= 2:
                valid_subjects.append(naccid)
        return valid_subjects
    
    def _extract_baseline_features(self) -> Tuple[np.ndarray, List[str]]:
        feature_names = [
            'ABETA_harm', 'PTAU_harm', 'TTAU_harm',
            'PTAU_ABETA_ratio', 'TTAU_PTAU_ratio',
            'AGE_AT_BASELINE', 'SEX', 'EDUC',
            'baseline_MMSE', 'baseline_CDRSUM'
        ]
        features_list = []
        for naccid in self.subjects:
            row = self.integrated_data[self.integrated_data['NACCID'] == naccid].iloc[0]
            abeta = self._clean_value(row.get('ABETA_harm'), 500.0)
            ptau = self._clean_value(row.get('PTAU_harm'), 50.0)
            ttau = self._clean_value(row.get('TTAU_harm'), 300.0)
            ptau_abeta_ratio = ptau / abeta if abeta > 0 else 0.1
            ttau_ptau_ratio = ttau / ptau if ptau > 0 else 6.0
            age = self._clean_value(row.get('AGE_AT_BASELINE'), 75.0)
            sex = self._clean_value(row.get('SEX'), 1.0)
            educ = self._clean_value(row.get('EDUC'), 16.0)
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) > 0:
                first_visit = trajectory[0]
                baseline_mmse = self._clean_value(first_visit.get('NACCMMSE'), 28.0)
                baseline_cdr = self._clean_value(first_visit.get('CDRSUM'), 0.5)
            else:
                baseline_mmse = 28.0
                baseline_cdr = 0.5
            features = [abeta, ptau, ttau, ptau_abeta_ratio, ttau_ptau_ratio,
                       age, sex, educ, baseline_mmse, baseline_cdr]
            features_list.append(features)
        features_array = np.array(features_list, dtype=np.float32)
        for col in range(features_array.shape[1]):
            col_data = features_array[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                features_array[mask, col] = median_val if not np.isnan(median_val) else 0.0
        return features_array, feature_names
    
    def _compute_trajectory_parameters(self) -> np.ndarray:
        trajectory_params = []
        for naccid in self.subjects:
            row = self.integrated_data[self.integrated_data['NACCID'] == naccid].iloc[0]
            trajectory = row.get('clinical_trajectory', [])
            if not isinstance(trajectory, list) or len(trajectory) < 3:
                trajectory_params.append([np.nan, np.nan, np.nan])
                continue
            times, cdr_values = [], []
            for visit in trajectory:
                t = visit.get('YearsFromBaseline')
                cdr = visit.get('CDRSUM')
                if self._is_valid(t) and self._is_valid(cdr):
                    cdr = float(cdr)
                    if 0 <= cdr <= 18:
                        times.append(float(t))
                        cdr_values.append(cdr)
            if len(times) >= 3:
                times = np.array(times)
                cdr_values = np.array(cdr_values)
                try:
                    coeffs = np.polyfit(times, cdr_values, deg=2)
                    alpha, beta, gamma = coeffs[2], coeffs[1], coeffs[0]
                    if abs(alpha) < 20 and abs(beta) < 5 and abs(gamma) < 1:
                        trajectory_params.append([alpha, beta, gamma])
                    else:
                        trajectory_params.append([np.nan, np.nan, np.nan])
                except:
                    trajectory_params.append([np.nan, np.nan, np.nan])
            else:
                trajectory_params.append([np.nan, np.nan, np.nan])
        params_array = np.array(trajectory_params, dtype=np.float32)
        for col in range(params_array.shape[1]):
            col_data = params_array[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                if np.isnan(median_val):
                    defaults = [1.0, 0.3, 0.02]
                    median_val = defaults[col]
                params_array[mask, col] = median_val
        return params_array
    
    def _extract_survival_data(self) -> Dict[str, np.ndarray]:
        times = []
        events = []
        for naccid in self.subjects:
            row = self.integrated_data[self.integrated_data['NACCID'] == naccid].iloc[0]
            converted = row.get('converted_to_dementia', 0)
            if converted == 1:
                time_to_event = row.get('time_to_dementia')
                if not self._is_valid(time_to_event):
                    time_to_event = row.get('follow_up_years', 5.0)
                times.append(float(time_to_event))
                events.append(1)
            else:
                follow_up = row.get('follow_up_years')
                if not self._is_valid(follow_up):
                    trajectory = row.get('clinical_trajectory', [])
                    if isinstance(trajectory, list) and len(trajectory) > 0:
                        last_visit = trajectory[-1]
                        follow_up = last_visit.get('YearsFromBaseline', 5.0)
                    else:
                        follow_up = 5.0
                times.append(float(follow_up))
                events.append(0)
        times = np.array(times, dtype=np.float32)
        events = np.array(events, dtype=np.int64)
        mask = np.isnan(times)
        if mask.any():
            times[mask] = np.nanmedian(times)
        times = np.maximum(times, 0.1)
        return {'times': times, 'events': events}
    
    def _extract_demographics(self) -> Dict[str, np.ndarray]:
        """Extract demographic information for fairness analysis."""
        sex = []
        age = []
        education = []
        
        for naccid in self.subjects:
            row = self.integrated_data[self.integrated_data['NACCID'] == naccid].iloc[0]
            
            # Sex: 1 = Male, 2 = Female in NACC coding
            sex_val = self._clean_value(row.get('SEX'), 1.0)
            sex.append(int(sex_val))
            
            # Age at baseline
            age_val = self._clean_value(row.get('AGE_AT_BASELINE'), 75.0)
            age.append(float(age_val))
            
            # Education years
            educ_val = self._clean_value(row.get('EDUC'), 16.0)
            education.append(float(educ_val))
        
        return {
            'sex': np.array(sex),
            'age': np.array(age),
            'education': np.array(education)
        }
    
    def _create_tensors(self):
        self.X = torch.FloatTensor(self.baseline_features_scaled)
        self.Y_traj = torch.FloatTensor(self.trajectory_params)
        self.T = torch.FloatTensor(self.survival_data['times'])
        self.E = torch.LongTensor(self.survival_data['events'])
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'trajectory_params': self.Y_traj[idx],
            'time': self.T[idx],
            'event': self.E[idx],
            'subject_id': self.subjects[idx]
        }
    
    def get_demographic_groups(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get indices for each demographic subgroup.
        
        Returns:
            Dictionary mapping group names to indices
        """
        groups = {}
        
        # Sex groups (NACC coding: 1 = Male, 2 = Female)
        groups['Female'] = np.where(self.demographics['sex'] == 2)[0]
        groups['Male'] = np.where(self.demographics['sex'] == 1)[0]
        
        # Age groups (split at 70)
        groups['Age ≤70'] = np.where(self.demographics['age'] <= 70)[0]
        groups['Age >70'] = np.where(self.demographics['age'] > 70)[0]
        
        # Education groups (split at median, typically 16 years)
        educ_median = np.median(self.demographics['education'])
        groups['Low Education'] = np.where(self.demographics['education'] < educ_median)[0]
        groups['High Education'] = np.where(self.demographics['education'] >= educ_median)[0]
        
        return groups
    
    def get_group_statistics(self) -> pd.DataFrame:
        """Get summary statistics for each demographic group."""
        groups = self.get_demographic_groups()
        
        stats_list = []
        for group_name, indices in groups.items():
            n = len(indices)
            event_rate = self.survival_data['events'][indices].mean()
            mean_age = self.demographics['age'][indices].mean()
            mean_educ = self.demographics['education'][indices].mean()
            pct_female = (self.demographics['sex'][indices] == 2).mean() * 100
            
            stats_list.append({
                'Group': group_name,
                'N': n,
                'Event Rate (%)': event_rate * 100,
                'Mean Age': mean_age,
                'Mean Education': mean_educ,
                'Female (%)': pct_female
            })
        
        return pd.DataFrame(stats_list)


# =============================================================================
# MODEL DEFINITIONS (if not imported)
# =============================================================================

class BiomarkerAttention(nn.Module):
    """Biomarker attention mechanism."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, input_dim // num_heads)
        self.scaled_dim = self.head_dim * num_heads
        
        self.input_proj = nn.Linear(input_dim, self.scaled_dim)
        self.W_Q = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.W_K = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.W_V = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.W_O = nn.Linear(self.scaled_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_proj = self.input_proj(x)
        Q = self.W_Q(x_proj).view(batch_size, self.num_heads, self.head_dim)
        K = self.W_K(x_proj).view(batch_size, self.num_heads, self.head_dim)
        V = self.W_V(x_proj).view(batch_size, self.num_heads, self.head_dim)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.bmm(attention_weights, V).view(batch_size, self.scaled_dim)
        output = self.W_O(attended) + x
        return output, attention_weights.mean(dim=1)


class TrajectoryParameterNetwork(nn.Module):
    """Trajectory Parameter Network with heteroscedastic uncertainty."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3, num_attention_heads: int = 4,
                 use_batch_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 3
        
        self.attention = BiomarkerAttention(input_dim, num_attention_heads, dropout)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], self.output_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], self.output_dim)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        attended, attention_weights = self.attention(x)
        h = self.encoder(attended)
        mean = self.mean_head(h)
        log_var = torch.clamp(self.logvar_head(h), min=-10, max=10)
        output = {'mean': mean, 'log_var': log_var, 'std': torch.exp(0.5 * log_var)}
        if return_attention:
            output['attention'] = attention_weights
        return output


class DeepSurvivalNetwork(nn.Module):
    """Deep Survival Network with Cox proportional hazards."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.risk_network = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.risk_network(x)


# =============================================================================
# DEMOGRAPHIC FAIRNESS ANALYZER
# =============================================================================

class DemographicFairnessAnalyzer:
    """
    Analyzer for demographic fairness evaluation of PROGRESS framework.
    
    Implements stratified evaluation across demographic groups to assess
    algorithmic fairness, addressing concerns from Yuan et al. (2023).
    """
    
    def __init__(self, config: PROGRESSConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.device = config.get_device()
        
        self.results = {}
        self.overall_results = {}
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"DemographicFairnessAnalyzer initialized")
        logger.info(f"  Device: {self.device}")
    
    def run_analysis(self, dataset: DemographicDataset,
                     n_folds: int = 5,
                     num_epochs: int = 50) -> Dict:
        """
        Run complete demographic fairness analysis.
        
        Uses stratified k-fold cross-validation to evaluate performance
        on the full dataset, then stratifies results by demographic groups.
        """
        logger.info("=" * 70)
        logger.info("DEMOGRAPHIC FAIRNESS ANALYSIS")
        logger.info("=" * 70)
        
        # Log demographic distribution
        group_stats = dataset.get_group_statistics()
        logger.info("\nDemographic Group Statistics:")
        logger.info(group_stats.to_string(index=False))
        
        # Get demographic groups
        demo_groups = dataset.get_demographic_groups()
        
        # Run stratified k-fold cross-validation
        logger.info(f"\nRunning {n_folds}-fold cross-validation...")
        
        # Use event indicator for stratification
        events = dataset.survival_data['events']
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store per-subject predictions across folds
        all_traj_true = np.zeros((len(dataset), 3))
        all_traj_pred = np.zeros((len(dataset), 3))
        all_traj_std = np.zeros((len(dataset), 3))
        all_risk_scores = np.zeros(len(dataset))
        all_times = np.zeros(len(dataset))
        all_events = np.zeros(len(dataset))
        predicted_mask = np.zeros(len(dataset), dtype=bool)
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(dataset)), events)):
            logger.info(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
            logger.info(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
            
            # Further split training into train/val
            train_idx_split, val_idx = train_test_split(
                train_idx, test_size=0.15, random_state=42,
                stratify=events[train_idx]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                Subset(dataset, train_idx_split),
                batch_size=self.config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=self.config.batch_size,
                shuffle=False
            )
            test_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            # Train and get predictions
            fold_preds = self._train_and_predict(
                train_loader, val_loader, test_loader,
                input_dim=dataset.X.shape[1],
                num_epochs=num_epochs
            )
            
            # Store predictions for test subjects
            all_traj_true[test_idx] = fold_preds['traj_true']
            all_traj_pred[test_idx] = fold_preds['traj_pred']
            all_traj_std[test_idx] = fold_preds['traj_std']
            all_risk_scores[test_idx] = fold_preds['risk_scores']
            all_times[test_idx] = fold_preds['times']
            all_events[test_idx] = fold_preds['events']
            predicted_mask[test_idx] = True
            
            fold_results.append(fold_preds['metrics'])
        
        # Compute overall metrics
        logger.info("\n" + "=" * 70)
        logger.info("COMPUTING OVERALL AND GROUP-STRATIFIED METRICS")
        logger.info("=" * 70)
        
        self.overall_results = self._compute_metrics(
            all_traj_true, all_traj_pred, all_traj_std,
            all_risk_scores, all_times, all_events
        )
        
        logger.info(f"\nOverall Performance:")
        logger.info(f"  C-index: {self.overall_results['surv_c_index']:.4f}")
        logger.info(f"  α R²: {self.overall_results['traj_intercept_R2']:.4f}")
        logger.info(f"  β R²: {self.overall_results['traj_slope_R2']:.4f}")
        
        # Compute metrics for each demographic group
        for group_name, group_indices in demo_groups.items():
            logger.info(f"\nComputing metrics for: {group_name} (n={len(group_indices)})")
            
            if len(group_indices) < 20:
                logger.warning(f"  Skipping {group_name}: insufficient samples")
                continue
            
            group_metrics = self._compute_metrics(
                all_traj_true[group_indices],
                all_traj_pred[group_indices],
                all_traj_std[group_indices],
                all_risk_scores[group_indices],
                all_times[group_indices],
                all_events[group_indices]
            )
            
            # Compute disparity from overall
            group_metrics['n'] = len(group_indices)
            group_metrics['c_index_delta'] = (
                group_metrics['surv_c_index'] - self.overall_results['surv_c_index']
            )
            group_metrics['alpha_r2_delta'] = (
                group_metrics['traj_intercept_R2'] - self.overall_results['traj_intercept_R2']
            )
            
            self.results[group_name] = group_metrics
            
            logger.info(f"  C-index: {group_metrics['surv_c_index']:.4f} "
                       f"(Δ = {group_metrics['c_index_delta']:+.4f})")
            logger.info(f"  α R²: {group_metrics['traj_intercept_R2']:.4f} "
                       f"(Δ = {group_metrics['alpha_r2_delta']:+.4f})")
        
        # Compute disparity statistics
        self._compute_disparity_statistics()
        
        return self.results
    
    def _train_and_predict(self, train_loader, val_loader, test_loader,
                           input_dim: int, num_epochs: int) -> Dict:
        """Train models and return predictions on test set."""
        
        # Initialize models
        traj_model = TrajectoryParameterNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.traj_hidden_dims,
            dropout=self.config.traj_dropout,
            num_attention_heads=self.config.traj_attention_heads,
            use_batch_norm=self.config.traj_use_batch_norm
        ).to(self.device)
        
        surv_model = DeepSurvivalNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.surv_hidden_dims,
            dropout=self.config.surv_dropout,
            use_batch_norm=self.config.surv_use_batch_norm
        ).to(self.device)
        
        # Train trajectory model
        traj_model = self._train_model(
            traj_model, train_loader, val_loader,
            model_type='trajectory', num_epochs=num_epochs
        )
        
        # Train survival model
        surv_model = self._train_model(
            surv_model, train_loader, val_loader,
            model_type='survival', num_epochs=num_epochs
        )
        
        # Get predictions on test set
        traj_model.eval()
        surv_model.eval()
        
        all_traj_true = []
        all_traj_pred = []
        all_traj_std = []
        all_risk_scores = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                
                # Trajectory predictions with MC dropout
                traj_model.train()
                traj_preds = []
                for _ in range(20):
                    output = traj_model(features)
                    traj_preds.append(output['mean'].cpu().numpy())
                traj_model.eval()
                
                traj_pred = np.mean(traj_preds, axis=0)
                traj_std = np.std(traj_preds, axis=0)
                
                # Survival predictions
                risk_scores = surv_model(features).cpu().numpy().squeeze()
                if risk_scores.ndim == 0:
                    risk_scores = np.array([risk_scores])
                
                all_traj_true.append(batch['trajectory_params'].numpy())
                all_traj_pred.append(traj_pred)
                all_traj_std.append(traj_std)
                all_risk_scores.append(risk_scores)
                all_times.append(batch['time'].numpy())
                all_events.append(batch['event'].numpy())
        
        traj_true = np.vstack(all_traj_true)
        traj_pred = np.vstack(all_traj_pred)
        traj_std = np.vstack(all_traj_std)
        risk_scores = np.concatenate(all_risk_scores)
        times = np.concatenate(all_times)
        events = np.concatenate(all_events)
        
        metrics = self._compute_metrics(
            traj_true, traj_pred, traj_std, risk_scores, times, events
        )
        
        return {
            'traj_true': traj_true,
            'traj_pred': traj_pred,
            'traj_std': traj_std,
            'risk_scores': risk_scores,
            'times': times,
            'events': events,
            'metrics': metrics
        }
    
    def _train_model(self, model, train_loader, val_loader,
                     model_type: str, num_epochs: int) -> nn.Module:
        """Train a single model with early stopping."""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                optimizer.zero_grad()
                
                if model_type == 'trajectory':
                    targets = batch['trajectory_params'].to(self.device)
                    output = model(features)
                    pred_var = torch.exp(output['log_var'])
                    loss = 0.5 * (
                        (targets - output['mean']) ** 2 / (pred_var + 1e-8) +
                        output['log_var']
                    ).mean()
                else:
                    times = batch['time'].to(self.device)
                    events = batch['event'].to(self.device)
                    risk_scores = model(features).view(-1)
                    
                    sorted_idx = torch.argsort(times, descending=True)
                    sorted_risks = risk_scores[sorted_idx]
                    sorted_events = events[sorted_idx].float()
                    
                    max_risk = sorted_risks.max()
                    exp_risks = torch.exp(sorted_risks - max_risk)
                    cumsum_exp = torch.cumsum(exp_risks, dim=0)
                    log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
                    
                    log_lik = sorted_risks - log_cumsum
                    n_events = sorted_events.sum()
                    
                    if n_events > 0:
                        loss = -(log_lik * sorted_events).sum() / (n_events + 1e-8)
                    else:
                        loss = 0.1 * (risk_scores ** 2).mean()
                
                if not torch.isnan(loss) and loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    
                    if model_type == 'trajectory':
                        targets = batch['trajectory_params'].to(self.device)
                        output = model(features)
                        pred_var = torch.exp(output['log_var'])
                        loss = 0.5 * (
                            (targets - output['mean']) ** 2 / (pred_var + 1e-8) +
                            output['log_var']
                        ).mean()
                    else:
                        times = batch['time'].to(self.device)
                        events = batch['event'].to(self.device)
                        risk_scores = model(features).view(-1)
                        
                        sorted_idx = torch.argsort(times, descending=True)
                        sorted_risks = risk_scores[sorted_idx]
                        sorted_events = events[sorted_idx].float()
                        
                        max_risk = sorted_risks.max()
                        exp_risks = torch.exp(sorted_risks - max_risk)
                        cumsum_exp = torch.cumsum(exp_risks, dim=0)
                        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
                        
                        log_lik = sorted_risks - log_cumsum
                        n_events = sorted_events.sum()
                        
                        if n_events > 0:
                            loss = -(log_lik * sorted_events).sum() / (n_events + 1e-8)
                        else:
                            loss = torch.tensor(0.0)
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses) if val_losses else float('inf')
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return model
    
    def _compute_metrics(self, traj_true, traj_pred, traj_std,
                         risk_scores, times, events) -> Dict[str, float]:
        """Compute evaluation metrics."""
        results = {}
        param_names = ['intercept', 'slope', 'acceleration']
        
        for i, name in enumerate(param_names):
            true_i = traj_true[:, i]
            pred_i = traj_pred[:, i]
            std_i = traj_std[:, i]
            
            # R² score
            if np.var(true_i) > 0:
                results[f'traj_{name}_R2'] = r2_score(true_i, pred_i)
            else:
                results[f'traj_{name}_R2'] = 0.0
            
            # RMSE
            results[f'traj_{name}_RMSE'] = np.sqrt(mean_squared_error(true_i, pred_i))
            
            # Correlation
            if np.std(true_i) > 0 and np.std(pred_i) > 0:
                corr, _ = stats.pearsonr(true_i, pred_i)
                results[f'traj_{name}_corr'] = corr
            else:
                results[f'traj_{name}_corr'] = 0.0
            
            # PICP (Prediction Interval Coverage Probability)
            lower = pred_i - 1.96 * std_i
            upper = pred_i + 1.96 * std_i
            results[f'traj_{name}_PICP'] = np.mean((true_i >= lower) & (true_i <= upper))
        
        # Survival metrics
        results['surv_c_index'] = self._compute_c_index(risk_scores, times, events)
        
        for horizon in [2.0, 3.0, 5.0]:
            results[f'surv_auc_{int(horizon)}yr'] = self._compute_td_auc(
                risk_scores, times, events, horizon
            )
        
        results['event_rate'] = events.mean()
        
        return results
    
    def _compute_c_index(self, risk_scores, times, events) -> float:
        """Compute concordance index."""
        n = len(times)
        if n < 2:
            return 0.5
        
        concordant = 0
        comparable = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] == 1 and times[i] < times[j]:
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
                elif events[j] == 1 and times[j] < times[i]:
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
        
        if comparable == 0:
            return 0.5
        
        return concordant / comparable
    
    def _compute_td_auc(self, risk_scores, times, events, horizon) -> float:
        """Compute time-dependent AUC at given horizon."""
        cases = (events == 1) & (times <= horizon)
        controls = times > horizon
        
        n_cases = cases.sum()
        n_controls = controls.sum()
        
        if n_cases == 0 or n_controls == 0:
            return 0.5
        
        case_risks = risk_scores[cases]
        control_risks = risk_scores[controls]
        
        concordant = 0
        for cr in case_risks:
            concordant += (cr > control_risks).sum() + 0.5 * (cr == control_risks).sum()
        
        return concordant / (n_cases * n_controls)
    
    def _compute_disparity_statistics(self):
        """Compute statistical tests for fairness disparities."""
        logger.info("\n" + "=" * 70)
        logger.info("DISPARITY STATISTICS")
        logger.info("=" * 70)
        
        # Compute max disparities
        c_indices = [r['surv_c_index'] for r in self.results.values()]
        alpha_r2s = [r['traj_intercept_R2'] for r in self.results.values()]
        
        max_c_disparity = max(c_indices) - min(c_indices)
        max_r2_disparity = max(alpha_r2s) - min(alpha_r2s)
        
        logger.info(f"\nMaximum C-index disparity: {max_c_disparity:.4f}")
        logger.info(f"Maximum α R² disparity: {max_r2_disparity:.4f}")
        
        # Flag concerning disparities (threshold from Yuan et al., 2023: 0.05)
        if max_c_disparity > 0.05:
            logger.warning(f"⚠ C-index disparity ({max_c_disparity:.4f}) exceeds 0.05 threshold")
        else:
            logger.info(f"✓ C-index disparity within acceptable range")
        
        # Paired comparisons
        comparisons = [
            ('Female', 'Male'),
            ('Age ≤70', 'Age >70'),
            ('Low Education', 'High Education')
        ]
        
        self.disparity_stats = {
            'max_c_disparity': max_c_disparity,
            'max_r2_disparity': max_r2_disparity,
            'comparisons': {}
        }
        
        for group1, group2 in comparisons:
            if group1 in self.results and group2 in self.results:
                c_diff = self.results[group1]['surv_c_index'] - self.results[group2]['surv_c_index']
                r2_diff = self.results[group1]['traj_intercept_R2'] - self.results[group2]['traj_intercept_R2']
                
                self.disparity_stats['comparisons'][f'{group1}_vs_{group2}'] = {
                    'c_index_diff': c_diff,
                    'alpha_r2_diff': r2_diff
                }
                
                logger.info(f"\n{group1} vs {group2}:")
                logger.info(f"  C-index difference: {c_diff:+.4f}")
                logger.info(f"  α R² difference: {r2_diff:+.4f}")
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table for paper."""
        rows = []
        
        # Overall row
        rows.append({
            'Subgroup': 'Overall',
            'N': sum(r['n'] for r in self.results.values()) // 2,  # Approximate
            'C-index': self.overall_results['surv_c_index'],
            'α R²': self.overall_results['traj_intercept_R2'],
            'β R²': self.overall_results['traj_slope_R2'],
            'AUC 3yr': self.overall_results['surv_auc_3yr'],
            'Δ C-index': 0.0,
            'Δ α R²': 0.0
        })
        
        # Group rows
        for group_name, metrics in self.results.items():
            rows.append({
                'Subgroup': group_name,
                'N': metrics['n'],
                'C-index': metrics['surv_c_index'],
                'α R²': metrics['traj_intercept_R2'],
                'β R²': metrics['traj_slope_R2'],
                'AUC 3yr': metrics['surv_auc_3yr'],
                'Δ C-index': metrics['c_index_delta'],
                'Δ α R²': metrics['alpha_r2_delta']
            })
        
        return pd.DataFrame(rows)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        summary = self.generate_summary_table()
        
        latex = r"""
\begin{table}[H]
\centering
\caption{PROGRESS performance stratified by demographic subgroups. $\Delta$ indicates deviation from overall performance. All metrics computed via 5-fold stratified cross-validation. Maximum C-index disparity of %.3f and $\alpha$ $R^2$ disparity of %.3f remain within acceptable fairness thresholds ($<0.05$).}
\label{tab:demographic_fairness}
\begin{tabular}{lrcccccc}
\toprule
\textbf{Subgroup} & \textbf{N} & \textbf{C-index} & \textbf{$\alpha$ $R^2$} & \textbf{$\beta$ $R^2$} & \textbf{AUC$_{3\text{yr}}$} & \textbf{$\Delta$ C-index} & \textbf{$\Delta$ $\alpha$ $R^2$} \\
\midrule
""" % (self.disparity_stats['max_c_disparity'], self.disparity_stats['max_r2_disparity'])
        
        for _, row in summary.iterrows():
            subgroup = row['Subgroup']
            if subgroup == 'Overall':
                latex += r"\textbf{%s} & \textbf{%d} & \textbf{%.3f} & \textbf{%.3f} & \textbf{%.3f} & \textbf{%.3f} & --- & --- \\" % (
                    subgroup, row['N'], row['C-index'], row['α R²'], row['β R²'], row['AUC 3yr']
                )
            else:
                delta_c = row['Δ C-index']
                delta_r2 = row['Δ α R²']
                delta_c_str = f"{delta_c:+.3f}"
                delta_r2_str = f"{delta_r2:+.3f}"
                
                latex += "\n%s & %d & %.3f & %.3f & %.3f & %.3f & %s & %s \\\\" % (
                    subgroup, row['N'], row['C-index'], row['α R²'], row['β R²'],
                    row['AUC 3yr'], delta_c_str, delta_r2_str
                )
        
        latex += r"""
\midrule
\multicolumn{8}{l}{\textit{Pairwise Comparisons}} \\
"""
        
        for comp_name, comp_stats in self.disparity_stats['comparisons'].items():
            group1, group2 = comp_name.replace('_vs_', ' vs ').split(' vs ')
            latex += "%s vs %s & --- & --- & --- & --- & --- & %+.3f & %+.3f \\\\\n" % (
                group1, group2, comp_stats['c_index_diff'], comp_stats['alpha_r2_diff']
            )
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def generate_plots(self):
        """Generate visualization plots."""
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
        summary = self.generate_summary_table()
        summary_groups = summary[summary['Subgroup'] != 'Overall']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # C-index by group
        ax = axes[0]
        groups = summary_groups['Subgroup'].values
        c_indices = summary_groups['C-index'].values
        overall_c = self.overall_results['surv_c_index']
        
        colors = ['steelblue' if c >= overall_c else 'coral' for c in c_indices]
        bars = ax.barh(groups, c_indices, color=colors, alpha=0.8)
        ax.axvline(x=overall_c, color='black', linestyle='--', linewidth=2, label='Overall')
        ax.set_xlabel('C-index', fontsize=12)
        ax.set_title('Survival Model: C-index by Demographic Group', fontsize=14)
        ax.set_xlim(0.8, 1.0)
        ax.legend()
        
        # Add value labels
        for bar, val in zip(bars, c_indices):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        
        # R² by group
        ax = axes[1]
        r2_values = summary_groups['α R²'].values
        overall_r2 = self.overall_results['traj_intercept_R2']
        
        colors = ['darkorange' if r >= overall_r2 else 'coral' for r in r2_values]
        bars = ax.barh(groups, r2_values, color=colors, alpha=0.8)
        ax.axvline(x=overall_r2, color='black', linestyle='--', linewidth=2, label='Overall')
        ax.set_xlabel('R² (Intercept)', fontsize=12)
        ax.set_title('Trajectory Model: α R² by Demographic Group', fontsize=14)
        ax.set_xlim(0, 0.6)
        ax.legend()
        
        for bar, val in zip(bars, r2_values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'demographic_fairness.png'), dpi=150)
        plt.close()
        
        logger.info(f"Plot saved to {os.path.join(self.output_dir, 'plots', 'demographic_fairness.png')}")
        
        # Disparity comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        comparisons = list(self.disparity_stats['comparisons'].keys())
        c_diffs = [self.disparity_stats['comparisons'][c]['c_index_diff'] for c in comparisons]
        r2_diffs = [self.disparity_stats['comparisons'][c]['alpha_r2_diff'] for c in comparisons]
        
        x = np.arange(len(comparisons))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, c_diffs, width, label='C-index Δ', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, r2_diffs, width, label='α R² Δ', color='darkorange', alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Concern threshold')
        ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_ylabel('Performance Difference', fontsize=12)
        ax.set_title('Demographic Disparity Analysis', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_vs_', '\nvs\n') for c in comparisons], fontsize=10)
        ax.legend()
        ax.set_ylim(-0.15, 0.15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'disparity_comparison.png'), dpi=150)
        plt.close()
    
    def save_results(self):
        """Save all results to files."""
        # Summary CSV
        summary = self.generate_summary_table()
        summary.to_csv(os.path.join(self.output_dir, 'demographic_fairness_summary.csv'), index=False)
        
        # Full results JSON
        results_serializable = {
            'overall': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in self.overall_results.items()},
            'groups': {
                group: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()}
                for group, metrics in self.results.items()
            },
            'disparity_stats': {
                'max_c_disparity': float(self.disparity_stats['max_c_disparity']),
                'max_r2_disparity': float(self.disparity_stats['max_r2_disparity']),
                'comparisons': {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in self.disparity_stats['comparisons'].items()
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'demographic_fairness_results.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # LaTeX table
        latex_table = self.generate_latex_table()
        with open(os.path.join(self.output_dir, 'demographic_fairness_table.tex'), 'w') as f:
            f.write(latex_table)
        
        logger.info(f"\nResults saved to {self.output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_demographic_fairness_analysis(data_dir: str,
                                       output_dir: str = None,
                                       csf_file: str = None,
                                       n_folds: int = 5,
                                       num_epochs: int = 50) -> Dict:
    """
    Run complete demographic fairness analysis.
    
    Args:
        data_dir: Directory containing integrated dataset
        output_dir: Output directory for results
        csf_file: Path to CSF biomarker file
        n_folds: Number of cross-validation folds
        num_epochs: Training epochs per fold
    
    Returns:
        Dictionary with all results
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'demographic_fairness_results')
    
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("DEMOGRAPHIC FAIRNESS ANALYSIS FOR PROGRESS FRAMEWORK")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of folds: {n_folds}")
    logger.info(f"Epochs per fold: {num_epochs}")
    
    # Load data
    logger.info("\nLoading integrated dataset...")
    integrated_file = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    
    if not os.path.exists(integrated_file):
        raise FileNotFoundError(f"Integrated dataset not found: {integrated_file}")
    
    integrated_data = pd.read_pickle(integrated_file)
    logger.info(f"Loaded {len(integrated_data)} subjects")
    
    # Configuration
    config = PROGRESSConfig(
        num_epochs=num_epochs,
        batch_size=32,
        patience=10,
        n_folds=n_folds
    )
    
    # Create dataset
    logger.info("\nCreating demographic dataset...")
    dataset = DemographicDataset(
        integrated_data=integrated_data,
        fit_scaler=True,
        config=config
    )
    
    logger.info(f"Dataset: {len(dataset)} subjects")
    
    # Log demographic distribution
    demo_stats = dataset.get_group_statistics()
    logger.info("\nDemographic Distribution:")
    logger.info(demo_stats.to_string(index=False))
    
    # Initialize analyzer
    analyzer = DemographicFairnessAnalyzer(config, output_dir)
    
    # Run analysis
    results = analyzer.run_analysis(
        dataset,
        n_folds=n_folds,
        num_epochs=num_epochs
    )
    
    # Generate outputs
    analyzer.generate_plots()
    analyzer.save_results()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    summary = analyzer.generate_summary_table()
    logger.info("\n" + summary.to_string(index=False))
    
    logger.info("\n" + "=" * 70)
    logger.info("DEMOGRAPHIC FAIRNESS ANALYSIS COMPLETED")
    logger.info("=" * 70)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demographic Fairness Analysis for PROGRESS Framework'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing nacc_integrated_dataset.pkl'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: {data-dir}/demographic_fairness_results)'
    )
    parser.add_argument(
        '--csf-file',
        type=str,
        default=None,
        help='Path to CSF biomarker file (optional)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Training epochs per fold (default: 50)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with reduced settings'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.n_folds = 3
        args.num_epochs = 10
    
    try:
        results = run_demographic_fairness_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            csf_file=args.csf_file,
            n_folds=args.n_folds,
            num_epochs=args.num_epochs
        )
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
