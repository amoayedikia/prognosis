#!/usr/bin/env python3
"""
Learning Rate Sensitivity Analysis for Cross-Center Validation

This script tests PROGRESS cross-center performance across multiple learning rates
to assess model robustness to this hyperparameter.

Usage:
    python cross_center_lr_analysis.py --data-dir ./dataset --csf-file ./investigator_fcsf_nacc69.csv

Author: Alireza
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
from sklearn.model_selection import train_test_split
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
    log_file = os.path.join(output_dir, 'lr_analysis.log')
    
    logger = logging.getLogger('lr_analysis')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger

logger = logging.getLogger('lr_analysis')


# =============================================================================
# DATASET (same as cross_center_validation.py)
# =============================================================================

class CrossCenterDataset(Dataset):
    """Dataset with center tracking for cross-center validation."""
    
    MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}
    
    def __init__(self, integrated_data: pd.DataFrame, scaler: RobustScaler = None,
                 fit_scaler: bool = False, config: 'PROGRESSConfig' = None):
        self.config = config or PROGRESSConfig()
        self.integrated_data = integrated_data
        
        self.subjects = self._get_valid_subjects()
        self.baseline_features, self.feature_names = self._extract_baseline_features()
        self.trajectory_params = self._compute_trajectory_parameters()
        self.survival_data = self._extract_survival_data()
        self.center_ids = self._extract_center_ids()
        
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
                params_array[mask, col] = median_val if not np.isnan(median_val) else 0.0
        return params_array
    
    def _extract_survival_data(self) -> Dict[str, np.ndarray]:
        times, events = [], []
        for naccid in self.subjects:
            row = self.integrated_data[self.integrated_data['NACCID'] == naccid].iloc[0]
            trajectory = row.get('clinical_trajectory', [])
            time_to_event, event_occurred = 5.0, 0
            if isinstance(trajectory, list) and len(trajectory) > 0:
                for visit in trajectory:
                    t = visit.get('YearsFromBaseline', 0)
                    dx = visit.get('NACCUDSD', 0)
                    if dx == 4:
                        time_to_event = max(0.1, float(t))
                        event_occurred = 1
                        break
                    time_to_event = max(time_to_event, float(t) if self._is_valid(t) else 0)
            times.append(time_to_event)
            events.append(event_occurred)
        return {'times': np.array(times, dtype=np.float32), 
                'events': np.array(events, dtype=np.int32)}
    
    def _extract_center_ids(self) -> np.ndarray:
        center_ids = []
        for naccid in self.subjects:
            row = self.integrated_data[self.integrated_data['NACCID'] == naccid].iloc[0]
            adc = row.get('NACCADC', 0)
            if pd.isna(adc) or adc in self.MISSING_CODES:
                adc = 0
            center_ids.append(int(adc))
        return np.array(center_ids, dtype=np.int32)
    
    def _create_tensors(self):
        self.X = torch.tensor(self.baseline_features_scaled, dtype=torch.float32)
        self.y_traj = torch.tensor(self.trajectory_params, dtype=torch.float32)
        self.times = torch.tensor(self.survival_data['times'], dtype=torch.float32)
        self.events = torch.tensor(self.survival_data['events'], dtype=torch.float32)
        self.centers = torch.tensor(self.center_ids, dtype=torch.int32)
    
    def __len__(self) -> int:
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.X[idx],
            'trajectory_params': self.y_traj[idx],
            'time': self.times[idx],
            'event': self.events[idx],
            'center': self.centers[idx]
        }
    
    def get_center_indices(self, center_id: int) -> np.ndarray:
        return np.where(self.center_ids == center_id)[0]
    
    def get_unique_centers(self) -> np.ndarray:
        return np.unique(self.center_ids)
    
    def get_center_sizes(self) -> Dict[int, int]:
        unique, counts = np.unique(self.center_ids, return_counts=True)
        return dict(zip(unique, counts))


# =============================================================================
# LEARNING RATE ANALYSIS
# =============================================================================

class LearningRateAnalyzer:
    """Analyze cross-center performance across different learning rates."""
    
    def __init__(self, config: 'PROGRESSConfig', output_dir: str):
        self.base_config = config
        self.output_dir = output_dir
        self.device = config.get_device()
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
        
        logger.info(f"LearningRateAnalyzer initialized")
        logger.info(f"  Device: {self.device}")
    
    def run_analysis(self,
                     dataset: CrossCenterDataset,
                     learning_rates: List[float],
                     min_center_size: int = 20,
                     max_centers: int = None,
                     num_epochs: int = 50) -> Dict[str, Any]:
        """
        Run LOCO validation for each learning rate.
        
        Args:
            dataset: CrossCenterDataset
            learning_rates: List of learning rates to test
            min_center_size: Minimum subjects per center
            max_centers: Maximum centers to evaluate
            num_epochs: Training epochs per fold
            
        Returns:
            Dictionary with results per learning rate
        """
        logger.info("=" * 70)
        logger.info("LEARNING RATE SENSITIVITY ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Learning rates to test: {learning_rates}")
        
        # Get valid centers
        center_sizes = dataset.get_center_sizes()
        valid_centers = [c for c, size in center_sizes.items() 
                        if size >= min_center_size and c != -1 and c != 0]
        
        if max_centers:
            valid_centers = valid_centers[:max_centers]
        
        logger.info(f"Centers to evaluate: {len(valid_centers)}")
        
        all_results = {}
        
        for lr in learning_rates:
            logger.info(f"\n{'='*70}")
            logger.info(f"TESTING LEARNING RATE: {lr}")
            logger.info(f"{'='*70}")
            
            lr_results = self._run_loco_for_lr(
                dataset, valid_centers, lr, num_epochs
            )
            all_results[lr] = lr_results
            
            # Log summary for this LR
            c_indices = [r['surv_c_index'] for r in lr_results.values()]
            r2_values = [r['traj_intercept_R2'] for r in lr_results.values()]
            
            logger.info(f"\nLR={lr} Summary:")
            logger.info(f"  C-index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
            logger.info(f"  Traj R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
        
        self.results = all_results
        return all_results
    
    def _run_loco_for_lr(self,
                         dataset: CrossCenterDataset,
                         valid_centers: List[int],
                         learning_rate: float,
                         num_epochs: int) -> Dict[int, Dict]:
        """Run LOCO validation for a specific learning rate."""
        results = {}
        
        for i, held_out_center in enumerate(tqdm(valid_centers, desc=f"LR={learning_rate}")):
            center_sizes = dataset.get_center_sizes()
            
            # Get train/test indices
            test_indices = dataset.get_center_indices(held_out_center)
            train_indices = np.array([
                idx for idx in range(len(dataset))
                if dataset.center_ids[idx] != held_out_center 
                and dataset.center_ids[idx] != -1
                and dataset.center_ids[idx] != 0
            ])
            
            if len(train_indices) < 50 or len(test_indices) < 5:
                continue
            
            train_idx, val_idx = train_test_split(
                train_indices, test_size=0.15, random_state=42
            )
            
            # Create data loaders
            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=self.base_config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=self.base_config.batch_size,
                shuffle=False
            )
            test_loader = DataLoader(
                Subset(dataset, test_indices),
                batch_size=self.base_config.batch_size,
                shuffle=False
            )
            
            # Train and evaluate with this learning rate
            fold_results = self._train_and_evaluate(
                train_loader, val_loader, test_loader,
                input_dim=dataset.X.shape[1],
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
            
            fold_results['center_id'] = int(held_out_center)
            fold_results['test_size'] = len(test_indices)
            
            results[held_out_center] = fold_results
        
        return results
    
    def _train_and_evaluate(self,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            test_loader: DataLoader,
                            input_dim: int,
                            learning_rate: float,
                            num_epochs: int) -> Dict[str, float]:
        """Train models and evaluate with specific learning rate."""
        
        # Initialize models
        traj_model = TrajectoryParameterNetwork(
            input_dim=input_dim,
            hidden_dims=self.base_config.traj_hidden_dims,
            dropout=self.base_config.traj_dropout,
            num_attention_heads=self.base_config.traj_attention_heads,
            use_batch_norm=self.base_config.traj_use_batch_norm
        ).to(self.device)
        
        surv_model = DeepSurvivalNetwork(
            input_dim=input_dim,
            hidden_dims=self.base_config.surv_hidden_dims,
            dropout=self.base_config.surv_dropout,
            use_batch_norm=self.base_config.surv_use_batch_norm
        ).to(self.device)
        
        # Train trajectory model
        traj_model = self._train_model(
            traj_model, train_loader, val_loader,
            model_type='trajectory',
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        
        # Train survival model
        surv_model = self._train_model(
            surv_model, train_loader, val_loader,
            model_type='survival',
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        
        # Evaluate
        return self._evaluate_models(traj_model, surv_model, test_loader)
    
    def _train_model(self,
                     model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_type: str,
                     learning_rate: float,
                     num_epochs: int) -> nn.Module:
        """Train a model with specific learning rate."""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.base_config.weight_decay
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
                    sorted_events = events[sorted_idx]
                    
                    max_risk = sorted_risks.max()
                    exp_risks = torch.exp(sorted_risks - max_risk)
                    cumsum_exp = torch.cumsum(exp_risks, dim=0)
                    log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
                    
                    log_lik = sorted_risks - log_cumsum
                    loss = -(log_lik * sorted_events).sum() / (sorted_events.sum() + 1e-8)
                
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
                        sorted_events = events[sorted_idx]
                        
                        max_risk = sorted_risks.max()
                        exp_risks = torch.exp(sorted_risks - max_risk)
                        cumsum_exp = torch.cumsum(exp_risks, dim=0)
                        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
                        
                        log_lik = sorted_risks - log_cumsum
                        loss = -(log_lik * sorted_events).sum() / (sorted_events.sum() + 1e-8)
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses) if val_losses else float('inf')
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return model
    
    def _evaluate_models(self,
                        traj_model: nn.Module,
                        surv_model: nn.Module,
                        test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate both models."""
        traj_model.eval()
        surv_model.eval()
        
        all_traj_true, all_traj_pred = [], []
        all_risk_scores, all_times, all_events = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                
                # Trajectory
                traj_model.train()
                traj_preds = []
                for _ in range(20):
                    output = traj_model(features)
                    traj_preds.append(output['mean'].cpu().numpy())
                traj_model.eval()
                traj_pred = np.mean(traj_preds, axis=0)
                
                # Survival
                risk_scores = surv_model(features).cpu().numpy().squeeze()
                
                all_traj_true.append(batch['trajectory_params'].numpy())
                all_traj_pred.append(traj_pred)
                all_risk_scores.append(risk_scores)
                all_times.append(batch['time'].numpy())
                all_events.append(batch['event'].numpy())
        
        traj_true = np.vstack(all_traj_true)
        traj_pred = np.vstack(all_traj_pred)
        risk_scores = np.concatenate(all_risk_scores)
        times = np.concatenate(all_times)
        events = np.concatenate(all_events)
        
        # Metrics
        results = {}
        param_names = ['intercept', 'slope', 'acceleration']
        
        for i, name in enumerate(param_names):
            true_i, pred_i = traj_true[:, i], traj_pred[:, i]
            results[f'traj_{name}_R2'] = r2_score(true_i, pred_i) if np.var(true_i) > 0 else 0
            results[f'traj_{name}_RMSE'] = np.sqrt(mean_squared_error(true_i, pred_i))
        
        results['surv_c_index'] = self._compute_c_index(risk_scores, times, events)
        
        for horizon in [2.0, 3.0, 5.0]:
            results[f'surv_auc_{int(horizon)}yr'] = self._compute_td_auc(
                risk_scores, times, events, horizon
            )
        
        return results
    
    def _compute_c_index(self, risk_scores, times, events) -> float:
        n = len(times)
        concordant, discordant, tied = 0, 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] == 1 and times[i] < times[j]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied += 0.5
                elif events[j] == 1 and times[j] < times[i]:
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied += 0.5
        total = concordant + discordant + tied
        return (concordant + 0.5 * tied) / total if total > 0 else 0.5
    
    def _compute_td_auc(self, risk_scores, times, events, horizon) -> float:
        cases = (times <= horizon) & (events == 1)
        controls = times > horizon
        n_cases, n_controls = cases.sum(), controls.sum()
        if n_cases == 0 or n_controls == 0:
            return 0.5
        case_risks, control_risks = risk_scores[cases], risk_scores[controls]
        concordant = 0
        for cr in case_risks:
            concordant += (cr > control_risks).sum() + 0.5 * (cr == control_risks).sum()
        return concordant / (n_cases * n_controls)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of results across learning rates."""
        rows = []
        
        for lr, lr_results in self.results.items():
            c_indices = [r['surv_c_index'] for r in lr_results.values()]
            auc_3yr = [r['surv_auc_3yr'] for r in lr_results.values()]
            r2_int = [r['traj_intercept_R2'] for r in lr_results.values()]
            r2_slope = [r['traj_slope_R2'] for r in lr_results.values()]
            
            rows.append({
                'Learning Rate': lr,
                'C-index (mean)': np.mean(c_indices),
                'C-index (std)': np.std(c_indices),
                'AUC 3yr (mean)': np.mean(auc_3yr),
                'AUC 3yr (std)': np.std(auc_3yr),
                'R² Intercept (mean)': np.mean(r2_int),
                'R² Intercept (std)': np.std(r2_int),
                'R² Slope (mean)': np.mean(r2_slope),
                'R² Slope (std)': np.std(r2_slope)
            })
        
        return pd.DataFrame(rows)
    
    def generate_plots(self):
        """Generate comparison plots."""
        if not self.results:
            return
        
        summary = self.generate_summary_table()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # C-index by learning rate
        ax = axes[0]
        lrs = summary['Learning Rate'].values
        c_means = summary['C-index (mean)'].values
        c_stds = summary['C-index (std)'].values
        
        x_pos = np.arange(len(lrs))
        ax.bar(x_pos, c_means, yerr=c_stds, capsize=5, color='steelblue', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('C-index')
        ax.set_title('Survival C-index by Learning Rate')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.set_ylim(0.8, 1.0)
        
        # R² by learning rate
        ax = axes[1]
        r2_means = summary['R² Intercept (mean)'].values
        r2_stds = summary['R² Intercept (std)'].values
        
        ax.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, color='darkorange', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('R² (Intercept)')
        ax.set_title('Trajectory R² by Learning Rate')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lr_comparison.png'), dpi=150)
        plt.close()
        
        logger.info(f"Plot saved to {os.path.join(self.output_dir, 'lr_comparison.png')}")
    
    def save_results(self):
        """Save all results."""
        # Summary table
        summary = self.generate_summary_table()
        summary.to_csv(os.path.join(self.output_dir, 'lr_comparison_summary.csv'), index=False)
        
        # Full results
        results_serializable = {}
        for lr, lr_results in self.results.items():
            results_serializable[str(lr)] = {
                int(k): {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                        for kk, vv in v.items()}
                for k, v in lr_results.items()
            }
        
        with open(os.path.join(self.output_dir, 'lr_analysis_full.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        
        # Print summary table
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY TABLE")
        logger.info("=" * 70)
        print(summary.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def run_lr_analysis(data_dir: str,
                    output_dir: str,
                    csf_file: str,
                    learning_rates: List[float] = None,
                    min_center_size: int = 20,
                    max_centers: int = None,
                    num_epochs: int = 50) -> Dict:
    """Run learning rate analysis."""
    
    if learning_rates is None:
        learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("LEARNING RATE SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Learning rates: {learning_rates}")
    
    # Load data
    integrated_file = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    integrated_data = pd.read_pickle(integrated_file)
    logger.info(f"Loaded {len(integrated_data)} subjects")
    
    # Merge NACCADC
    if 'NACCADC' not in integrated_data.columns:
        csf_data = pd.read_csv(csf_file)
        adc_mapping = csf_data[['NACCID', 'NACCADC']].drop_duplicates()
        integrated_data = integrated_data.merge(adc_mapping, on='NACCID', how='left')
        integrated_data['NACCADC'] = integrated_data['NACCADC'].fillna(-1).astype(int)
        logger.info("Merged NACCADC from CSF file")
    
    # Config
    config = PROGRESSConfig(num_epochs=num_epochs, batch_size=32, patience=10)
    
    # Dataset
    dataset = CrossCenterDataset(integrated_data=integrated_data, fit_scaler=True, config=config)
    logger.info(f"Dataset: {len(dataset)} subjects, {len(dataset.get_unique_centers())} centers")
    
    # Run analysis
    analyzer = LearningRateAnalyzer(config, output_dir)
    results = analyzer.run_analysis(
        dataset,
        learning_rates=learning_rates,
        min_center_size=min_center_size,
        max_centers=max_centers,
        num_epochs=num_epochs
    )
    
    # Generate outputs
    analyzer.generate_plots()
    analyzer.save_results()
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETED")
    logger.info("=" * 70)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Learning Rate Sensitivity Analysis')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./lr_analysis_results')
    parser.add_argument('--csf-file', type=str, required=True)
    parser.add_argument('--learning-rates', type=float, nargs='+', 
                        default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                        help='Learning rates to test (default: 1e-4 5e-4 1e-3 5e-3 1e-2)')
    parser.add_argument('--min-center-size', type=int, default=20)
    parser.add_argument('--max-centers', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    try:
        run_lr_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            csf_file=args.csf_file,
            learning_rates=args.learning_rates,
            min_center_size=args.min_center_size,
            max_centers=args.max_centers,
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
