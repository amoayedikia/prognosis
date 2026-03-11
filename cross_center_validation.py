#!/usr/bin/env python3
"""
Cross-Center Validation for PROGRESS Framework

This script implements the cross-center generalizability analysis described in
Section 3.3 of the PROGRESS paper. It evaluates model performance across the
43 Alzheimer's Disease Research Centers (ADRCs) in the NACC dataset.

Experiments:
    1. Leave-One-Center-Out (LOCO) Validation
    2. Performance stratified by center size (small/medium/large)
    3. Center-wise performance variability analysis
    4. Generalizability metrics across heterogeneous clinical settings

Output:
    - cross_center_results.json: Complete results for all centers
    - cross_center_summary.csv: Summary statistics by center
    - cross_center_plots/: Visualization of cross-center performance

Usage:
    python cross_center_validation.py --data-dir ./dataset --output-dir ./cross_center_results

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

# Import from PROGRESS module (assumes PROGRESS.py is in the same directory or path)
# If running standalone, copy necessary classes or adjust import
try:
    from PROGRESS import (
        PROGRESSConfig, PROGRESSDataset, PROGRESSTrainer,
        TrajectoryParameterNetwork, DeepSurvivalNetwork,
        PROGRESSMetrics, setup_logging
    )
except ImportError:
    print("Warning: Could not import from PROGRESS.py. Using local definitions.")
    # We'll define necessary components inline if import fails


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_cross_center_logging(output_dir: str) -> logging.Logger:
    """Configure logging for cross-center validation."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file = os.path.join(output_dir, 'cross_center_validation.log')
    
    # Clear existing handlers
    logger = logging.getLogger('cross_center')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger

logger = logging.getLogger('cross_center')


# =============================================================================
# CROSS-CENTER DATASET
# =============================================================================

class CrossCenterDataset(Dataset):
    """
    Dataset wrapper that tracks center (ADC) membership for cross-center validation.
    
    Extends the base PROGRESSDataset to include center information for:
    - Leave-One-Center-Out validation
    - Center-stratified analysis
    - Generalizability assessment
    """
    
    MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}
    
    def __init__(self,
                 integrated_data: pd.DataFrame,
                 scaler: RobustScaler = None,
                 fit_scaler: bool = False,
                 config: 'PROGRESSConfig' = None):
        """
        Initialize dataset with center tracking.
        
        Args:
            integrated_data: Subject-level integrated dataset
            scaler: Pre-fitted feature scaler
            fit_scaler: Whether to fit a new scaler
            config: Configuration object
        """
        self.config = config or PROGRESSConfig()
        self.integrated_data = integrated_data
        
        # Get valid subjects
        self.subjects = self._get_valid_subjects()
        logger.info(f"CrossCenterDataset: {len(self.subjects)} valid subjects")
        
        # Extract features, targets, and center info
        self.baseline_features, self.feature_names = self._extract_baseline_features()
        self.trajectory_params = self._compute_trajectory_parameters()
        self.survival_data = self._extract_survival_data()
        self.center_ids = self._extract_center_ids()
        
        # Log center distribution
        unique_centers = np.unique(self.center_ids)
        logger.info(f"  Centers: {len(unique_centers)} unique ADCs")
        
        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()
            self.baseline_features_scaled = self.scaler.fit_transform(self.baseline_features)
        elif scaler is not None:
            self.scaler = scaler
            self.baseline_features_scaled = self.scaler.transform(self.baseline_features)
        else:
            self.scaler = None
            self.baseline_features_scaled = self.baseline_features
        
        # Convert to tensors
        self._create_tensors()
    
    def _is_valid(self, value) -> bool:
        """Check if value is valid."""
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if value in self.MISSING_CODES:
            return False
        return True
    
    def _clean_value(self, value, default: float = np.nan) -> float:
        """Clean a value by handling missing codes."""
        if not self._is_valid(value):
            return default
        return float(value)
    
    def _get_valid_subjects(self) -> List[str]:
        """Get subjects with valid data for both models."""
        valid_subjects = []
        
        for _, row in self.integrated_data.iterrows():
            naccid = row.get('NACCID')
            if naccid is None:
                continue
            
            # Check for CSF biomarkers
            has_biomarker = any([
                self._is_valid(row.get('ABETA_harm')),
                self._is_valid(row.get('PTAU_harm')),
                self._is_valid(row.get('TTAU_harm'))
            ])
            
            if not has_biomarker:
                continue
            
            # Check for clinical trajectory
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) >= 2:
                valid_subjects.append(naccid)
        
        return valid_subjects
    
    def _extract_baseline_features(self) -> Tuple[np.ndarray, List[str]]:
        """Extract baseline feature vector for each subject."""
        feature_names = [
            'ABETA_harm', 'PTAU_harm', 'TTAU_harm',
            'PTAU_ABETA_ratio', 'TTAU_PTAU_ratio',
            'AGE_AT_BASELINE', 'SEX', 'EDUC',
            'baseline_MMSE', 'baseline_CDRSUM'
        ]
        
        features_list = []
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            # CSF biomarkers
            abeta = self._clean_value(row.get('ABETA_harm'), 500.0)
            ptau = self._clean_value(row.get('PTAU_harm'), 50.0)
            ttau = self._clean_value(row.get('TTAU_harm'), 300.0)
            
            # Derived ratios
            ptau_abeta_ratio = ptau / abeta if abeta > 0 else 0.1
            ttau_ptau_ratio = ttau / ptau if ptau > 0 else 6.0
            
            # Demographics
            age = self._clean_value(row.get('AGE_AT_BASELINE'), 75.0)
            sex = self._clean_value(row.get('SEX'), 1.0)
            educ = self._clean_value(row.get('EDUC'), 16.0)
            
            # Baseline cognition
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) > 0:
                first_visit = trajectory[0]
                baseline_mmse = self._clean_value(first_visit.get('NACCMMSE'), 28.0)
                baseline_cdr = self._clean_value(first_visit.get('CDRSUM'), 0.5)
            else:
                baseline_mmse = 28.0
                baseline_cdr = 0.5
            
            features = [
                abeta, ptau, ttau,
                ptau_abeta_ratio, ttau_ptau_ratio,
                age, sex, educ,
                baseline_mmse, baseline_cdr
            ]
            
            features_list.append(features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # Handle NaN with column medians
        for col in range(features_array.shape[1]):
            col_data = features_array[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                features_array[mask, col] = median_val if not np.isnan(median_val) else 0.0
        
        return features_array, feature_names
    
    def _compute_trajectory_parameters(self) -> np.ndarray:
        """Compute trajectory parameters using quadratic regression."""
        trajectory_params = []
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            trajectory = row.get('clinical_trajectory', [])
            
            if not isinstance(trajectory, list) or len(trajectory) < 3:
                trajectory_params.append([np.nan, np.nan, np.nan])
                continue
            
            times = []
            cdr_values = []
            
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
                    alpha = coeffs[2]  # Intercept
                    beta = coeffs[1]   # Slope
                    gamma = coeffs[0]  # Acceleration
                    
                    if abs(alpha) < 20 and abs(beta) < 5 and abs(gamma) < 1:
                        trajectory_params.append([alpha, beta, gamma])
                    else:
                        trajectory_params.append([np.nan, np.nan, np.nan])
                except:
                    trajectory_params.append([np.nan, np.nan, np.nan])
            else:
                trajectory_params.append([np.nan, np.nan, np.nan])
        
        params_array = np.array(trajectory_params, dtype=np.float32)
        
        # Impute missing with column medians
        for col in range(params_array.shape[1]):
            col_data = params_array[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                params_array[mask, col] = median_val if not np.isnan(median_val) else 0.0
        
        return params_array
    
    def _extract_survival_data(self) -> Dict[str, np.ndarray]:
        """Extract survival data (time-to-event)."""
        times = []
        events = []
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            trajectory = row.get('clinical_trajectory', [])
            
            # Default values
            time_to_event = 5.0
            event_occurred = 0
            
            if isinstance(trajectory, list) and len(trajectory) > 0:
                # Find conversion event
                for visit in trajectory:
                    t = visit.get('YearsFromBaseline', 0)
                    dx = visit.get('NACCUDSD', 0)
                    
                    if dx == 4:  # Dementia diagnosis
                        time_to_event = max(0.1, float(t))
                        event_occurred = 1
                        break
                    
                    time_to_event = max(time_to_event, float(t) if self._is_valid(t) else 0)
            
            times.append(time_to_event)
            events.append(event_occurred)
        
        return {
            'times': np.array(times, dtype=np.float32),
            'events': np.array(events, dtype=np.int32)
        }
    
    def _extract_center_ids(self) -> np.ndarray:
        """Extract ADC center IDs for each subject."""
        center_ids = []
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            # NACCADC is the ADC identifier
            adc = row.get('NACCADC', 0)
            if pd.isna(adc) or adc in self.MISSING_CODES:
                adc = 0
            
            center_ids.append(int(adc))
        
        return np.array(center_ids, dtype=np.int32)
    
    def _create_tensors(self):
        """Convert arrays to tensors."""
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
            'center': self.centers[idx],
            'subject_id': self.subjects[idx]
        }
    
    def get_center_indices(self, center_id: int) -> np.ndarray:
        """Get indices for subjects from a specific center."""
        return np.where(self.center_ids == center_id)[0]
    
    def get_unique_centers(self) -> np.ndarray:
        """Get list of unique center IDs."""
        return np.unique(self.center_ids)
    
    def get_center_sizes(self) -> Dict[int, int]:
        """Get number of subjects per center."""
        unique, counts = np.unique(self.center_ids, return_counts=True)
        return dict(zip(unique, counts))


# =============================================================================
# CROSS-CENTER VALIDATION
# =============================================================================

class CrossCenterValidator:
    """
    Implements cross-center validation for PROGRESS framework.
    
    Validation strategies:
    1. Leave-One-Center-Out (LOCO): Train on N-1 centers, test on held-out center
    2. Center size stratification: Compare performance on small/medium/large centers
    3. Aggregated metrics: Overall generalizability assessment
    """
    
    def __init__(self, config: 'PROGRESSConfig', output_dir: str):
        """
        Initialize cross-center validator.
        
        Args:
            config: PROGRESS configuration
            output_dir: Directory for saving results
        """
        self.config = config
        self.output_dir = output_dir
        self.device = config.get_device()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Results storage
        self.loco_results = {}
        self.stratified_results = {}
        self.summary_metrics = {}
        
        logger.info(f"CrossCenterValidator initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output: {output_dir}")
    
    def run_loco_validation(self,
                           dataset: CrossCenterDataset,
                           min_center_size: int = 20,
                           max_centers: int = None) -> Dict[str, Any]:
        """
        Run Leave-One-Center-Out validation.
        
        Args:
            dataset: CrossCenterDataset with center information
            min_center_size: Minimum subjects per center to include
            max_centers: Maximum number of centers to evaluate (for testing)
            
        Returns:
            Dictionary with results for each center
        """
        logger.info("=" * 70)
        logger.info("LEAVE-ONE-CENTER-OUT VALIDATION")
        logger.info("=" * 70)
        
        # Get centers meeting size threshold (exclude center -1 which means missing)
        center_sizes = dataset.get_center_sizes()
        valid_centers = [c for c, size in center_sizes.items() 
                        if size >= min_center_size and c != -1 and c != 0]
        
        if len(valid_centers) < 2:
            logger.error(f"Need at least 2 valid centers for LOCO validation, found {len(valid_centers)}")
            logger.error("This usually means NACCADC was not properly extracted from CSF data.")
            logger.error("Please check that --csf-file points to the correct file.")
            raise ValueError(f"Insufficient centers for LOCO validation: {len(valid_centers)}")
        
        if max_centers is not None:
            valid_centers = valid_centers[:max_centers]
        
        logger.info(f"Centers with >= {min_center_size} subjects: {len(valid_centers)}")
        logger.info(f"Total subjects in valid centers: {sum(center_sizes[c] for c in valid_centers)}")
        logger.info(f"Center IDs: {valid_centers[:10]}{'...' if len(valid_centers) > 10 else ''}")
        
        results = {}
        
        for i, held_out_center in enumerate(tqdm(valid_centers, desc="LOCO Validation")):
            logger.info(f"\n--- Fold {i+1}/{len(valid_centers)}: Holding out Center {held_out_center} ---")
            logger.info(f"    Test set size: {center_sizes[held_out_center]}")
            
            # Get train/test indices
            test_indices = dataset.get_center_indices(held_out_center)
            train_indices = np.array([
                idx for idx in range(len(dataset))
                if dataset.center_ids[idx] != held_out_center 
                and dataset.center_ids[idx] != -1  # Exclude unknown centers
                and dataset.center_ids[idx] != 0   # Exclude default center
            ])
            
            # Safety check
            if len(train_indices) < 50:
                logger.warning(f"    Skipping center {held_out_center}: insufficient training data ({len(train_indices)})")
                continue
            
            if len(test_indices) < 5:
                logger.warning(f"    Skipping center {held_out_center}: insufficient test data ({len(test_indices)})")
                continue
            
            # Further split training into train/val
            train_idx, val_idx = train_test_split(
                train_indices, test_size=0.15, random_state=42
            )
            
            logger.info(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_indices)}")
            
            # Create data loaders
            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=self.config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=self.config.batch_size,
                shuffle=False
            )
            test_loader = DataLoader(
                Subset(dataset, test_indices),
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            # Train and evaluate
            fold_results = self._train_and_evaluate_fold(
                train_loader, val_loader, test_loader,
                input_dim=dataset.X.shape[1]
            )
            
            fold_results['center_id'] = int(held_out_center)
            fold_results['test_size'] = len(test_indices)
            fold_results['train_size'] = len(train_idx)
            
            results[held_out_center] = fold_results
            
            logger.info(f"    Trajectory RÂ² (Î±): {fold_results['traj_intercept_R2']:.4f}")
            logger.info(f"    Survival C-index: {fold_results['surv_c_index']:.4f}")
        
        if not results:
            raise ValueError("No centers could be evaluated. Check data quality and center assignments.")
        
        self.loco_results = results
        return results
    
    def _train_and_evaluate_fold(self,
                                 train_loader: DataLoader,
                                 val_loader: DataLoader,
                                 test_loader: DataLoader,
                                 input_dim: int) -> Dict[str, float]:
        """
        Train models and evaluate on test set for one fold.
        
        Returns:
            Dictionary with evaluation metrics
        """
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
            model_type='trajectory',
            num_epochs=min(50, self.config.num_epochs)  # Reduced epochs for LOCO
        )
        
        # Train survival model
        surv_model = self._train_model(
            surv_model, train_loader, val_loader,
            model_type='survival',
            num_epochs=min(50, self.config.num_epochs)
        )
        
        # Evaluate on test set
        results = self._evaluate_models(traj_model, surv_model, test_loader)
        
        return results
    
    def _train_model(self,
                     model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_type: str,
                     num_epochs: int) -> nn.Module:
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
                else:  # survival
                    times = batch['time'].to(self.device)
                    events = batch['event'].to(self.device)
                    risk_scores = model(features).view(-1)
                    
                    # Cox partial likelihood
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
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= 10:  # Reduced patience for LOCO
                break
        
        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return model
    
    def _evaluate_models(self,
                        traj_model: nn.Module,
                        surv_model: nn.Module,
                        test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate both models on test set."""
        traj_model.eval()
        surv_model.eval()
        
        # Collect predictions
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
                traj_model.train()  # Enable dropout
                traj_preds = []
                for _ in range(20):  # MC samples
                    output = traj_model(features)
                    traj_preds.append(output['mean'].cpu().numpy())
                traj_model.eval()
                
                traj_pred = np.mean(traj_preds, axis=0)
                traj_std = np.std(traj_preds, axis=0)
                
                # Survival predictions
                risk_scores = surv_model(features).cpu().numpy().squeeze()
                
                all_traj_true.append(batch['trajectory_params'].numpy())
                all_traj_pred.append(traj_pred)
                all_traj_std.append(traj_std)
                all_risk_scores.append(risk_scores)
                all_times.append(batch['time'].numpy())
                all_events.append(batch['event'].numpy())
        
        # Concatenate
        traj_true = np.vstack(all_traj_true)
        traj_pred = np.vstack(all_traj_pred)
        traj_std = np.vstack(all_traj_std)
        risk_scores = np.concatenate(all_risk_scores)
        times = np.concatenate(all_times)
        events = np.concatenate(all_events)
        
        # Compute trajectory metrics
        results = {}
        param_names = ['intercept', 'slope', 'acceleration']
        
        for i, name in enumerate(param_names):
            true_i = traj_true[:, i]
            pred_i = traj_pred[:, i]
            std_i = traj_std[:, i]
            
            results[f'traj_{name}_R2'] = r2_score(true_i, pred_i) if np.var(true_i) > 0 else 0
            results[f'traj_{name}_RMSE'] = np.sqrt(mean_squared_error(true_i, pred_i))
            results[f'traj_{name}_MAE'] = mean_absolute_error(true_i, pred_i)
            
            if np.std(true_i) > 0 and np.std(pred_i) > 0:
                corr, p = stats.pearsonr(true_i, pred_i)
                results[f'traj_{name}_corr'] = corr
            else:
                results[f'traj_{name}_corr'] = 0
            
            # PICP
            lower = pred_i - 1.96 * std_i
            upper = pred_i + 1.96 * std_i
            results[f'traj_{name}_PICP'] = np.mean((true_i >= lower) & (true_i <= upper))
        
        # Compute survival metrics
        results['surv_c_index'] = self._compute_c_index(risk_scores, times, events)
        
        for horizon in [2.0, 3.0, 5.0]:
            results[f'surv_auc_{int(horizon)}yr'] = self._compute_td_auc(
                risk_scores, times, events, horizon
            )
        
        # Event rate
        results['event_rate'] = events.mean()
        
        return results
    
    def _compute_c_index(self,
                        risk_scores: np.ndarray,
                        times: np.ndarray,
                        events: np.ndarray) -> float:
        """Compute Harrell's C-index."""
        n = len(times)
        concordant = 0
        discordant = 0
        tied = 0
        
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
    
    def _compute_td_auc(self,
                       risk_scores: np.ndarray,
                       times: np.ndarray,
                       events: np.ndarray,
                       horizon: float) -> float:
        """Compute time-dependent AUC."""
        cases = (times <= horizon) & (events == 1)
        controls = times > horizon
        
        n_cases = cases.sum()
        n_controls = controls.sum()
        
        if n_cases == 0 or n_controls == 0:
            return 0.5
        
        case_risks = risk_scores[cases]
        control_risks = risk_scores[controls]
        
        concordant = 0
        for cr in case_risks:
            concordant += (cr > control_risks).sum()
            concordant += 0.5 * (cr == control_risks).sum()
        
        return concordant / (n_cases * n_controls)
    
    def run_stratified_analysis(self,
                                dataset: CrossCenterDataset) -> Dict[str, Any]:
        """
        Analyze performance stratified by center size.
        
        Categorizes centers as:
        - Small: < 30 subjects
        - Medium: 30-100 subjects
        - Large: > 100 subjects
        """
        logger.info("\n" + "=" * 70)
        logger.info("CENTER SIZE STRATIFICATION ANALYSIS")
        logger.info("=" * 70)
        
        center_sizes = dataset.get_center_sizes()
        
        # Categorize centers
        small_centers = [c for c, s in center_sizes.items() if s < 30]
        medium_centers = [c for c, s in center_sizes.items() if 30 <= s <= 100]
        large_centers = [c for c, s in center_sizes.items() if s > 100]
        
        logger.info(f"Small centers (<30): {len(small_centers)}")
        logger.info(f"Medium centers (30-100): {len(medium_centers)}")
        logger.info(f"Large centers (>100): {len(large_centers)}")
        
        results = {
            'small': {'centers': small_centers, 'metrics': {}},
            'medium': {'centers': medium_centers, 'metrics': {}},
            'large': {'centers': large_centers, 'metrics': {}}
        }
        
        # Aggregate LOCO results by size category
        if self.loco_results:
            for category, info in results.items():
                category_results = [
                    self.loco_results[c] for c in info['centers']
                    if c in self.loco_results
                ]
                
                if category_results:
                    # Aggregate metrics
                    metrics = {}
                    for key in ['traj_intercept_R2', 'traj_slope_R2', 'surv_c_index',
                               'surv_auc_2yr', 'surv_auc_3yr', 'surv_auc_5yr']:
                        values = [r.get(key, np.nan) for r in category_results]
                        values = [v for v in values if not np.isnan(v)]
                        if values:
                            metrics[f'{key}_mean'] = np.mean(values)
                            metrics[f'{key}_std'] = np.std(values)
                            metrics[f'{key}_min'] = np.min(values)
                            metrics[f'{key}_max'] = np.max(values)
                    
                    info['metrics'] = metrics
                    info['n_centers_evaluated'] = len(category_results)
        
        self.stratified_results = results
        return results
    
    def compute_summary_metrics(self) -> Dict[str, Any]:
        """Compute overall summary metrics across all centers."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPUTING SUMMARY METRICS")
        logger.info("=" * 70)
        
        if not self.loco_results:
            logger.warning("No LOCO results available")
            return {}
        
        # Aggregate across all centers
        all_results = list(self.loco_results.values())
        
        summary = {
            'n_centers': len(all_results),
            'total_subjects': sum(r['test_size'] for r in all_results)
        }
        
        # Key metrics aggregation
        key_metrics = [
            'traj_intercept_R2', 'traj_intercept_PICP',
            'traj_slope_R2', 'traj_slope_PICP',
            'surv_c_index', 'surv_auc_3yr'
        ]
        
        for metric in key_metrics:
            values = [r.get(metric, np.nan) for r in all_results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_median'] = np.median(values)
                summary[f'{metric}_iqr'] = stats.iqr(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        # Generalizability index (coefficient of variation)
        for metric in ['traj_intercept_R2', 'surv_c_index']:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in summary and std_key in summary and summary[mean_key] > 0:
                summary[f'{metric}_cv'] = summary[std_key] / summary[mean_key]
        
        self.summary_metrics = summary
        
        # Log summary
        logger.info(f"\nOverall Performance (n={summary['n_centers']} centers):")
        logger.info(f"  Trajectory Î± RÂ²: {summary.get('traj_intercept_R2_mean', 0):.3f} "
                   f"Â± {summary.get('traj_intercept_R2_std', 0):.3f}")
        logger.info(f"  Trajectory Î± PICP: {summary.get('traj_intercept_PICP_mean', 0):.3f} "
                   f"Â± {summary.get('traj_intercept_PICP_std', 0):.3f}")
        logger.info(f"  Survival C-index: {summary.get('surv_c_index_mean', 0):.3f} "
                   f"Â± {summary.get('surv_c_index_std', 0):.3f}")
        logger.info(f"  Survival AUC(3yr): {summary.get('surv_auc_3yr_mean', 0):.3f} "
                   f"Â± {summary.get('surv_auc_3yr_std', 0):.3f}")
        
        return summary
    
    def generate_plots(self):
        """Generate visualization plots for cross-center analysis."""
        plot_dir = os.path.join(self.output_dir, 'plots')
        
        if not self.loco_results:
            logger.warning("No results to plot")
            return
        
        # 1. C-index by center (bar plot)
        fig, ax = plt.subplots(figsize=(14, 6))
        centers = list(self.loco_results.keys())
        c_indices = [self.loco_results[c]['surv_c_index'] for c in centers]
        sizes = [self.loco_results[c]['test_size'] for c in centers]
        
        # Sort by C-index
        sorted_idx = np.argsort(c_indices)
        centers_sorted = [centers[i] for i in sorted_idx]
        c_indices_sorted = [c_indices[i] for i in sorted_idx]
        sizes_sorted = [sizes[i] for i in sorted_idx]
        
        colors = plt.cm.viridis(np.array(sizes_sorted) / max(sizes_sorted))
        bars = ax.bar(range(len(centers_sorted)), c_indices_sorted, color=colors)
        
        ax.axhline(y=0.5, color='r', linestyle='--', label='Random (C=0.5)')
        ax.axhline(y=np.mean(c_indices), color='g', linestyle='-', 
                   label=f'Mean (C={np.mean(c_indices):.3f})')
        
        ax.set_xlabel('Center (sorted by C-index)')
        ax.set_ylabel('C-index')
        ax.set_title('Survival Model Performance Across Centers (LOCO Validation)')
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        
        # Add colorbar for sample size
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                    norm=plt.Normalize(min(sizes), max(sizes)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Test Set Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'c_index_by_center.png'), dpi=150)
        plt.close()
        
        # 2. Performance vs. center size scatter
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # C-index vs size
        ax = axes[0]
        ax.scatter(sizes, c_indices, alpha=0.7, s=100)
        z = np.polyfit(sizes, c_indices, 1)
        p = np.poly1d(z)
        ax.plot(sorted(sizes), p(sorted(sizes)), 'r--', alpha=0.5)
        
        corr, pval = stats.pearsonr(sizes, c_indices)
        ax.set_xlabel('Center Size (n subjects)')
        ax.set_ylabel('C-index')
        ax.set_title(f'C-index vs Center Size\nr={corr:.3f}, p={pval:.3f}')
        
        # RÂ² vs size
        ax = axes[1]
        r2_values = [self.loco_results[c]['traj_intercept_R2'] for c in centers]
        ax.scatter(sizes, r2_values, alpha=0.7, s=100, color='orange')
        z = np.polyfit(sizes, r2_values, 1)
        p = np.poly1d(z)
        ax.plot(sorted(sizes), p(sorted(sizes)), 'r--', alpha=0.5)
        
        corr, pval = stats.pearsonr(sizes, r2_values)
        ax.set_xlabel('Center Size (n subjects)')
        ax.set_ylabel('RÂ² (Intercept)')
        ax.set_title(f'Trajectory RÂ² vs Center Size\nr={corr:.3f}, p={pval:.3f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'performance_vs_size.png'), dpi=150)
        plt.close()
        
        # 3. Box plots by center size category
        if self.stratified_results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Prepare data
            categories = ['small', 'medium', 'large']
            c_index_data = []
            r2_data = []
            
            for cat in categories:
                cat_centers = self.stratified_results[cat]['centers']
                c_vals = [self.loco_results[c]['surv_c_index'] 
                         for c in cat_centers if c in self.loco_results]
                r2_vals = [self.loco_results[c]['traj_intercept_R2'] 
                          for c in cat_centers if c in self.loco_results]
                c_index_data.append(c_vals if c_vals else [0.5])
                r2_data.append(r2_vals if r2_vals else [0])
            
            # C-index box plot
            ax = axes[0]
            bp = ax.boxplot(c_index_data, labels=['Small\n(<30)', 'Medium\n(30-100)', 'Large\n(>100)'])
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel('C-index')
            ax.set_title('Survival Performance by Center Size')
            
            # RÂ² box plot
            ax = axes[1]
            bp = ax.boxplot(r2_data, labels=['Small\n(<30)', 'Medium\n(30-100)', 'Large\n(>100)'])
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel('RÂ² (Intercept)')
            ax.set_title('Trajectory Performance by Center Size')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'performance_by_size_category.png'), dpi=150)
            plt.close()
        
        # 4. Distribution of metrics
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        metrics_to_plot = [
            ('surv_c_index', 'C-index'),
            ('surv_auc_3yr', 'AUC (3-year)'),
            ('traj_intercept_R2', 'RÂ² (Intercept)'),
            ('traj_intercept_PICP', 'PICP (Intercept)')
        ]
        
        for ax, (metric, label) in zip(axes.flatten(), metrics_to_plot):
            values = [self.loco_results[c].get(metric, np.nan) for c in centers]
            values = [v for v in values if not np.isnan(v)]
            
            ax.hist(values, bins=15, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(values), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(values):.3f}')
            ax.axvline(np.median(values), color='g', linestyle='-', 
                      label=f'Median: {np.median(values):.3f}')
            ax.set_xlabel(label)
            ax.set_ylabel('Number of Centers')
            ax.legend()
        
        plt.suptitle('Distribution of Performance Metrics Across Centers')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'metric_distributions.png'), dpi=150)
        plt.close()
        
        logger.info(f"Plots saved to {plot_dir}")
    
    def save_results(self):
        """Save all results to files."""
        # Save LOCO results
        loco_file = os.path.join(self.output_dir, 'loco_results.json')
        with open(loco_file, 'w') as f:
            # Convert numpy types to Python types
            loco_serializable = {}
            for center, results in self.loco_results.items():
                loco_serializable[int(center)] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in results.items()
                }
            json.dump(loco_serializable, f, indent=2)
        
        # Save summary metrics
        summary_file = os.path.join(self.output_dir, 'summary_metrics.json')
        with open(summary_file, 'w') as f:
            summary_serializable = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in self.summary_metrics.items()
            }
            json.dump(summary_serializable, f, indent=2)
        
        # Save stratified results
        strat_file = os.path.join(self.output_dir, 'stratified_results.json')
        with open(strat_file, 'w') as f:
            strat_serializable = {}
            for cat, info in self.stratified_results.items():
                strat_serializable[cat] = {
                    'centers': [int(c) for c in info['centers']],
                    'n_centers_evaluated': info.get('n_centers_evaluated', 0),
                    'metrics': {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in info.get('metrics', {}).items()
                    }
                }
            json.dump(strat_serializable, f, indent=2)
        
        # Save CSV summary
        if self.loco_results:
            rows = []
            for center, results in self.loco_results.items():
                row = {'center_id': center}
                row.update(results)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(self.output_dir, 'cross_center_summary.csv'), index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        if not self.summary_metrics:
            return "% No summary metrics available"
        
        latex = r"""
\begin{table}[H]
\centering
\caption{Cross-center generalizability of PROGRESS framework using Leave-One-Center-Out validation across """ + str(self.summary_metrics.get('n_centers', 0)) + r""" ADRCs.}
\label{tab:cross_center}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{SD} & \textbf{Min} & \textbf{Max} \\
\midrule
\multicolumn{5}{l}{\textit{Trajectory Prediction}} \\
Intercept $R^2$ & """ + f"{self.summary_metrics.get('traj_intercept_R2_mean', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_intercept_R2_std', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_intercept_R2_min', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_intercept_R2_max', 0):.3f}" + r""" \\
Intercept PICP & """ + f"{self.summary_metrics.get('traj_intercept_PICP_mean', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_intercept_PICP_std', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_intercept_PICP_min', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_intercept_PICP_max', 0):.3f}" + r""" \\
Slope $R^2$ & """ + f"{self.summary_metrics.get('traj_slope_R2_mean', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_slope_R2_std', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_slope_R2_min', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('traj_slope_R2_max', 0):.3f}" + r""" \\
\midrule
\multicolumn{5}{l}{\textit{Survival Prediction}} \\
C-index & """ + f"{self.summary_metrics.get('surv_c_index_mean', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('surv_c_index_std', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('surv_c_index_min', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('surv_c_index_max', 0):.3f}" + r""" \\
AUC (3-year) & """ + f"{self.summary_metrics.get('surv_auc_3yr_mean', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('surv_auc_3yr_std', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('surv_auc_3yr_min', 0):.3f}" + r""" & """ + f"{self.summary_metrics.get('surv_auc_3yr_max', 0):.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        return latex


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_cross_center_validation(data_dir: str,
                                output_dir: str = None,
                                csf_file: str = None,
                                min_center_size: int = 20,
                                max_centers: int = None,
                                quick_test: bool = False) -> Dict[str, Any]:
    """
    Run complete cross-center validation pipeline.
    
    Args:
        data_dir: Directory containing integrated dataset
        output_dir: Output directory for results
        csf_file: Path to CSF biomarker file (for NACCADC extraction)
        min_center_size: Minimum subjects per center
        max_centers: Maximum centers to evaluate (None for all)
        quick_test: If True, use reduced settings for testing
        
    Returns:
        Dictionary with all results
    """
    # Setup
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'cross_center_results')
    
    logger = setup_cross_center_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("CROSS-CENTER VALIDATION FOR PROGRESS FRAMEWORK")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CSF file: {csf_file}")
    logger.info(f"Min center size: {min_center_size}")
    logger.info(f"Quick test mode: {quick_test}")
    
    # Load data
    logger.info("\nLoading integrated dataset...")
    integrated_file = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    
    if not os.path.exists(integrated_file):
        raise FileNotFoundError(f"Integrated dataset not found: {integrated_file}")
    
    integrated_data = pd.read_pickle(integrated_file)
    logger.info(f"Loaded {len(integrated_data)} subjects")
    
    # Check for NACCADC column
    if 'NACCADC' not in integrated_data.columns:
        logger.warning("NACCADC column not found. Attempting to extract from CSF data...")
        
        # Try multiple possible locations for CSF file
        possible_csf_paths = [
            csf_file,  # User-provided path
            os.path.join(data_dir, 'investigator_fcsf_nacc69.csv'),
            'investigator_fcsf_nacc69.csv',  # Current directory
            os.path.join(os.path.dirname(data_dir), 'investigator_fcsf_nacc69.csv'),
        ]
        
        csf_data = None
        for path in possible_csf_paths:
            if path and os.path.exists(path):
                logger.info(f"Found CSF file at: {path}")
                csf_data = pd.read_csv(path)
                break
        
        if csf_data is not None and 'NACCADC' in csf_data.columns:
            # Get unique NACCID -> NACCADC mapping
            adc_mapping = csf_data[['NACCID', 'NACCADC']].drop_duplicates()
            logger.info(f"CSF data has {len(adc_mapping)} unique subjects with NACCADC")
            logger.info(f"Unique centers in CSF: {adc_mapping['NACCADC'].nunique()}")
            
            # Merge with integrated data
            n_before = len(integrated_data)
            integrated_data = integrated_data.merge(
                adc_mapping, on='NACCID', how='left'
            )
            
            # Check how many got NACCADC
            n_with_adc = integrated_data['NACCADC'].notna().sum()
            logger.info(f"Merged NACCADC: {n_with_adc}/{n_before} subjects have center info")
            
            # Fill missing with -1 (will be filtered out)
            integrated_data['NACCADC'] = integrated_data['NACCADC'].fillna(-1).astype(int)
        else:
            logger.error("Could not find CSF file with NACCADC column!")
            logger.error("Please provide the CSF file path using --csf-file argument")
            logger.error("Example: python cross_center_validation.py --data-dir ./dataset --csf-file ./investigator_fcsf_nacc69.csv")
            raise FileNotFoundError(
                "NACCADC column not found and CSF file not available. "
                "Please provide CSF file path with --csf-file argument."
            )
    
    # Configuration
    config = PROGRESSConfig(
        num_epochs=20 if quick_test else 50,
        batch_size=32,
        patience=10 if quick_test else 15
    )
    
    # Create dataset
    logger.info("\nCreating cross-center dataset...")
    dataset = CrossCenterDataset(
        integrated_data=integrated_data,
        fit_scaler=True,
        config=config
    )
    
    # Log center distribution
    center_sizes = dataset.get_center_sizes()
    logger.info(f"\nCenter distribution:")
    logger.info(f"  Total centers: {len(center_sizes)}")
    logger.info(f"  Size range: {min(center_sizes.values())} - {max(center_sizes.values())}")
    logger.info(f"  Median size: {np.median(list(center_sizes.values())):.0f}")
    
    # Initialize validator
    validator = CrossCenterValidator(config, output_dir)
    
    # Run LOCO validation
    if quick_test:
        max_centers = min(5, max_centers) if max_centers else 5
    
    loco_results = validator.run_loco_validation(
        dataset,
        min_center_size=min_center_size,
        max_centers=max_centers
    )
    
    # Run stratified analysis
    stratified_results = validator.run_stratified_analysis(dataset)
    
    # Compute summary
    summary = validator.compute_summary_metrics()
    
    # Generate plots
    validator.generate_plots()
    
    # Save results
    validator.save_results()
    
    # Generate LaTeX table
    latex_table = validator.generate_latex_table()
    latex_file = os.path.join(output_dir, 'cross_center_table.tex')
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    logger.info(f"LaTeX table saved to {latex_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-CENTER VALIDATION COMPLETED")
    logger.info("=" * 70)
    
    return {
        'loco_results': loco_results,
        'stratified_results': stratified_results,
        'summary': summary
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cross-Center Validation for PROGRESS Framework'
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
        help='Output directory (default: {data-dir}/cross_center_results)'
    )
    parser.add_argument(
        '--csf-file',
        type=str,
        default=None,
        help='Path to CSF biomarker file (investigator_fcsf_nacc69.csv) for NACCADC extraction'
    )
    parser.add_argument(
        '--min-center-size',
        type=int,
        default=20,
        help='Minimum subjects per center (default: 20)'
    )
    parser.add_argument(
        '--max-centers',
        type=int,
        default=None,
        help='Maximum centers to evaluate (default: all)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with reduced settings'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_cross_center_validation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            csf_file=args.csf_file,
            min_center_size=args.min_center_size,
            max_centers=args.max_centers,
            quick_test=args.quick_test
        )
        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())