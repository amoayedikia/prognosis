#!/usr/bin/env python3
"""
complete_baseline_comparison.py - Comprehensive Baseline Comparison for PROGRESS

This script provides a complete and fair comparison of PROGRESS against all relevant
baseline methods for both trajectory prediction and survival analysis.

Key improvements over previous version:
1. Added missing baselines (Lasso, ElasticNet)
2. Fixed time-dependent AUC calculation (proper handling of censoring)
3. Added all metrics matching paper tables (RMSE, MAE, correlation, PICP, MPIW)
4. Added uncertainty comparison for Bayesian Ridge
5. Added classification baselines for progressor/conversion prediction

Usage:
    python complete_baseline_comparison.py --data-dir ./dataset
    python complete_baseline_comparison.py --data-dir ./dataset --quick-test
    python complete_baseline_comparison.py --data-dir ./dataset --epochs 150

Output:
    - complete_comparison_results.json: Full results
    - trajectory_comparison_table.csv: Table 2 for paper
    - survival_comparison_table.csv: Table 3 for paper
    - classification_comparison_table.csv: Classification metrics
    - comparison_plots/: Visualization directory

Author: PROGRESS Paper
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import os
import sys
import json
import pickle
from datetime import datetime
from scipy import stats
from scipy.special import ndtr  # Normal CDF
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
    ElasticNet, ElasticNetCV, BayesianRidge, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve

# Optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Info: XGBoost not installed. Install with: pip install xgboost")

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as lifelines_cindex
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Info: lifelines not installed. Install with: pip install lifelines")

try:
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("Info: scikit-survival not installed. Install with: pip install scikit-survival")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ComparisonConfig:
    """Configuration for baseline comparison experiments."""
    
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    
    # Data split
    test_size: float = 0.2
    val_size: float = 0.15
    random_seed: int = 42
    
    # Classification thresholds
    slope_threshold: float = 0.5  # CDR-SB points/year for fast vs slow
    
    # Survival horizons
    survival_horizons: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    
    # Uncertainty
    confidence_level: float = 0.95
    
    # Device
    device: str = 'auto'
    
    def get_device(self) -> torch.device:
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(self.device)


# =============================================================================
# DATA LOADING
# =============================================================================

MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}


def is_valid(value) -> bool:
    """Check if value is valid (not missing or NaN)."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if value in MISSING_CODES:
        return False
    return True


def clean_value(value, default: float = np.nan) -> float:
    """Clean a value by handling missing codes."""
    if not is_valid(value):
        return default
    return float(value)


def load_nacc_data(data_dir: str, config: ComparisonConfig) -> Dict[str, Any]:
    """
    Load NACC data and prepare train/val/test splits.
    
    Returns dictionary with all data needed for comparison.
    """
    logger.info("=" * 70)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 70)
    
    integrated_path = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    
    if not os.path.exists(integrated_path):
        raise FileNotFoundError(f"Dataset not found: {integrated_path}")
    
    integrated_data = pd.read_pickle(integrated_path)
    logger.info(f"Loaded: {len(integrated_data)} subjects")
    
    # Get valid subjects
    valid_subjects = []
    for _, row in integrated_data.iterrows():
        naccid = row.get('NACCID')
        if naccid is None:
            continue
        
        has_biomarker = any([
            is_valid(row.get('ABETA_harm')),
            is_valid(row.get('PTAU_harm')),
            is_valid(row.get('TTAU_harm'))
        ])
        
        if not has_biomarker:
            continue
        
        trajectory = row.get('clinical_trajectory', [])
        if isinstance(trajectory, list) and len(trajectory) >= 2:
            valid_subjects.append(naccid)
    
    logger.info(f"Valid subjects: {len(valid_subjects)}")
    
    # Extract features and targets
    feature_names = [
        'ABETA_harm', 'PTAU_harm', 'TTAU_harm',
        'PTAU_ABETA_ratio', 'TTAU_PTAU_ratio',
        'AGE_AT_BASELINE', 'SEX', 'EDUC',
        'baseline_MMSE', 'baseline_CDRSUM'
    ]
    
    features_list = []
    trajectory_params_list = []
    times_list = []
    events_list = []
    
    for naccid in valid_subjects:
        row = integrated_data[integrated_data['NACCID'] == naccid].iloc[0]
        
        # === Features ===
        abeta = clean_value(row.get('ABETA_harm'), 500.0)
        ptau = clean_value(row.get('PTAU_harm'), 50.0)
        ttau = clean_value(row.get('TTAU_harm'), 300.0)
        
        ptau_abeta_ratio = ptau / abeta if abeta > 0 else 0.1
        ttau_ptau_ratio = ttau / ptau if ptau > 0 else 6.0
        
        age = clean_value(row.get('AGE_AT_BASELINE'), 75.0)
        sex = clean_value(row.get('SEX'), 1.0)
        educ = clean_value(row.get('EDUC'), 16.0)
        
        trajectory = row.get('clinical_trajectory', [])
        if isinstance(trajectory, list) and len(trajectory) > 0:
            first_visit = trajectory[0]
            baseline_mmse = clean_value(first_visit.get('NACCMMSE'), 28.0)
            baseline_cdr = clean_value(first_visit.get('CDRSUM'), 0.5)
        else:
            baseline_mmse = 28.0
            baseline_cdr = 0.5
        
        features = [abeta, ptau, ttau, ptau_abeta_ratio, ttau_ptau_ratio,
                   age, sex, educ, baseline_mmse, baseline_cdr]
        features_list.append(features)
        
        # === Trajectory Parameters (Quadratic Fit) ===
        if isinstance(trajectory, list) and len(trajectory) >= 3:
            times_traj = []
            scores_traj = []
            for visit in trajectory:
                t = visit.get('YearsFromBaseline', 0)
                score = visit.get('CDRSUM')
                if is_valid(score) and is_valid(t):
                    score = float(score)
                    if 0 <= score <= 18:
                        times_traj.append(float(t))
                        scores_traj.append(score)
            
            if len(times_traj) >= 3:
                try:
                    coeffs = np.polyfit(times_traj, scores_traj, deg=2)
                    alpha, beta, gamma = coeffs[2], coeffs[1], coeffs[0]
                    if abs(alpha) < 20 and abs(beta) < 5 and abs(gamma) < 1:
                        trajectory_params_list.append([alpha, beta, gamma])
                    else:
                        trajectory_params_list.append([np.nan, np.nan, np.nan])
                except:
                    trajectory_params_list.append([np.nan, np.nan, np.nan])
            else:
                trajectory_params_list.append([np.nan, np.nan, np.nan])
        else:
            trajectory_params_list.append([np.nan, np.nan, np.nan])
        
        # === Survival Data ===
        converted = row.get('converted_to_dementia', 0)
        if converted == 1:
            time_to_event = row.get('time_to_dementia')
            if not is_valid(time_to_event):
                time_to_event = row.get('follow_up_years', 5.0)
            times_list.append(float(time_to_event))
            events_list.append(1)
        else:
            follow_up = row.get('follow_up_years')
            if not is_valid(follow_up):
                if isinstance(trajectory, list) and len(trajectory) > 0:
                    last_visit = trajectory[-1]
                    follow_up = last_visit.get('YearsFromBaseline', 5.0)
                else:
                    follow_up = 5.0
            times_list.append(float(follow_up))
            events_list.append(0)
    
    # Convert to arrays
    X = np.array(features_list, dtype=np.float32)
    y_traj = np.array(trajectory_params_list, dtype=np.float32)
    times = np.array(times_list, dtype=np.float32)
    events = np.array(events_list, dtype=np.int32)
    
    # Impute missing values
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any():
            X[mask, col] = np.nanmedian(X[:, col])
    
    for col in range(y_traj.shape[1]):
        mask = np.isnan(y_traj[:, col])
        if mask.any():
            defaults = [1.0, 0.3, 0.02]
            median = np.nanmedian(y_traj[:, col])
            y_traj[mask, col] = median if not np.isnan(median) else defaults[col]
    
    times = np.nan_to_num(times, nan=np.nanmedian(times))
    times = np.maximum(times, 0.1)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # === CREATE FIXED TRAIN/VAL/TEST SPLIT ===
    logger.info("\nCreating data splits...")
    
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=events
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.val_size,
        random_state=config.random_seed,
        stratify=events[train_val_idx]
    )
    
    logger.info(f"Train: {len(train_idx)} samples ({100*len(train_idx)/len(X):.1f}%)")
    logger.info(f"Val: {len(val_idx)} samples ({100*len(val_idx)/len(X):.1f}%)")
    logger.info(f"Test: {len(test_idx)} samples ({100*len(test_idx)/len(X):.1f}%)")
    logger.info(f"Event rate - Train: {events[train_idx].mean()*100:.1f}%, "
               f"Val: {events[val_idx].mean()*100:.1f}%, "
               f"Test: {events[test_idx].mean()*100:.1f}%")
    
    return {
        'X': X_scaled,
        'X_raw': X,
        'y_traj': y_traj,
        'times': times,
        'events': events,
        'feature_names': feature_names,
        'scaler': scaler,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'subjects': valid_subjects
    }


# =============================================================================
# METRICS - COMPREHENSIVE IMPLEMENTATION
# =============================================================================

class ComprehensiveMetrics:
    """
    Comprehensive metrics for fair comparison with PROGRESS.
    
    Includes:
    - Regression metrics: R², RMSE, MAE, Correlation
    - Uncertainty metrics: PICP, MPIW
    - Survival metrics: C-index, Time-dependent AUC (corrected)
    - Classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
    """
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_std: np.ndarray = None,
                          confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_std: Predicted standard deviations (for uncertainty metrics)
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with R², RMSE, MAE, correlation, and optionally PICP/MPIW
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['R2'] = r2_score(y_true, y_pred) if np.var(y_true) > 0 else 0.0
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        
        # Correlation
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            corr, p_val = stats.pearsonr(y_true, y_pred)
            metrics['correlation'] = corr
            metrics['p_value'] = p_val
        else:
            metrics['correlation'] = 0.0
            metrics['p_value'] = 1.0
        
        # Uncertainty metrics (if std provided)
        if y_std is not None:
            # Z-score for confidence level
            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            
            # PICP: Prediction Interval Coverage Probability
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            metrics['PICP'] = coverage
            
            # MPIW: Mean Prediction Interval Width
            metrics['MPIW'] = np.mean(2 * z * y_std)
            
            # CRPS: Continuous Ranked Probability Score (optional)
            # Lower is better
            z_scores = (y_true - y_pred) / (y_std + 1e-8)
            crps = np.mean(y_std * (z_scores * (2 * ndtr(z_scores) - 1) + 
                                    2 * stats.norm.pdf(z_scores) - 1 / np.sqrt(np.pi)))
            metrics['CRPS'] = crps
        
        return metrics
    
    @staticmethod
    def concordance_index(risk_scores: np.ndarray, times: np.ndarray,
                         events: np.ndarray) -> float:
        """
        Compute Harrell's C-index (concordance index).
        
        C = P(risk_i > risk_j | T_i < T_j, δ_i = 1)
        
        Args:
            risk_scores: Predicted risk scores (higher = more risk)
            times: Observed times
            events: Event indicators (1 = event, 0 = censored)
            
        Returns:
            C-index value between 0 and 1
        """
        n = len(times)
        concordant = 0
        discordant = 0
        tied_risk = 0
        comparable = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if pair is comparable
                if events[i] == 1 and times[i] < times[j]:
                    # i had event before j's time
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
                        
                elif events[j] == 1 and times[j] < times[i]:
                    # j had event before i's time
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
        
        if comparable == 0:
            return 0.5
        
        return (concordant + tied_risk) / comparable
    
    @staticmethod
    def time_dependent_auc(risk_scores: np.ndarray, times: np.ndarray,
                          events: np.ndarray, horizon: float) -> float:
        """
        Compute TIME-DEPENDENT AUC at specific horizon (CORRECTED VERSION).
        
        This is the correct formulation that properly handles censoring:
        AUC(t) = P(risk_i > risk_j | T_i ≤ t, δ_i = 1, T_j > t)
        
        Cases: Subjects who experienced the event before or at horizon
        Controls: Subjects who are event-free at horizon (either censored after
                  horizon or have event after horizon)
        
        Args:
            risk_scores: Predicted risk scores
            times: Observed times
            events: Event indicators
            horizon: Time horizon for evaluation
            
        Returns:
            Time-dependent AUC
        """
        # Cases: Had event by time horizon
        cases = (times <= horizon) & (events == 1)
        
        # Controls: Event-free at horizon
        # This includes:
        # - Subjects censored after horizon
        # - Subjects with events after horizon
        controls = times > horizon
        
        n_cases = cases.sum()
        n_controls = controls.sum()
        
        if n_cases < 2 or n_controls < 2:
            return 0.5
        
        case_risks = risk_scores[cases]
        control_risks = risk_scores[controls]
        
        # AUC = proportion of concordant pairs
        concordant = 0
        for cr in case_risks:
            concordant += (cr > control_risks).sum()
            concordant += 0.5 * (cr == control_risks).sum()
        
        return concordant / (n_cases * n_controls)
    
    @staticmethod
    def integrated_brier_score(survival_probs: np.ndarray, times: np.ndarray,
                              events: np.ndarray, eval_times: np.ndarray) -> float:
        """
        Compute Integrated Brier Score for survival predictions.
        
        Args:
            survival_probs: Predicted survival probabilities (n_samples, n_times)
            times: Observed times
            events: Event indicators
            eval_times: Times at which survival_probs are evaluated
            
        Returns:
            IBS value (lower is better)
        """
        n = len(times)
        brier_scores = []
        
        # Kaplan-Meier estimate of censoring distribution G(t)
        # Simplified: assume uniform censoring for now
        for t_idx, t in enumerate(eval_times):
            if t <= 0:
                continue
                
            # True status at time t
            y_true = (times <= t) & (events == 1)
            
            # Predicted survival probability
            S_pred = survival_probs[:, t_idx] if survival_probs.ndim > 1 else survival_probs
            
            # Brier score at time t
            # Weight by inverse probability of censoring (simplified)
            bs = np.mean((y_true.astype(float) - (1 - S_pred)) ** 2)
            brier_scores.append(bs)
        
        # Integrate over time
        if len(brier_scores) > 1:
            return np.trapz(brier_scores, eval_times[:len(brier_scores)]) / eval_times[len(brier_scores)-1]
        return brier_scores[0] if brier_scores else 0.5
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary with accuracy, precision, recall, F1, specificity, AUC
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix derived
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics['ppv'] = metrics['precision']
        
        # AUC metrics
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics['auc_roc'] = 0.5
                metrics['auc_pr'] = y_true.mean()
        
        return metrics
    
    @staticmethod
    def risk_stratification_metrics(risk_scores: np.ndarray, times: np.ndarray,
                                   events: np.ndarray, n_groups: int = 3) -> Dict[str, Any]:
        """
        Compute risk stratification metrics.
        
        Args:
            risk_scores: Predicted risk scores
            times: Observed times
            events: Event indicators
            n_groups: Number of risk groups
            
        Returns:
            Dictionary with stratification metrics
        """
        # Create risk groups
        percentiles = np.linspace(0, 100, n_groups + 1)
        thresholds = np.percentile(risk_scores, percentiles[1:-1])
        risk_groups = np.digitize(risk_scores, thresholds)
        
        metrics = {
            'n_groups': n_groups,
            'group_sizes': [int((risk_groups == g).sum()) for g in range(n_groups)],
            'thresholds': thresholds.tolist()
        }
        
        # Event rate per group
        for g in range(n_groups):
            mask = risk_groups == g
            if mask.sum() > 0:
                metrics[f'group_{g}_event_rate'] = float(events[mask].mean())
                metrics[f'group_{g}_median_time'] = float(np.median(times[mask]))
        
        # Hazard ratio (high vs low)
        if n_groups >= 2:
            high_risk = risk_groups == (n_groups - 1)
            low_risk = risk_groups == 0
            
            high_events = events[high_risk].sum()
            low_events = events[low_risk].sum()
            high_person_time = times[high_risk].sum()
            low_person_time = times[low_risk].sum()
            
            high_rate = high_events / high_person_time if high_person_time > 0 else 0
            low_rate = low_events / low_person_time if low_person_time > 0 else 1e-8
            
            metrics['hazard_ratio'] = high_rate / low_rate if low_rate > 0 else np.inf
        
        return metrics


# =============================================================================
# NEURAL NETWORK BASELINES
# =============================================================================

class MLPRegressor(nn.Module):
    """Simple MLP baseline for trajectory prediction."""
    
    def __init__(self, input_dim: int, output_dim: int = 3,
                 hidden_dims: List[int] = [64, 32], dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPClassifier(nn.Module):
    """Simple MLP for classification tasks."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DeepSurvBaseline(nn.Module):
    """Simple DeepSurv baseline for survival prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# BASELINE RUNNERS
# =============================================================================

def run_trajectory_baselines(data: Dict, config: ComparisonConfig) -> Dict[str, Dict]:
    """
    Run all trajectory prediction baselines with comprehensive metrics.
    
    Includes: Linear, Ridge, Lasso, ElasticNet, RF, GB, XGBoost, SVR, 
              Bayesian Ridge (with uncertainty), MLP
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAJECTORY PREDICTION BASELINES")
    logger.info("=" * 70)
    
    X_train = data['X'][data['train_idx']]
    y_train = data['y_traj'][data['train_idx']]
    X_test = data['X'][data['test_idx']]
    y_test = data['y_traj'][data['test_idx']]
    
    param_names = ['intercept', 'slope', 'acceleration']
    results = {}
    
    def compute_metrics_for_method(y_pred, y_std=None, method_name=""):
        """Helper to compute metrics for all three parameters."""
        method_results = {}
        for i, name in enumerate(param_names):
            std_i = y_std[:, i] if y_std is not None else None
            metrics = ComprehensiveMetrics.regression_metrics(
                y_test[:, i], y_pred[:, i], std_i, config.confidence_level
            )
            for k, v in metrics.items():
                method_results[f'{name}_{k}'] = v
        return method_results
    
    # 1. Linear Regression
    logger.info("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Linear Regression'] = compute_metrics_for_method(y_pred)
    
    # 2. Ridge Regression
    logger.info("  Training Ridge Regression...")
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results['Ridge'] = compute_metrics_for_method(y_pred)
    results['Ridge']['best_alpha'] = ridge.alpha_
    
    # 3. Lasso Regression (NEW)
    logger.info("  Training Lasso Regression...")
    lasso_results = {}
    y_pred_lasso = np.zeros_like(y_test)
    for i, name in enumerate(param_names):
        lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=5000)
        lasso.fit(X_train, y_train[:, i])
        y_pred_lasso[:, i] = lasso.predict(X_test)
    results['Lasso'] = compute_metrics_for_method(y_pred_lasso)
    
    # 4. ElasticNet (NEW)
    logger.info("  Training ElasticNet...")
    y_pred_enet = np.zeros_like(y_test)
    for i, name in enumerate(param_names):
        enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95], 
                           alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=5000)
        enet.fit(X_train, y_train[:, i])
        y_pred_enet[:, i] = enet.predict(X_test)
    results['ElasticNet'] = compute_metrics_for_method(y_pred_enet)
    
    # 5. Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = compute_metrics_for_method(y_pred)
    
    # 6. Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    y_pred_gb = np.zeros_like(y_test)
    for i in range(3):
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train, y_train[:, i])
        y_pred_gb[:, i] = gb.predict(X_test)
    results['Gradient Boosting'] = compute_metrics_for_method(y_pred_gb)
    
    # 7. XGBoost
    if HAS_XGBOOST:
        logger.info("  Training XGBoost...")
        y_pred_xgb = np.zeros_like(y_test)
        for i in range(3):
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42,
                                     verbosity=0)
            model.fit(X_train, y_train[:, i])
            y_pred_xgb[:, i] = model.predict(X_test)
        results['XGBoost'] = compute_metrics_for_method(y_pred_xgb)
    
    # 8. SVR
    logger.info("  Training SVR...")
    y_pred_svr = np.zeros_like(y_test)
    for i in range(3):
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr.fit(X_train, y_train[:, i])
        y_pred_svr[:, i] = svr.predict(X_test)
    results['SVR'] = compute_metrics_for_method(y_pred_svr)
    
    # 9. Bayesian Ridge (WITH UNCERTAINTY)
    logger.info("  Training Bayesian Ridge (with uncertainty)...")
    y_pred_br = np.zeros_like(y_test)
    y_std_br = np.zeros_like(y_test)
    for i in range(3):
        br = BayesianRidge()
        br.fit(X_train, y_train[:, i])
        pred, std = br.predict(X_test, return_std=True)
        y_pred_br[:, i] = pred
        y_std_br[:, i] = std
    results['Bayesian Ridge'] = compute_metrics_for_method(y_pred_br, y_std_br)
    
    # 10. MLP Baseline
    logger.info("  Training MLP...")
    device = config.get_device()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    mlp = MLPRegressor(X_train.shape[1], 3, [64, 32], 0.3).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    mlp.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        pred = mlp(X_train_t)
        loss = nn.MSELoss()(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    mlp.eval()
    with torch.no_grad():
        y_pred_mlp = mlp(X_test_t).cpu().numpy()
    results['MLP'] = compute_metrics_for_method(y_pred_mlp)
    
    return results


def run_survival_baselines(data: Dict, config: ComparisonConfig) -> Dict[str, Dict]:
    """
    Run all survival analysis baselines with corrected metrics.
    
    Includes: Cox PH, RSF, Cox-Lasso, GB Survival, DeepSurv
    """
    logger.info("\n" + "=" * 70)
    logger.info("SURVIVAL PREDICTION BASELINES")
    logger.info("=" * 70)
    
    X_train = data['X'][data['train_idx']]
    times_train = data['times'][data['train_idx']]
    events_train = data['events'][data['train_idx']]
    
    X_test = data['X'][data['test_idx']]
    times_test = data['times'][data['test_idx']]
    events_test = data['events'][data['test_idx']]
    
    feature_names = data['feature_names']
    results = {}
    
    def compute_survival_metrics(risk_scores: np.ndarray) -> Dict[str, float]:
        """Helper to compute all survival metrics."""
        metrics = {
            'c_index': ComprehensiveMetrics.concordance_index(
                risk_scores, times_test, events_test
            )
        }
        for horizon in config.survival_horizons:
            metrics[f'auc_{horizon:.0f}yr'] = ComprehensiveMetrics.time_dependent_auc(
                risk_scores, times_test, events_test, horizon
            )
        
        # Risk stratification
        strat = ComprehensiveMetrics.risk_stratification_metrics(
            risk_scores, times_test, events_test, n_groups=3
        )
        metrics['hazard_ratio'] = strat.get('hazard_ratio', np.nan)
        
        return metrics
    
    # 1. Cox PH (lifelines)
    if HAS_LIFELINES:
        logger.info("  Training Cox PH...")
        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train['T'] = times_train
        df_train['E'] = events_train
        
        df_test = pd.DataFrame(X_test, columns=feature_names)
        
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df_train, duration_col='T', event_col='E')
            risk_scores = cph.predict_partial_hazard(df_test).values.flatten()
            results['Cox PH'] = compute_survival_metrics(risk_scores)
        except Exception as e:
            logger.warning(f"Cox PH failed: {e}")
    
    # 2-4. scikit-survival methods
    if HAS_SKSURV:
        y_train_surv = np.array([(bool(e), t) for e, t in zip(events_train, times_train)],
                               dtype=[('event', bool), ('time', float)])
        
        # 2. Random Survival Forest
        logger.info("  Training Random Survival Forest...")
        try:
            rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, 
                                       random_state=42, n_jobs=-1)
            rsf.fit(X_train, y_train_surv)
            risk_scores = rsf.predict(X_test)
            results['Random Survival Forest'] = compute_survival_metrics(risk_scores)
        except Exception as e:
            logger.warning(f"RSF failed: {e}")
        
        # 3. Cox-Lasso
        logger.info("  Training Cox-Lasso...")
        try:
            cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True)
            cox_lasso.fit(X_train, y_train_surv)
            risk_scores = cox_lasso.predict(X_test)
            results['Cox-Lasso'] = compute_survival_metrics(risk_scores)
        except Exception as e:
            logger.warning(f"Cox-Lasso failed: {e}")
        
        # 4. Gradient Boosting Survival
        logger.info("  Training Gradient Boosting Survival...")
        try:
            gbs = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=5, 
                                                    random_state=42)
            gbs.fit(X_train, y_train_surv)
            risk_scores = gbs.predict(X_test)
            results['GB Survival'] = compute_survival_metrics(risk_scores)
        except Exception as e:
            logger.warning(f"GBS failed: {e}")
    
    # 5. DeepSurv Baseline
    logger.info("  Training DeepSurv baseline...")
    device = config.get_device()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    times_train_t = torch.FloatTensor(times_train).to(device)
    events_train_t = torch.FloatTensor(events_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    deepsurv = DeepSurvBaseline(X_train.shape[1], [64, 32], 0.3).to(device)
    optimizer = optim.AdamW(deepsurv.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    deepsurv.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        risk_scores = deepsurv(X_train_t).squeeze()
        
        # Cox partial likelihood loss
        sorted_idx = torch.argsort(times_train_t, descending=True)
        sorted_risks = risk_scores[sorted_idx]
        sorted_events = events_train_t[sorted_idx]
        
        max_risk = sorted_risks.max()
        exp_risks = torch.exp(sorted_risks - max_risk)
        cumsum_exp = torch.cumsum(exp_risks, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
        
        log_lik = sorted_risks - log_cumsum
        n_events = sorted_events.sum()
        
        if n_events > 0:
            loss = -(log_lik * sorted_events).sum() / n_events
        else:
            loss = risk_scores.mean() * 0.0
        
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
    
    deepsurv.eval()
    with torch.no_grad():
        risk_scores = deepsurv(X_test_t).cpu().numpy().squeeze()
    
    results['DeepSurv (baseline)'] = compute_survival_metrics(risk_scores)
    
    return results


def run_classification_baselines(data: Dict, config: ComparisonConfig) -> Dict[str, Dict]:
    """
    Run classification baselines for:
    1. Fast vs Slow Progressor Classification (based on slope)
    2. Conversion Prediction at Different Time Horizons
    
    NEW: Addresses missing classification baselines.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CLASSIFICATION BASELINES")
    logger.info("=" * 70)
    
    X_train = data['X'][data['train_idx']]
    y_traj_train = data['y_traj'][data['train_idx']]
    times_train = data['times'][data['train_idx']]
    events_train = data['events'][data['train_idx']]
    
    X_test = data['X'][data['test_idx']]
    y_traj_test = data['y_traj'][data['test_idx']]
    times_test = data['times'][data['test_idx']]
    events_test = data['events'][data['test_idx']]
    
    results = {'progressor': {}, 'conversion': {}}
    
    # =========================================================================
    # TASK 1: FAST VS SLOW PROGRESSOR CLASSIFICATION
    # =========================================================================
    logger.info("\n  === Fast vs Slow Progressor Classification ===")
    
    # Create labels based on slope threshold
    y_prog_train = (y_traj_train[:, 1] > config.slope_threshold).astype(int)
    y_prog_test = (y_traj_test[:, 1] > config.slope_threshold).astype(int)
    
    logger.info(f"  Train: {y_prog_train.sum()}/{len(y_prog_train)} fast progressors ({100*y_prog_train.mean():.1f}%)")
    logger.info(f"  Test: {y_prog_test.sum()}/{len(y_prog_test)} fast progressors ({100*y_prog_test.mean():.1f}%)")
    
    # 1. Logistic Regression
    logger.info("    Training Logistic Regression...")
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train, y_prog_train)
    y_pred = lr_clf.predict(X_test)
    y_prob = lr_clf.predict_proba(X_test)[:, 1]
    results['progressor']['Logistic Regression'] = ComprehensiveMetrics.classification_metrics(
        y_prog_test, y_pred, y_prob
    )
    
    # 2. Random Forest Classifier
    logger.info("    Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_prog_train)
    y_pred = rf_clf.predict(X_test)
    y_prob = rf_clf.predict_proba(X_test)[:, 1]
    results['progressor']['Random Forest'] = ComprehensiveMetrics.classification_metrics(
        y_prog_test, y_pred, y_prob
    )
    
    # 3. Gradient Boosting Classifier
    logger.info("    Training Gradient Boosting Classifier...")
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb_clf.fit(X_train, y_prog_train)
    y_pred = gb_clf.predict(X_test)
    y_prob = gb_clf.predict_proba(X_test)[:, 1]
    results['progressor']['Gradient Boosting'] = ComprehensiveMetrics.classification_metrics(
        y_prog_test, y_pred, y_prob
    )
    
    # 4. XGBoost Classifier
    if HAS_XGBOOST:
        logger.info("    Training XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
        xgb_clf.fit(X_train, y_prog_train)
        y_pred = xgb_clf.predict(X_test)
        y_prob = xgb_clf.predict_proba(X_test)[:, 1]
        results['progressor']['XGBoost'] = ComprehensiveMetrics.classification_metrics(
            y_prog_test, y_pred, y_prob
        )
    
    # 5. SVM Classifier
    logger.info("    Training SVM Classifier...")
    svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
    svm_clf.fit(X_train, y_prog_train)
    y_pred = svm_clf.predict(X_test)
    y_prob = svm_clf.predict_proba(X_test)[:, 1]
    results['progressor']['SVM'] = ComprehensiveMetrics.classification_metrics(
        y_prog_test, y_pred, y_prob
    )
    
    # 6. MLP Classifier
    logger.info("    Training MLP Classifier...")
    device = config.get_device()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_prog_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    mlp_clf = MLPClassifier(X_train.shape[1], [64, 32], 0.3).to(device)
    optimizer = optim.AdamW(mlp_clf.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()
    
    mlp_clf.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        pred = mlp_clf(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    mlp_clf.eval()
    with torch.no_grad():
        y_prob = mlp_clf(X_test_t).cpu().numpy().squeeze()
    y_pred = (y_prob > 0.5).astype(int)
    results['progressor']['MLP'] = ComprehensiveMetrics.classification_metrics(
        y_prog_test, y_pred, y_prob
    )
    
    # =========================================================================
    # TASK 2: CONVERSION PREDICTION AT DIFFERENT HORIZONS
    # =========================================================================
    logger.info("\n  === Conversion Prediction ===")
    
    for horizon in config.survival_horizons:
        logger.info(f"\n    --- {horizon:.0f}-Year Conversion ---")
        
        # Create labels: converted within horizon
        y_conv_train = ((times_train <= horizon) & (events_train == 1)).astype(int)
        y_conv_test = ((times_test <= horizon) & (events_test == 1)).astype(int)
        
        # Only use subjects who were followed long enough OR had event
        valid_train = (times_train >= horizon) | (events_train == 1)
        valid_test = (times_test >= horizon) | (events_test == 1)
        
        X_train_h = X_train[valid_train]
        y_train_h = y_conv_train[valid_train]
        X_test_h = X_test[valid_test]
        y_test_h = y_conv_test[valid_test]
        
        logger.info(f"      Train: {y_train_h.sum()}/{len(y_train_h)} converters ({100*y_train_h.mean():.1f}%)")
        logger.info(f"      Test: {y_test_h.sum()}/{len(y_test_h)} converters ({100*y_test_h.mean():.1f}%)")
        
        horizon_key = f'{horizon:.0f}yr'
        results['conversion'][horizon_key] = {}
        
        if len(np.unique(y_train_h)) < 2 or len(np.unique(y_test_h)) < 2:
            logger.warning(f"      Skipping {horizon}yr - insufficient class diversity")
            continue
        
        # Logistic Regression
        lr_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr_clf.fit(X_train_h, y_train_h)
        y_pred = lr_clf.predict(X_test_h)
        y_prob = lr_clf.predict_proba(X_test_h)[:, 1]
        results['conversion'][horizon_key]['Logistic Regression'] = \
            ComprehensiveMetrics.classification_metrics(y_test_h, y_pred, y_prob)
        
        # Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                        random_state=42, class_weight='balanced', n_jobs=-1)
        rf_clf.fit(X_train_h, y_train_h)
        y_pred = rf_clf.predict(X_test_h)
        y_prob = rf_clf.predict_proba(X_test_h)[:, 1]
        results['conversion'][horizon_key]['Random Forest'] = \
            ComprehensiveMetrics.classification_metrics(y_test_h, y_pred, y_prob)
        
        # Gradient Boosting
        gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb_clf.fit(X_train_h, y_train_h)
        y_pred = gb_clf.predict(X_test_h)
        y_prob = gb_clf.predict_proba(X_test_h)[:, 1]
        results['conversion'][horizon_key]['Gradient Boosting'] = \
            ComprehensiveMetrics.classification_metrics(y_test_h, y_pred, y_prob)
        
        # XGBoost
        if HAS_XGBOOST:
            scale_pos = (len(y_train_h) - y_train_h.sum()) / max(y_train_h.sum(), 1)
            xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, 
                                        scale_pos_weight=scale_pos,
                                        random_state=42, verbosity=0)
            xgb_clf.fit(X_train_h, y_train_h)
            y_pred = xgb_clf.predict(X_test_h)
            y_prob = xgb_clf.predict_proba(X_test_h)[:, 1]
            results['conversion'][horizon_key]['XGBoost'] = \
                ComprehensiveMetrics.classification_metrics(y_test_h, y_pred, y_prob)
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_comparison_plots(traj_results: Dict, surv_results: Dict,
                             class_results: Dict, output_dir: str):
    """Generate comparison visualization plots."""
    plots_dir = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Trajectory R² Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(traj_results.keys())
    r2_intercept = [traj_results[m].get('intercept_R2', 0) for m in methods]
    r2_slope = [traj_results[m].get('slope_R2', 0) for m in methods]
    r2_accel = [traj_results[m].get('acceleration_R2', 0) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, r2_intercept, width, label='Intercept', color='steelblue')
    ax.bar(x, r2_slope, width, label='Slope', color='darkorange')
    ax.bar(x + width, r2_accel, width, label='Acceleration', color='green')
    
    ax.set_ylabel('R²')
    ax.set_title('Trajectory Parameter Prediction: R² Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'trajectory_r2_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Survival C-index Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(surv_results.keys())
    c_indices = [surv_results[m].get('c_index', 0.5) for m in methods]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(methods)))
    bars = ax.bar(methods, c_indices, color=colors)
    
    ax.set_ylabel('C-index')
    ax.set_title('Survival Prediction: C-index Comparison')
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, c_indices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'survival_cindex_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Classification F1 Comparison (Progressor)
    if 'progressor' in class_results and class_results['progressor']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(class_results['progressor'].keys())
        f1_scores = [class_results['progressor'][m].get('f1_score', 0) for m in methods]
        auc_scores = [class_results['progressor'][m].get('auc_roc', 0.5) for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='steelblue')
        ax.bar(x + width/2, auc_scores, width, label='AUC-ROC', color='darkorange')
        
        ax.set_ylabel('Score')
        ax.set_title('Progressor Classification: F1 and AUC Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'progressor_classification_comparison.png'), dpi=150)
        plt.close()
    
    # 4. Time-dependent AUC across horizons
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [2, 3, 5]
    for method in surv_results.keys():
        aucs = [surv_results[method].get(f'auc_{h}yr', 0.5) for h in horizons]
        ax.plot(horizons, aucs, 'o-', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Time Horizon (years)')
    ax.set_ylabel('Time-Dependent AUC')
    ax.set_title('Survival Prediction: AUC by Time Horizon')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'survival_auc_by_horizon.png'), dpi=150)
    plt.close()
    
    logger.info(f"Plots saved to: {plots_dir}")


# =============================================================================
# MAIN COMPARISON RUNNER
# =============================================================================

def run_complete_comparison(data_dir: str, output_dir: str = None,
                           quick_test: bool = False, epochs: int = 100) -> Dict:
    """
    Run complete baseline comparison.
    
    Args:
        data_dir: Directory containing nacc_integrated_dataset.pkl
        output_dir: Output directory for results
        quick_test: If True, use reduced epochs
        epochs: Number of training epochs
        
    Returns:
        Dictionary with all comparison results
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'complete_baseline_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    config = ComparisonConfig(
        num_epochs=30 if quick_test else epochs,
        patience=5 if quick_test else 15
    )
    
    logger.info("=" * 70)
    logger.info("COMPLETE BASELINE COMPARISON FOR PROGRESS")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Device: {config.get_device()}")
    
    # Load data
    data = load_nacc_data(data_dir, config)
    
    # Run all baselines
    traj_results = run_trajectory_baselines(data, config)
    surv_results = run_survival_baselines(data, config)
    class_results = run_classification_baselines(data, config)
    
    # =========================================================================
    # PRINT RESULTS TABLES
    # =========================================================================
    
    # Trajectory Table
    logger.info("\n" + "=" * 70)
    logger.info("TRAJECTORY PREDICTION RESULTS (Table 2)")
    logger.info("=" * 70)
    
    print("\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Method", "α R²", "β R²", "γ R²", "α RMSE", "α Corr", "PICP", "MPIW"
    ))
    print("-" * 100)
    
    for method, metrics in traj_results.items():
        picp = metrics.get('intercept_PICP', 'N/A')
        mpiw = metrics.get('intercept_MPIW', 'N/A')
        picp_str = f"{picp:.2%}" if isinstance(picp, float) else picp
        mpiw_str = f"{mpiw:.2f}" if isinstance(mpiw, float) else mpiw
        
        print("{:<20} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.3f} {:>8.3f} {:>8} {:>8}".format(
            method[:20],
            metrics.get('intercept_R2', 0),
            metrics.get('slope_R2', 0),
            metrics.get('acceleration_R2', 0),
            metrics.get('intercept_RMSE', 0),
            metrics.get('intercept_correlation', 0),
            picp_str,
            mpiw_str
        ))
    
    # Survival Table
    logger.info("\n" + "=" * 70)
    logger.info("SURVIVAL PREDICTION RESULTS (Table 3)")
    logger.info("=" * 70)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "C-index", "AUC-2yr", "AUC-3yr", "AUC-5yr", "HR"
    ))
    print("-" * 80)
    
    for method, metrics in surv_results.items():
        hr = metrics.get('hazard_ratio', np.nan)
        hr_str = f"{hr:.2f}" if not np.isnan(hr) and not np.isinf(hr) else "N/A"
        
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
            method[:25],
            metrics.get('c_index', 0),
            metrics.get('auc_2yr', 0),
            metrics.get('auc_3yr', 0),
            metrics.get('auc_5yr', 0),
            hr_str
        ))
    
    # Classification Table
    logger.info("\n" + "=" * 70)
    logger.info("PROGRESSOR CLASSIFICATION RESULTS")
    logger.info("=" * 70)
    
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"
    ))
    print("-" * 75)
    
    for method, metrics in class_results['progressor'].items():
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            method[:20],
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('auc_roc', 0)
        ))
    
    # Conversion Classification Table
    logger.info("\n" + "=" * 70)
    logger.info("CONVERSION CLASSIFICATION RESULTS")
    logger.info("=" * 70)
    
    for horizon_key, methods in class_results['conversion'].items():
        if not methods:
            continue
        print(f"\n--- {horizon_key} Conversion ---")
        print("{:<20} {:>10} {:>10} {:>10} {:>10}".format(
            "Method", "F1", "AUC-ROC", "Precision", "Recall"
        ))
        print("-" * 60)
        
        for method, metrics in methods.items():
            print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                method[:20],
                metrics.get('f1_score', 0),
                metrics.get('auc_roc', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0)
            ))
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Full results JSON
    all_results = {
        'trajectory': traj_results,
        'survival': surv_results,
        'classification': class_results,
        'config': {
            'epochs': config.num_epochs,
            'test_size': config.test_size,
            'val_size': config.val_size,
            'random_seed': config.random_seed,
            'slope_threshold': config.slope_threshold
        },
        'data_info': {
            'n_total': len(data['X']),
            'n_train': len(data['train_idx']),
            'n_val': len(data['val_idx']),
            'n_test': len(data['test_idx']),
            'event_rate': float(data['events'].mean())
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(output_dir, 'complete_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Trajectory CSV
    traj_rows = []
    for method, metrics in traj_results.items():
        row = {'Method': method}
        for k, v in metrics.items():
            row[k] = v
        traj_rows.append(row)
    traj_df = pd.DataFrame(traj_rows)
    traj_csv = os.path.join(output_dir, 'trajectory_comparison_table.csv')
    traj_df.to_csv(traj_csv, index=False)
    
    # Survival CSV
    surv_rows = []
    for method, metrics in surv_results.items():
        row = {'Method': method}
        for k, v in metrics.items():
            row[k] = v
        surv_rows.append(row)
    surv_df = pd.DataFrame(surv_rows)
    surv_csv = os.path.join(output_dir, 'survival_comparison_table.csv')
    surv_df.to_csv(surv_csv, index=False)
    
    # Classification CSV
    class_rows = []
    for method, metrics in class_results['progressor'].items():
        row = {'Method': method, 'Task': 'Progressor'}
        for k, v in metrics.items():
            row[k] = v
        class_rows.append(row)
    
    for horizon_key, methods in class_results['conversion'].items():
        for method, metrics in methods.items():
            row = {'Method': method, 'Task': f'Conversion_{horizon_key}'}
            for k, v in metrics.items():
                row[k] = v
            class_rows.append(row)
    
    class_df = pd.DataFrame(class_rows)
    class_csv = os.path.join(output_dir, 'classification_comparison_table.csv')
    class_df.to_csv(class_csv, index=False)
    
    # Generate plots
    generate_comparison_plots(traj_results, surv_results, class_results, output_dir)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {results_file}")
    logger.info(f"  - {traj_csv}")
    logger.info(f"  - {surv_csv}")
    logger.info(f"  - {class_csv}")
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE BASELINE COMPARISON FINISHED")
    logger.info("=" * 70)
    
    return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete Baseline Comparison for PROGRESS Framework'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing nacc_integrated_dataset.pkl')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced epochs')
    
    args = parser.parse_args()
    
    try:
        run_complete_comparison(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            quick_test=args.quick_test,
            epochs=args.epochs
        )
        return 0
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
