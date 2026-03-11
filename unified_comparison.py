#!/usr/bin/env python3
"""
unified_comparison.py - Fair Head-to-Head Comparison: PROGRESS vs Baselines

This script runs PROGRESS and all baseline methods on the EXACT SAME data split
for a fair comparison. Results are formatted for direct inclusion in the paper.

Usage:
    python unified_comparison.py --data-dir ./dataset
    python unified_comparison.py --data-dir ./dataset --quick-test
    python unified_comparison.py --data-dir ./dataset --epochs 100

Output:
    - unified_comparison_results.json: Complete results
    - comparison_table_trajectory.csv: Table 2 for paper
    - comparison_table_survival.csv: Table 3 for paper
    - comparison_plots/: Visualization directory

Author: PROGRESS Paper
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import os
import sys
import json
import pickle
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, BayesianRidge
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Info: XGBoost not installed. Install with: pip install xgboost")

try:
    from lifelines import CoxPHFitter
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
    """Configuration for comparison experiments."""
    
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    gradient_clip: float = 1.0
    
    # PROGRESS architecture
    traj_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    surv_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    attention_heads: int = 4
    
    # Uncertainty
    mc_dropout_samples: int = 50
    calibration_weight: float = 0.1
    ranking_weight: float = 0.5
    
    # Evaluation
    survival_horizons: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    
    # Data split (FIXED for reproducibility)
    test_size: float = 0.2
    val_size: float = 0.15
    random_seed: int = 42
    
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
# DATA LOADING
# =============================================================================

MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}

def is_valid(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if value in MISSING_CODES:
        return False
    return True

def clean_value(value, default: float = np.nan) -> float:
    if not is_valid(value):
        return default
    return float(value)


def load_and_prepare_data(data_dir: str, config: ComparisonConfig) -> Dict:
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
        
        # === Trajectory Parameters ===
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
    logger.info("\n" + "=" * 70)
    logger.info("CREATING FIXED DATA SPLITS")
    logger.info("=" * 70)
    
    # First split: train+val vs test
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=events
    )
    
    # Second split: train vs val
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
# PROGRESS MODELS (from PROGRESS.py)
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
        
    def forward(self, x):
        batch_size = x.size(0)
        x_proj = self.input_proj(x)
        
        Q = self.W_Q(x_proj).view(batch_size, self.num_heads, self.head_dim)
        K = self.W_K(x_proj).view(batch_size, self.num_heads, self.head_dim)
        V = self.W_V(x_proj).view(batch_size, self.num_heads, self.head_dim)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.bmm(attention_weights, V)
        attended = attended.view(batch_size, self.scaled_dim)
        output = self.W_O(attended) + x
        
        return output, attention_weights.mean(dim=1)


class TrajectoryParameterNetwork(nn.Module):
    """PROGRESS Model 1: Probabilistic Trajectory Parameter Network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3, num_attention_heads: int = 4):
        super().__init__()
        
        self.attention = BiomarkerAttention(input_dim, num_attention_heads, dropout)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], 3)
        self.logvar_head = nn.Linear(hidden_dims[-1], 3)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        attended, attn_weights = self.attention(x)
        h = self.encoder(attended)
        mean = self.mean_head(h)
        log_var = torch.clamp(self.logvar_head(h), -10, 10)
        return {'mean': mean, 'log_var': log_var, 'std': torch.exp(0.5 * log_var)}
    
    def predict_with_uncertainty(self, x, n_samples: int = 50):
        # Enable dropout ONLY, keep BatchNorm in eval mode
        # This is critical - calling self.train() would break BatchNorm!
        self.eval()  # First set to eval (BatchNorm uses running stats)
        
        # Manually enable dropout layers only
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        all_means, all_vars = [], []
        
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)
                all_means.append(out['mean'])
                all_vars.append(torch.exp(out['log_var']))
        
        all_means = torch.stack(all_means)
        all_vars = torch.stack(all_vars)
        
        mean_pred = all_means.mean(dim=0)
        aleatoric_var = all_vars.mean(dim=0)
        epistemic_var = all_means.var(dim=0)
        total_var = aleatoric_var + epistemic_var
        
        self.eval()  # Back to full eval mode
        return {
            'mean': mean_pred,
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'epistemic_std': torch.sqrt(epistemic_var),
            'total_std': torch.sqrt(total_var)
        }


class DeepSurvivalNetwork(nn.Module):
    """PROGRESS Model 2: Deep Survival Network."""
    
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
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# BASELINE MODELS
# =============================================================================

class MLPBaseline(nn.Module):
    """Simple MLP baseline (no attention)."""
    
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


class DeepSurvBaseline(nn.Module):
    """DeepSurv baseline (simpler than PROGRESS)."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [32, 16],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# METRICS
# =============================================================================

def compute_c_index(risk_scores: np.ndarray, times: np.ndarray,
                   events: np.ndarray) -> float:
    """Harrell's C-index."""
    n = len(times)
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
    
    return concordant / comparable if comparable > 0 else 0.5


def compute_time_dependent_auc(risk_scores: np.ndarray, times: np.ndarray,
                              events: np.ndarray, horizon: float) -> float:
    """Time-dependent AUC."""
    cases = (times <= horizon) & (events == 1)
    controls = times > horizon
    
    n_cases = cases.sum()
    n_controls = controls.sum()
    
    if n_cases < 2 or n_controls < 2:
        return 0.5
    
    case_risks = risk_scores[cases]
    control_risks = risk_scores[controls]
    
    concordant = 0
    for cr in case_risks:
        concordant += (cr > control_risks).sum()
        concordant += 0.5 * (cr == control_risks).sum()
    
    return concordant / (n_cases * n_controls)


def compute_picp(y_true: np.ndarray, y_pred: np.ndarray, 
                y_std: np.ndarray, alpha: float = 0.05) -> float:
    """Prediction Interval Coverage Probability."""
    z = stats.norm.ppf(1 - alpha/2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return np.mean((y_true >= lower) & (y_true <= upper))


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_progress(data: Dict, config: ComparisonConfig) -> Dict:
    """Train PROGRESS models."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PROGRESS")
    logger.info("=" * 70)
    
    device = config.get_device()
    
    # Prepare data
    X_train = torch.FloatTensor(data['X'][data['train_idx']])
    y_train = torch.FloatTensor(data['y_traj'][data['train_idx']])
    t_train = torch.FloatTensor(data['times'][data['train_idx']])
    e_train = torch.FloatTensor(data['events'][data['train_idx']])
    
    X_val = torch.FloatTensor(data['X'][data['val_idx']]).to(device)
    y_val = torch.FloatTensor(data['y_traj'][data['val_idx']]).to(device)
    t_val = torch.FloatTensor(data['times'][data['val_idx']]).to(device)
    e_val = torch.FloatTensor(data['events'][data['val_idx']]).to(device)
    
    X_test = torch.FloatTensor(data['X'][data['test_idx']]).to(device)
    y_test = data['y_traj'][data['test_idx']]
    t_test = data['times'][data['test_idx']]
    e_test = data['events'][data['test_idx']]
    
    input_dim = X_train.shape[1]
    
    # Create DataLoaders for proper mini-batch training
    train_traj_dataset = TensorDataset(X_train, y_train)
    train_traj_loader = DataLoader(train_traj_dataset, batch_size=config.batch_size, 
                                    shuffle=True, drop_last=True)
    
    train_surv_dataset = TensorDataset(X_train, t_train, e_train)
    train_surv_loader = DataLoader(train_surv_dataset, batch_size=config.batch_size,
                                    shuffle=True, drop_last=True)
    
    # Initialize models
    traj_model = TrajectoryParameterNetwork(
        input_dim, config.traj_hidden_dims, config.dropout, config.attention_heads
    ).to(device)
    
    surv_model = DeepSurvivalNetwork(
        input_dim, config.surv_hidden_dims, config.dropout
    ).to(device)
    
    # === Train Trajectory Model ===
    logger.info("Training Trajectory Network...")
    
    traj_optimizer = optim.AdamW(traj_model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    traj_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        traj_optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_traj_state = None
    
    for epoch in range(config.num_epochs):
        # Train with mini-batches
        traj_model.train()
        epoch_losses = []
        
        for batch_X, batch_y in train_traj_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            traj_optimizer.zero_grad()
            
            output = traj_model(batch_X)
            
            # Heteroscedastic NLL loss with stability improvements
            pred_var = torch.exp(output['log_var'])
            
            # Clamp variance to prevent instability
            pred_var = torch.clamp(pred_var, min=1e-4, max=10.0)
            
            # NLL = 0.5 * [(y - μ)²/σ² + log(σ²)]
            mse_term = (batch_y - output['mean'])**2 / pred_var
            log_var_term = torch.log(pred_var)
            nll = 0.5 * (mse_term + log_var_term).mean()
            
            # Add MSE regularization to ensure good point predictions
            mse_loss = nn.MSELoss()(output['mean'], batch_y)
            
            # Combined loss: NLL + MSE regularization
            loss = nll + 0.5 * mse_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(traj_model.parameters(), config.gradient_clip)
            traj_optimizer.step()
            
            epoch_losses.append(loss.item())
        
        traj_scheduler.step()
        
        # Validate
        traj_model.eval()
        with torch.no_grad():
            val_out = traj_model(X_val)
            val_mse = nn.MSELoss()(val_out['mean'], y_val).item()
            # Use MSE for validation (more stable than NLL)
            val_loss = val_mse
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_traj_state = {k: v.cpu().clone() for k, v in traj_model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            train_loss = np.mean(epoch_losses)
            logger.info(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Val MSE={val_loss:.4f}")
    
    if best_traj_state:
        traj_model.load_state_dict(best_traj_state)
    
    # === Train Survival Model ===
    logger.info("Training Survival Network...")
    
    surv_optimizer = optim.AdamW(surv_model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    surv_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        surv_optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    
    best_val_cindex = 0.0
    patience_counter = 0
    best_surv_state = None
    
    for epoch in range(config.num_epochs):
        # Train with mini-batches
        surv_model.train()
        epoch_losses = []
        
        for batch_X, batch_t, batch_e in train_surv_loader:
            batch_X = batch_X.to(device)
            batch_t = batch_t.to(device)
            batch_e = batch_e.to(device)
            
            surv_optimizer.zero_grad()
            
            risk_scores = surv_model(batch_X).squeeze()
            
            # Cox partial likelihood
            sorted_idx = torch.argsort(batch_t, descending=True)
            sorted_risks = risk_scores[sorted_idx]
            sorted_events = batch_e[sorted_idx]
            
            max_risk = sorted_risks.max()
            exp_risks = torch.exp(sorted_risks - max_risk)
            cumsum_exp = torch.cumsum(exp_risks, dim=0)
            log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
            
            log_lik = sorted_risks - log_cumsum
            n_events = sorted_events.sum()
            
            if n_events > 0:
                cox_loss = -(log_lik * sorted_events).sum() / n_events
            else:
                cox_loss = risk_scores.mean() * 0.0
            
            cox_loss.backward()
            torch.nn.utils.clip_grad_norm_(surv_model.parameters(), config.gradient_clip)
            surv_optimizer.step()
            
            epoch_losses.append(cox_loss.item())
        
        surv_scheduler.step()
        
        # Validate using C-index (better metric than loss)
        surv_model.eval()
        with torch.no_grad():
            val_risks = surv_model(X_val).cpu().numpy().squeeze()
            val_cindex = compute_c_index(val_risks, 
                                         data['times'][data['val_idx']], 
                                         data['events'][data['val_idx']])
        
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
            best_surv_state = {k: v.cpu().clone() for k, v in surv_model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            train_loss = np.mean(epoch_losses)
            logger.info(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val C-index={val_cindex:.4f}")
    
    if best_surv_state:
        surv_model.load_state_dict(best_surv_state)
    
    # === Evaluate on Test Set ===
    logger.info("Evaluating on test set...")
    
    traj_model.eval()
    surv_model.eval()
    
    with torch.no_grad():
        # Trajectory predictions with uncertainty
        traj_output = traj_model.predict_with_uncertainty(X_test, config.mc_dropout_samples)
        traj_pred = traj_output['mean'].cpu().numpy()
        traj_std = traj_output['total_std'].cpu().numpy()
        
        # Survival predictions
        risk_scores = surv_model(X_test).cpu().numpy().squeeze()
    
    # Compute metrics
    results = {'method': 'PROGRESS'}
    
    # Trajectory metrics
    param_names = ['intercept', 'slope', 'acceleration']
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], traj_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], traj_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], traj_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(traj_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], traj_pred[:, i])[0, 1]
        else:
            results[f'{name}_corr'] = 0.0
        results[f'{name}_picp'] = compute_picp(y_test[:, i], traj_pred[:, i], traj_std[:, i])
    
    # Survival metrics
    results['c_index'] = compute_c_index(risk_scores, t_test, e_test)
    for horizon in config.survival_horizons:
        results[f'auc_{horizon:.0f}yr'] = compute_time_dependent_auc(
            risk_scores, t_test, e_test, horizon
        )
    
    return results


def train_baselines(data: Dict, config: ComparisonConfig) -> List[Dict]:
    """Train all baseline models."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("=" * 70)
    
    device = config.get_device()
    
    # Extract splits
    X_train = data['X'][data['train_idx']]
    y_train = data['y_traj'][data['train_idx']]
    t_train = data['times'][data['train_idx']]
    e_train = data['events'][data['train_idx']]
    
    X_test = data['X'][data['test_idx']]
    y_test = data['y_traj'][data['test_idx']]
    t_test = data['times'][data['test_idx']]
    e_test = data['events'][data['test_idx']]
    
    feature_names = data['feature_names']
    all_results = []
    param_names = ['intercept', 'slope', 'acceleration']
    
    # === TRAJECTORY BASELINES ===
    
    # 1. Linear Regression
    logger.info("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results = {'method': 'Linear Regression'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    all_results.append(results)
    
    # 2. Ridge Regression
    logger.info("  Training Ridge Regression...")
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results = {'method': 'Ridge'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    all_results.append(results)
    
    # 3. Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results = {'method': 'Random Forest'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    all_results.append(results)
    
    # 4. Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    y_pred = np.zeros_like(y_test)
    for i in range(3):
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train, y_train[:, i])
        y_pred[:, i] = gb.predict(X_test)
    results = {'method': 'Gradient Boosting'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    all_results.append(results)
    
    # 5. XGBoost
    if HAS_XGBOOST:
        logger.info("  Training XGBoost...")
        y_pred = np.zeros_like(y_test)
        for i in range(3):
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42,
                                     verbosity=0)
            model.fit(X_train, y_train[:, i])
            y_pred[:, i] = model.predict(X_test)
        results = {'method': 'XGBoost'}
        for i, name in enumerate(param_names):
            results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
            results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
            if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        all_results.append(results)
    
    # 6. SVR
    logger.info("  Training SVR...")
    y_pred = np.zeros_like(y_test)
    for i in range(3):
        svr = SVR(kernel='rbf', C=1.0)
        svr.fit(X_train, y_train[:, i])
        y_pred[:, i] = svr.predict(X_test)
    results = {'method': 'SVR'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    all_results.append(results)
    
    # 7. Bayesian Ridge
    logger.info("  Training Bayesian Ridge...")
    y_pred = np.zeros_like(y_test)
    y_std = np.zeros_like(y_test)
    for i in range(3):
        br = BayesianRidge()
        br.fit(X_train, y_train[:, i])
        pred, std = br.predict(X_test, return_std=True)
        y_pred[:, i] = pred
        y_std[:, i] = std
    results = {'method': 'Bayesian Ridge'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        results[f'{name}_picp'] = compute_picp(y_test[:, i], y_pred[:, i], y_std[:, i])
    all_results.append(results)
    
    # 8. MLP Baseline
    logger.info("  Training MLP Baseline...")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    mlp = MLPBaseline(X_train.shape[1], 3, [64, 32], 0.3).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3)
    
    mlp.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        pred = mlp(X_train_t)
        loss = nn.MSELoss()(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_test_t).cpu().numpy()
    
    results = {'method': 'MLP'}
    for i, name in enumerate(param_names):
        results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
        if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            results[f'{name}_corr'] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    all_results.append(results)
    
    # === SURVIVAL BASELINES ===
    
    # 9. Cox PH
    if HAS_LIFELINES:
        logger.info("  Training Cox PH...")
        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train['T'] = t_train
        df_train['E'] = e_train
        
        df_test = pd.DataFrame(X_test, columns=feature_names)
        
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df_train, duration_col='T', event_col='E')
            risk_scores = cph.predict_partial_hazard(df_test).values.flatten()
            
            # Find Cox PH result to add survival metrics
            cox_results = {'method': 'Cox PH'}
            cox_results['c_index'] = compute_c_index(risk_scores, t_test, e_test)
            for horizon in config.survival_horizons:
                cox_results[f'auc_{horizon:.0f}yr'] = compute_time_dependent_auc(
                    risk_scores, t_test, e_test, horizon
                )
            all_results.append(cox_results)
        except Exception as e:
            logger.warning(f"Cox PH failed: {e}")
    
    # 10. Random Survival Forest
    if HAS_SKSURV:
        logger.info("  Training Random Survival Forest...")
        y_train_surv = np.array([(bool(e), t) for e, t in zip(e_train, t_train)],
                               dtype=[('event', bool), ('time', float)])
        
        try:
            rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, 
                                       random_state=42, n_jobs=-1)
            rsf.fit(X_train, y_train_surv)
            risk_scores = rsf.predict(X_test)
            
            c_index, _, _, _, _ = concordance_index_censored(
                e_test.astype(bool), t_test, risk_scores
            )
            
            rsf_results = {'method': 'Random Survival Forest'}
            rsf_results['c_index'] = c_index
            for horizon in config.survival_horizons:
                rsf_results[f'auc_{horizon:.0f}yr'] = compute_time_dependent_auc(
                    risk_scores, t_test, e_test, horizon
                )
            all_results.append(rsf_results)
        except Exception as e:
            logger.warning(f"RSF failed: {e}")
        
        # 11. Cox-Lasso
        logger.info("  Training Cox-Lasso...")
        try:
            cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True)
            cox_lasso.fit(X_train, y_train_surv)
            risk_scores = cox_lasso.predict(X_test)
            
            c_index, _, _, _, _ = concordance_index_censored(
                e_test.astype(bool), t_test, risk_scores
            )
            
            cl_results = {'method': 'Cox-Lasso'}
            cl_results['c_index'] = c_index
            for horizon in config.survival_horizons:
                cl_results[f'auc_{horizon:.0f}yr'] = compute_time_dependent_auc(
                    risk_scores, t_test, e_test, horizon
                )
            all_results.append(cl_results)
        except Exception as e:
            logger.warning(f"Cox-Lasso failed: {e}")
        
        # 12. Gradient Boosting Survival
        logger.info("  Training GB Survival...")
        try:
            gbs = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=5, 
                                                    random_state=42)
            gbs.fit(X_train, y_train_surv)
            risk_scores = gbs.predict(X_test)
            
            c_index, _, _, _, _ = concordance_index_censored(
                e_test.astype(bool), t_test, risk_scores
            )
            
            gbs_results = {'method': 'GB Survival'}
            gbs_results['c_index'] = c_index
            for horizon in config.survival_horizons:
                gbs_results[f'auc_{horizon:.0f}yr'] = compute_time_dependent_auc(
                    risk_scores, t_test, e_test, horizon
                )
            all_results.append(gbs_results)
        except Exception as e:
            logger.warning(f"GBS failed: {e}")
    
    # 13. DeepSurv Baseline
    logger.info("  Training DeepSurv Baseline...")
    X_train_t = torch.FloatTensor(X_train).to(device)
    t_train_t = torch.FloatTensor(t_train).to(device)
    e_train_t = torch.FloatTensor(e_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    deepsurv = DeepSurvBaseline(X_train.shape[1], [32, 16], 0.3).to(device)
    optimizer = optim.AdamW(deepsurv.parameters(), lr=1e-3)
    
    deepsurv.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        risk_scores = deepsurv(X_train_t).squeeze()
        
        sorted_idx = torch.argsort(t_train_t, descending=True)
        sorted_risks = risk_scores[sorted_idx]
        sorted_events = e_train_t[sorted_idx]
        
        max_risk = sorted_risks.max()
        exp_risks = torch.exp(sorted_risks - max_risk)
        cumsum_exp = torch.cumsum(exp_risks, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
        
        log_lik = sorted_risks - log_cumsum
        n_events = sorted_events.sum()
        
        if n_events > 0:
            loss = -(log_lik * sorted_events).sum() / n_events
            loss.backward()
            optimizer.step()
    
    deepsurv.eval()
    with torch.no_grad():
        risk_scores = deepsurv(X_test_t).cpu().numpy().squeeze()
    
    ds_results = {'method': 'DeepSurv (baseline)'}
    ds_results['c_index'] = compute_c_index(risk_scores, t_test, e_test)
    for horizon in config.survival_horizons:
        ds_results[f'auc_{horizon:.0f}yr'] = compute_time_dependent_auc(
            risk_scores, t_test, e_test, horizon
        )
    all_results.append(ds_results)
    
    return all_results


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(data_dir: str, output_dir: str = None,
                   quick_test: bool = False, epochs: int = 100) -> Dict:
    """Run complete comparison."""
    
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'unified_comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    config = ComparisonConfig(
        num_epochs=20 if quick_test else epochs,
        patience=5 if quick_test else 15
    )
    
    logger.info("=" * 70)
    logger.info("UNIFIED COMPARISON: PROGRESS vs BASELINES")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Device: {config.get_device()}")
    
    # Load data with FIXED splits
    data = load_and_prepare_data(data_dir, config)
    
    # Train PROGRESS
    progress_results = train_progress(data, config)
    
    # Train baselines
    baseline_results = train_baselines(data, config)
    
    # Combine all results
    all_results = [progress_results] + baseline_results
    
    # === PRINT TRAJECTORY COMPARISON TABLE ===
    logger.info("\n" + "=" * 70)
    logger.info("TRAJECTORY PREDICTION COMPARISON (Table 2)")
    logger.info("=" * 70)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format(
        "Method", "Intercept R²", "Slope R²", "Accel R²", "PICP (95%)"
    ))
    print("-" * 75)
    
    traj_methods = [r for r in all_results if 'intercept_r2' in r]
    for r in traj_methods:
        picp = r.get('intercept_picp', 'N/A')
        if isinstance(picp, float):
            picp_str = f"{picp:.2%}"
        else:
            picp_str = 'N/A'
        
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>10}".format(
            r['method'],
            r.get('intercept_r2', 0),
            r.get('slope_r2', 0),
            r.get('acceleration_r2', 0),
            picp_str
        ))
    
    # === PRINT SURVIVAL COMPARISON TABLE ===
    logger.info("\n" + "=" * 70)
    logger.info("SURVIVAL PREDICTION COMPARISON (Table 3)")
    logger.info("=" * 70)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "C-index", "AUC-2yr", "AUC-3yr", "AUC-5yr"
    ))
    print("-" * 70)
    
    surv_methods = [r for r in all_results if 'c_index' in r]
    for r in surv_methods:
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r['method'],
            r.get('c_index', 0),
            r.get('auc_2yr', 0),
            r.get('auc_3yr', 0),
            r.get('auc_5yr', 0)
        ))
    
    # === SAVE RESULTS ===
    
    # Full results JSON
    results_file = os.path.join(output_dir, 'unified_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'config': {
                'epochs': config.num_epochs,
                'test_size': config.test_size,
                'random_seed': config.random_seed
            },
            'data_info': {
                'n_total': len(data['X']),
                'n_train': len(data['train_idx']),
                'n_val': len(data['val_idx']),
                'n_test': len(data['test_idx']),
                'event_rate': float(data['events'].mean())
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    # Trajectory comparison CSV
    traj_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Intercept R²': r.get('intercept_r2', ''),
            'Intercept RMSE': r.get('intercept_rmse', ''),
            'Slope R²': r.get('slope_r2', ''),
            'Slope RMSE': r.get('slope_rmse', ''),
            'Acceleration R²': r.get('acceleration_r2', ''),
            'Acceleration RMSE': r.get('acceleration_rmse', ''),
            'PICP (95%)': r.get('intercept_picp', '')
        }
        for r in traj_methods
    ])
    traj_csv = os.path.join(output_dir, 'comparison_table_trajectory.csv')
    traj_df.to_csv(traj_csv, index=False)
    
    # Survival comparison CSV
    surv_df = pd.DataFrame([
        {
            'Method': r['method'],
            'C-index': r.get('c_index', ''),
            'AUC-2yr': r.get('auc_2yr', ''),
            'AUC-3yr': r.get('auc_3yr', ''),
            'AUC-5yr': r.get('auc_5yr', '')
        }
        for r in surv_methods
    ])
    surv_csv = os.path.join(output_dir, 'comparison_table_survival.csv')
    surv_df.to_csv(surv_csv, index=False)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {results_file}")
    logger.info(f"  - {traj_csv}")
    logger.info(f"  - {surv_csv}")
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON COMPLETED")
    logger.info("=" * 70)
    
    return {'results': all_results, 'data': data}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified Comparison: PROGRESS vs Baselines'
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
        run_comparison(
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