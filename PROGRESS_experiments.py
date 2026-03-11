#!/usr/bin/env python3
"""
PROGRESS: PRognostic Generalization from REsting Static Signatures
Extended with Hyperparameter Sensitivity Experiments

This version adds experiments for testing different hidden layer widths
(32, 64, 128, 256) to analyze model robustness and optimal architecture.

================================================================================
USAGE
================================================================================

# Run hidden layer width experiments:
python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width

# Run with specific widths:
python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width --widths 32 64 128 256

# Quick test:
python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width --quick-test

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
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

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str = '.', log_level: int = logging.INFO):
    """Configure logging to both file and console."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'progress_experiments.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PROGRESSConfig:
    """Configuration for PROGRESS framework."""
    
    # === Data Parameters ===
    sequence_length: int = 5
    min_visits_trajectory: int = 3
    
    # === Model 1: Trajectory Network Architecture ===
    traj_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    traj_dropout: float = 0.3
    traj_attention_heads: int = 4
    traj_use_batch_norm: bool = True
    
    # === Model 2: Survival Network Architecture ===
    surv_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    surv_dropout: float = 0.3
    surv_use_batch_norm: bool = True
    
    # === Training Parameters ===
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15
    gradient_clip: float = 1.0
    
    # === Loss Weights ===
    nll_weight: float = 1.0
    calibration_weight: float = 0.1
    ranking_weight: float = 0.5
    
    # === Uncertainty Estimation ===
    mc_dropout_samples: int = 50
    
    # === Cross-Validation ===
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    
    # === Evaluation ===
    survival_horizons: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    
    # === Device ===
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
    
    def to_dict(self) -> Dict:
        return {
            k: v if not isinstance(v, list) else v.copy()
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PROGRESSConfig':
        return cls(**d)


# =============================================================================
# DATA PROCESSING (Same as original)
# =============================================================================

class PROGRESSDataset(Dataset):
    """Dataset for PROGRESS framework."""
    
    MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}
    
    def __init__(self,
                 integrated_data: pd.DataFrame,
                 sequences_data: pd.DataFrame = None,
                 scaler: StandardScaler = None,
                 fit_scaler: bool = False,
                 config: PROGRESSConfig = None):
        self.config = config or PROGRESSConfig()
        self.integrated_data = integrated_data
        self.sequences_data = sequences_data
        
        self.subjects = self._get_valid_subjects()
        logger.info(f"Dataset: {len(self.subjects)} valid subjects")
        
        self.baseline_features, self.feature_names = self._extract_baseline_features()
        self.trajectory_params = self._compute_trajectory_parameters()
        self.survival_data = self._extract_survival_data()
        
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
    
    def _get_valid_subjects(self) -> List[str]:
        valid_subjects = []
        for _, row in self.integrated_data.iterrows():
            naccid = row.get('NACCID')
            if naccid is None:
                continue
            
            abeta = row.get('ABETA_harm')
            ptau = row.get('PTAU_harm')
            ttau = row.get('TTAU_harm')
            
            has_biomarker = any([
                self._is_valid(abeta),
                self._is_valid(ptau),
                self._is_valid(ttau)
            ])
            
            if not has_biomarker:
                continue
            
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) >= 2:
                valid_subjects.append(naccid)
        
        return valid_subjects
    
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
    
    def _extract_baseline_features(self) -> Tuple[np.ndarray, List[str]]:
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
            
            abeta = self._clean_value(row.get('ABETA_harm'), 500.0)
            ptau = self._clean_value(row.get('PTAU_harm'), 50.0)
            ttau = self._clean_value(row.get('TTAU_harm'), 300.0)
            
            if abeta > 0:
                ptau_abeta_ratio = ptau / abeta
            else:
                ptau_abeta_ratio = 0.1
            
            if ptau > 0:
                ttau_ptau_ratio = ttau / ptau
            else:
                ttau_ptau_ratio = 6.0
            
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
            
            features = [
                abeta, ptau, ttau,
                ptau_abeta_ratio, ttau_ptau_ratio,
                age, sex, educ,
                baseline_mmse, baseline_cdr
            ]
            
            features_list.append(features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        for col in range(features_array.shape[1]):
            col_data = features_array[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                if np.isnan(median_val):
                    median_val = 0.0
                features_array[mask, col] = median_val
        
        return features_array, feature_names
    
    def _compute_trajectory_parameters(self) -> np.ndarray:
        trajectory_params = []
        valid_trajectory_count = 0
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            trajectory = row.get('clinical_trajectory', [])
            
            if not isinstance(trajectory, list) or len(trajectory) < self.config.min_visits_trajectory:
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
            
            if len(times) >= self.config.min_visits_trajectory:
                times = np.array(times)
                cdr_values = np.array(cdr_values)
                
                try:
                    coeffs = np.polyfit(times, cdr_values, deg=2)
                    alpha = coeffs[2]
                    beta = coeffs[1]
                    gamma = coeffs[0]
                    
                    if abs(alpha) < 20 and abs(beta) < 5 and abs(gamma) < 1:
                        trajectory_params.append([alpha, beta, gamma])
                        valid_trajectory_count += 1
                    else:
                        trajectory_params.append([np.nan, np.nan, np.nan])
                        
                except (np.linalg.LinAlgError, ValueError):
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
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
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
    
    def get_subject_ids(self) -> List[str]:
        return self.subjects.copy()
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class BiomarkerAttention(nn.Module):
    """Biomarker attention mechanism for interpretability."""
    
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
        Q = self.W_Q(x_proj)
        K = self.W_K(x_proj)
        V = self.W_V(x_proj)
        
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.bmm(attention_weights, V)
        attended = attended.view(batch_size, self.scaled_dim)
        output = self.W_O(attended)
        output = output + x
        
        return output, attention_weights.mean(dim=1)


class TrajectoryParameterNetwork(nn.Module):
    """Probabilistic Trajectory Parameter Network (Model 1)."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3,
                 num_attention_heads: int = 4,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = 3
        self.hidden_dims = hidden_dims
        
        self.attention = BiomarkerAttention(input_dim, num_attention_heads, dropout)
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
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
        log_var = self.logvar_head(h)
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        output = {
            'mean': mean,
            'log_var': log_var,
            'std': torch.exp(0.5 * log_var)
        }
        
        if return_attention:
            output['attention'] = attention_weights
        
        return output
    
    def predict_with_uncertainty(self, 
                                 x: torch.Tensor, 
                                 n_samples: int = 50) -> Dict[str, torch.Tensor]:
        self.train()
        
        all_means = []
        all_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                all_means.append(output['mean'])
                all_vars.append(torch.exp(output['log_var']))
        
        all_means = torch.stack(all_means)
        all_vars = torch.stack(all_vars)
        
        mean_pred = all_means.mean(dim=0)
        aleatoric_var = all_vars.mean(dim=0)
        epistemic_var = all_means.var(dim=0)
        total_var = aleatoric_var + epistemic_var
        
        self.eval()
        
        return {
            'mean': mean_pred,
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'epistemic_std': torch.sqrt(epistemic_var),
            'total_std': torch.sqrt(total_var)
        }


class DeepSurvivalNetwork(nn.Module):
    """Deep Survival Network (Model 2)."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
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
    
    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.forward(x))


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory parameter prediction."""
    
    def __init__(self, calibration_weight: float = 0.1):
        super().__init__()
        self.calibration_weight = calibration_weight
    
    def forward(self,
                pred_mean: torch.Tensor,
                pred_log_var: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        pred_var = torch.exp(pred_log_var)
        
        nll = 0.5 * (
            (targets - pred_mean) ** 2 / (pred_var + 1e-8) + 
            pred_log_var
        ).mean()
        
        pred_std = torch.sqrt(pred_var + 1e-8)
        z_scores = torch.abs((targets - pred_mean) / pred_std)
        within_95 = (z_scores < 1.96).float().mean()
        calibration_loss = (within_95 - 0.95) ** 2
        
        total = nll + self.calibration_weight * calibration_loss
        mse = F.mse_loss(pred_mean, targets)
        
        return {
            'total': total,
            'nll': nll,
            'calibration': calibration_loss,
            'coverage_95': within_95,
            'mse': mse
        }


class CoxPartialLikelihoodLoss(nn.Module):
    """Cox partial likelihood loss for survival analysis."""
    
    def __init__(self, ranking_weight: float = 0.5):
        super().__init__()
        self.ranking_weight = ranking_weight
    
    def forward(self,
                risk_scores: torch.Tensor,
                times: torch.Tensor,
                events: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        risk_scores = risk_scores.view(-1)
        
        sorted_indices = torch.argsort(times, descending=True)
        sorted_risks = risk_scores[sorted_indices]
        sorted_events = events[sorted_indices].float()
        
        max_risk = sorted_risks.max()
        exp_risks = torch.exp(sorted_risks - max_risk)
        cumsum_exp = torch.cumsum(exp_risks, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
        
        log_lik = sorted_risks - log_cumsum
        n_events = sorted_events.sum()
        
        if n_events > 0:
            cox_loss = -(log_lik * sorted_events).sum() / (n_events + 1e-8)
        else:
            cox_loss = risk_scores.mean() * 0.0 + 0.1 * (risk_scores ** 2).mean()
        
        ranking_loss = self._ranking_loss(risk_scores, times, events)
        total = cox_loss + self.ranking_weight * ranking_loss
        
        return {
            'total': total,
            'cox': cox_loss,
            'ranking': ranking_loss
        }
    
    def _ranking_loss(self,
                      risk_scores: torch.Tensor,
                      times: torch.Tensor,
                      events: torch.Tensor,
                      margin: float = 0.1) -> torch.Tensor:
        n = len(times)
        
        if n < 2:
            return risk_scores.sum() * 0.0
        
        risk_scores = risk_scores.view(-1)
        
        risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)
        
        valid_pairs = (events.unsqueeze(1) == 1) & (time_diff < 0)
        n_valid = valid_pairs.sum().float()
        
        if n_valid == 0:
            return risk_scores.sum() * 0.0
        
        violations = torch.sigmoid(-risk_diff + margin)
        ranking_loss = (violations * valid_pairs.float()).sum() / (n_valid + 1e-8)
        
        return ranking_loss


# =============================================================================
# METRICS
# =============================================================================

class PROGRESSMetrics:
    """Evaluation metrics for PROGRESS framework."""
    
    @staticmethod
    def trajectory_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_std: np.ndarray = None) -> Dict[str, float]:
        param_names = ['intercept', 'slope', 'acceleration']
        metrics = {}
        
        for i, name in enumerate(param_names):
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]
            
            metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(true_i, pred_i))
            metrics[f'{name}_MAE'] = mean_absolute_error(true_i, pred_i)
            
            if np.var(true_i) > 0:
                metrics[f'{name}_R2'] = r2_score(true_i, pred_i)
            else:
                metrics[f'{name}_R2'] = 0.0
            
            if np.std(true_i) > 0 and np.std(pred_i) > 0:
                corr, p_val = stats.pearsonr(true_i, pred_i)
                metrics[f'{name}_correlation'] = corr
            else:
                metrics[f'{name}_correlation'] = 0.0
            
            if y_std is not None:
                std_i = y_std[:, i]
                lower = pred_i - 1.96 * std_i
                upper = pred_i + 1.96 * std_i
                coverage = np.mean((true_i >= lower) & (true_i <= upper))
                metrics[f'{name}_PICP'] = coverage
                mpiw = np.mean(2 * 1.96 * std_i)
                metrics[f'{name}_MPIW'] = mpiw
        
        return metrics
    
    @staticmethod
    def concordance_index(risk_scores: np.ndarray,
                         times: np.ndarray,
                         events: np.ndarray) -> float:
        n = len(times)
        concordant = 0
        discordant = 0
        tied_risk = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] == 1 and times[i] < times[j]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
                        
                elif events[j] == 1 and times[j] < times[i]:
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
        
        total = concordant + discordant + tied_risk
        
        if total == 0:
            return 0.5
        
        return (concordant + 0.5 * tied_risk) / total
    
    @staticmethod
    def time_dependent_auc(risk_scores: np.ndarray,
                          times: np.ndarray,
                          events: np.ndarray,
                          horizon: float) -> float:
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


# =============================================================================
# TRAINER
# =============================================================================

class PROGRESSTrainer:
    """Trainer for PROGRESS framework."""
    
    def __init__(self, config: PROGRESSConfig):
        self.config = config
        self.device = config.get_device()
        
        self.traj_model = None
        self.surv_model = None
        
        self.traj_loss_fn = TrajectoryLoss(config.calibration_weight)
        self.surv_loss_fn = CoxPartialLikelihoodLoss(config.ranking_weight)
        
        self.history = {
            'traj_train_loss': [], 'traj_val_loss': [],
            'surv_train_loss': [], 'surv_val_loss': [],
        }
    
    def setup_models(self, input_dim: int, 
                     traj_hidden_dims: List[int] = None,
                     surv_hidden_dims: List[int] = None):
        """Initialize models with specified hidden dimensions."""
        
        if traj_hidden_dims is None:
            traj_hidden_dims = self.config.traj_hidden_dims
        if surv_hidden_dims is None:
            surv_hidden_dims = self.config.surv_hidden_dims
        
        self.traj_model = TrajectoryParameterNetwork(
            input_dim=input_dim,
            hidden_dims=traj_hidden_dims,
            dropout=self.config.traj_dropout,
            num_attention_heads=self.config.traj_attention_heads,
            use_batch_norm=self.config.traj_use_batch_norm
        ).to(self.device)
        
        self.surv_model = DeepSurvivalNetwork(
            input_dim=input_dim,
            hidden_dims=surv_hidden_dims,
            dropout=self.config.surv_dropout,
            use_batch_norm=self.config.surv_use_batch_norm
        ).to(self.device)
        
        traj_params = sum(p.numel() for p in self.traj_model.parameters())
        surv_params = sum(p.numel() for p in self.surv_model.parameters())
        
        return traj_params, surv_params
    
    def train_trajectory_model(self,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               num_epochs: int = None) -> Dict:
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        optimizer = optim.AdamW(
            self.traj_model.parameters(),
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
            self.traj_model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                targets = batch['trajectory_params'].to(self.device)
                
                optimizer.zero_grad()
                
                output = self.traj_model(features)
                loss_dict = self.traj_loss_fn(
                    output['mean'], output['log_var'], targets
                )
                
                loss = loss_dict['total']
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.traj_model.parameters(), 
                        self.config.gradient_clip
                    )
                    optimizer.step()
                    train_losses.append(loss.item())
            
            self.traj_model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['trajectory_params'].to(self.device)
                    
                    output = self.traj_model(features)
                    loss_dict = self.traj_loss_fn(
                        output['mean'], output['log_var'], targets
                    )
                    
                    if not torch.isnan(loss_dict['total']):
                        val_losses.append(loss_dict['total'].item())
            
            train_loss = np.mean(train_losses) if train_losses else float('nan')
            val_loss = np.mean(val_losses) if val_losses else float('nan')
            
            self.history['traj_train_loss'].append(train_loss)
            self.history['traj_val_loss'].append(val_loss)
            
            scheduler.step()
            
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.traj_model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                break
        
        if best_state is not None:
            self.traj_model.load_state_dict(best_state)
        
        return {'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
    
    def train_survival_model(self,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            num_epochs: int = None) -> Dict:
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        optimizer = optim.AdamW(
            self.surv_model.parameters(),
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
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                if not loss.requires_grad:
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.surv_model.parameters(),
                    self.config.gradient_clip
                )
                optimizer.step()
                train_losses.append(loss.item())
            
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
            
            train_loss = np.mean(train_losses) if train_losses else float('nan')
            val_loss = np.mean(val_losses) if val_losses else float('nan')
            
            self.history['surv_train_loss'].append(train_loss)
            self.history['surv_val_loss'].append(val_loss)
            
            scheduler.step()
            
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.surv_model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                break
        
        if best_state is not None:
            self.surv_model.load_state_dict(best_state)
        
        return {'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate both models on test set."""
        self.traj_model.eval()
        self.surv_model.eval()
        
        all_features = []
        all_traj_targets = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in test_loader:
                all_features.append(batch['features'])
                all_traj_targets.append(batch['trajectory_params'])
                all_times.append(batch['time'])
                all_events.append(batch['event'])
        
        features = torch.cat(all_features).to(self.device)
        traj_targets = torch.cat(all_traj_targets).numpy()
        times = torch.cat(all_times).numpy()
        events = torch.cat(all_events).numpy()
        
        results = {}
        
        # Trajectory evaluation
        traj_output = self.traj_model.predict_with_uncertainty(
            features, n_samples=self.config.mc_dropout_samples
        )
        
        traj_pred = traj_output['mean'].cpu().numpy()
        traj_std = traj_output['total_std'].cpu().numpy()
        
        traj_metrics = PROGRESSMetrics.trajectory_metrics(
            traj_targets, traj_pred, traj_std
        )
        results['trajectory'] = traj_metrics
        
        # Survival evaluation
        self.surv_model.eval()
        with torch.no_grad():
            risk_scores = self.surv_model(features).cpu().numpy().squeeze()
        
        c_index = PROGRESSMetrics.concordance_index(risk_scores, times, events)
        results['survival'] = {'c_index': c_index}
        
        for horizon in self.config.survival_horizons:
            if times.max() >= horizon:
                auc = PROGRESSMetrics.time_dependent_auc(
                    risk_scores, times, events, horizon
                )
                results['survival'][f'AUC_{horizon:.0f}yr'] = auc
        
        return results
    
    def reset_history(self):
        """Reset training history for new experiment."""
        self.history = {
            'traj_train_loss': [], 'traj_val_loss': [],
            'surv_train_loss': [], 'surv_val_loss': [],
        }


# =============================================================================
# HIDDEN WIDTH EXPERIMENT
# =============================================================================

@dataclass
class HiddenWidthExperimentConfig:
    """Configuration for hidden layer width experiments."""
    
    # Widths to test
    widths: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Architecture patterns (relative to base width)
    # e.g., [1.0, 0.5, 0.25] means [width, width/2, width/4]
    traj_pattern: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    surv_pattern: List[float] = field(default_factory=lambda: [1.0, 0.5])
    
    # Number of runs per configuration (for statistical significance)
    n_runs: int = 3
    
    # Random seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


def generate_hidden_dims(base_width: int, pattern: List[float]) -> List[int]:
    """Generate hidden dimensions based on base width and pattern."""
    return [max(8, int(base_width * p)) for p in pattern]


def run_hidden_width_experiment(
    data_dir: str,
    output_dir: str,
    base_config: PROGRESSConfig,
    exp_config: HiddenWidthExperimentConfig
) -> Dict:
    """
    Run experiments testing different hidden layer widths.
    
    Args:
        data_dir: Directory containing data files
        output_dir: Output directory for results
        base_config: Base configuration
        exp_config: Experiment configuration
        
    Returns:
        Dictionary with all experimental results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("HIDDEN LAYER WIDTH SENSITIVITY EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Testing widths: {exp_config.widths}")
    logger.info(f"Runs per width: {exp_config.n_runs}")
    logger.info(f"Trajectory pattern: {exp_config.traj_pattern}")
    logger.info(f"Survival pattern: {exp_config.surv_pattern}")
    
    # Load data
    integrated_path = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    if not os.path.exists(integrated_path):
        raise FileNotFoundError(f"Dataset not found: {integrated_path}")
    
    integrated_data = pd.read_pickle(integrated_path)
    logger.info(f"Loaded dataset: {len(integrated_data)} subjects")
    
    # Create dataset
    full_dataset = PROGRESSDataset(
        integrated_data=integrated_data,
        fit_scaler=True,
        config=base_config
    )
    
    n_samples = len(full_dataset)
    input_dim = full_dataset.X.shape[1]
    events = full_dataset.survival_data['events']
    
    # Store all results
    all_results = {
        'experiment_config': {
            'widths': exp_config.widths,
            'traj_pattern': exp_config.traj_pattern,
            'surv_pattern': exp_config.surv_pattern,
            'n_runs': exp_config.n_runs,
            'seeds': exp_config.seeds
        },
        'base_config': base_config.to_dict(),
        'results_by_width': {}
    }
    
    # Run experiments for each width
    for width in exp_config.widths:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"TESTING WIDTH: {width}")
        logger.info("=" * 70)
        
        # Generate hidden dimensions
        traj_hidden = generate_hidden_dims(width, exp_config.traj_pattern)
        surv_hidden = generate_hidden_dims(width, exp_config.surv_pattern)
        
        logger.info(f"Trajectory hidden dims: {traj_hidden}")
        logger.info(f"Survival hidden dims: {surv_hidden}")
        
        width_results = {
            'traj_hidden_dims': traj_hidden,
            'surv_hidden_dims': surv_hidden,
            'runs': []
        }
        
        # Multiple runs for statistical significance
        for run_idx in range(exp_config.n_runs):
            seed = exp_config.seeds[run_idx % len(exp_config.seeds)]
            
            logger.info(f"\n--- Run {run_idx + 1}/{exp_config.n_runs} (seed={seed}) ---")
            
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Split data
            indices = np.arange(n_samples)
            train_idx, test_idx = train_test_split(
                indices, test_size=0.15, random_state=seed, stratify=events
            )
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.15, random_state=seed, stratify=events[train_idx]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=base_config.batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=base_config.batch_size,
                shuffle=False
            )
            test_loader = DataLoader(
                Subset(full_dataset, test_idx),
                batch_size=base_config.batch_size,
                shuffle=False
            )
            
            # Initialize trainer and models
            trainer = PROGRESSTrainer(base_config)
            traj_params, surv_params = trainer.setup_models(
                input_dim=input_dim,
                traj_hidden_dims=traj_hidden,
                surv_hidden_dims=surv_hidden
            )
            
            logger.info(f"Trajectory model params: {traj_params:,}")
            logger.info(f"Survival model params: {surv_params:,}")
            
            # Train trajectory model
            logger.info("Training trajectory model...")
            traj_train_result = trainer.train_trajectory_model(train_loader, val_loader)
            
            # Train survival model
            logger.info("Training survival model...")
            surv_train_result = trainer.train_survival_model(train_loader, val_loader)
            
            # Evaluate
            logger.info("Evaluating...")
            test_results = trainer.evaluate(test_loader)
            
            # Store run results
            run_result = {
                'seed': seed,
                'traj_params': traj_params,
                'surv_params': surv_params,
                'traj_train': traj_train_result,
                'surv_train': surv_train_result,
                'test_metrics': test_results
            }
            
            width_results['runs'].append(run_result)
            
            # Log key metrics
            logger.info(f"  Trajectory - Intercept R²: {test_results['trajectory']['intercept_R2']:.4f}")
            logger.info(f"  Trajectory - Slope R²: {test_results['trajectory']['slope_R2']:.4f}")
            logger.info(f"  Survival - C-index: {test_results['survival']['c_index']:.4f}")
            
            # Reset trainer history for next run
            trainer.reset_history()
        
        # Aggregate results across runs
        width_results['summary'] = aggregate_run_results(width_results['runs'])
        all_results['results_by_width'][width] = width_results
        
        logger.info(f"\nWidth {width} Summary:")
        logger.info(f"  Mean C-index: {width_results['summary']['c_index_mean']:.4f} ± {width_results['summary']['c_index_std']:.4f}")
        logger.info(f"  Mean Intercept R²: {width_results['summary']['intercept_R2_mean']:.4f} ± {width_results['summary']['intercept_R2_std']:.4f}")
    
    # Generate comparison plots
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING COMPARISON PLOTS")
    logger.info("=" * 70)
    
    plot_hidden_width_comparison(all_results, output_dir)
    
    # Save results
    results_path = os.path.join(output_dir, 'hidden_width_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_path}")
    
    # Generate summary table
    summary_table = generate_summary_table(all_results)
    table_path = os.path.join(output_dir, 'hidden_width_summary.csv')
    summary_table.to_csv(table_path, index=False)
    logger.info(f"Summary table saved to: {table_path}")
    
    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    print_summary_table(summary_table)
    
    return all_results


def aggregate_run_results(runs: List[Dict]) -> Dict:
    """Aggregate metrics across multiple runs."""
    
    # Collect metrics
    c_indices = []
    intercept_r2s = []
    slope_r2s = []
    accel_r2s = []
    intercept_picps = []
    slope_picps = []
    auc_2yrs = []
    auc_3yrs = []
    auc_5yrs = []
    
    for run in runs:
        metrics = run['test_metrics']
        
        c_indices.append(metrics['survival']['c_index'])
        intercept_r2s.append(metrics['trajectory']['intercept_R2'])
        slope_r2s.append(metrics['trajectory']['slope_R2'])
        accel_r2s.append(metrics['trajectory']['acceleration_R2'])
        
        if 'intercept_PICP' in metrics['trajectory']:
            intercept_picps.append(metrics['trajectory']['intercept_PICP'])
        if 'slope_PICP' in metrics['trajectory']:
            slope_picps.append(metrics['trajectory']['slope_PICP'])
        
        if 'AUC_2yr' in metrics['survival']:
            auc_2yrs.append(metrics['survival']['AUC_2yr'])
        if 'AUC_3yr' in metrics['survival']:
            auc_3yrs.append(metrics['survival']['AUC_3yr'])
        if 'AUC_5yr' in metrics['survival']:
            auc_5yrs.append(metrics['survival']['AUC_5yr'])
    
    summary = {
        'c_index_mean': np.mean(c_indices),
        'c_index_std': np.std(c_indices),
        'c_index_values': c_indices,
        
        'intercept_R2_mean': np.mean(intercept_r2s),
        'intercept_R2_std': np.std(intercept_r2s),
        'intercept_R2_values': intercept_r2s,
        
        'slope_R2_mean': np.mean(slope_r2s),
        'slope_R2_std': np.std(slope_r2s),
        'slope_R2_values': slope_r2s,
        
        'acceleration_R2_mean': np.mean(accel_r2s),
        'acceleration_R2_std': np.std(accel_r2s),
        
        'n_runs': len(runs)
    }
    
    if intercept_picps:
        summary['intercept_PICP_mean'] = np.mean(intercept_picps)
        summary['intercept_PICP_std'] = np.std(intercept_picps)
    
    if slope_picps:
        summary['slope_PICP_mean'] = np.mean(slope_picps)
        summary['slope_PICP_std'] = np.std(slope_picps)
    
    if auc_2yrs:
        summary['AUC_2yr_mean'] = np.mean(auc_2yrs)
        summary['AUC_2yr_std'] = np.std(auc_2yrs)
    
    if auc_3yrs:
        summary['AUC_3yr_mean'] = np.mean(auc_3yrs)
        summary['AUC_3yr_std'] = np.std(auc_3yrs)
    
    if auc_5yrs:
        summary['AUC_5yr_mean'] = np.mean(auc_5yrs)
        summary['AUC_5yr_std'] = np.std(auc_5yrs)
    
    return summary


def plot_hidden_width_comparison(results: Dict, output_dir: str):
    """Generate comparison plots for hidden width experiment."""
    
    widths = list(results['results_by_width'].keys())
    widths_sorted = sorted(widths)
    
    # Extract metrics for plotting
    c_index_means = []
    c_index_stds = []
    intercept_r2_means = []
    intercept_r2_stds = []
    slope_r2_means = []
    slope_r2_stds = []
    auc_3yr_means = []
    auc_3yr_stds = []
    model_params = []
    
    for w in widths_sorted:
        summary = results['results_by_width'][w]['summary']
        c_index_means.append(summary['c_index_mean'])
        c_index_stds.append(summary['c_index_std'])
        intercept_r2_means.append(summary['intercept_R2_mean'])
        intercept_r2_stds.append(summary['intercept_R2_std'])
        slope_r2_means.append(summary['slope_R2_mean'])
        slope_r2_stds.append(summary['slope_R2_std'])
        
        if 'AUC_3yr_mean' in summary:
            auc_3yr_means.append(summary['AUC_3yr_mean'])
            auc_3yr_stds.append(summary['AUC_3yr_std'])
        
        # Get model params from first run
        model_params.append(results['results_by_width'][w]['runs'][0]['traj_params'] + 
                          results['results_by_width'][w]['runs'][0]['surv_params'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: C-index vs Width
    ax = axes[0, 0]
    ax.errorbar(widths_sorted, c_index_means, yerr=c_index_stds, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.set_xlabel('Hidden Layer Width', fontsize=12)
    ax.set_ylabel('C-index', fontsize=12)
    ax.set_title('Survival Prediction: C-index vs Hidden Width', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(widths_sorted)
    
    # Plot 2: Trajectory R² vs Width
    ax = axes[0, 1]
    ax.errorbar(widths_sorted, intercept_r2_means, yerr=intercept_r2_stds,
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, label='Intercept')
    ax.errorbar(widths_sorted, slope_r2_means, yerr=slope_r2_stds,
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, label='Slope')
    ax.set_xlabel('Hidden Layer Width', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Trajectory Prediction: R² vs Hidden Width', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(widths_sorted)
    
    # Plot 3: AUC at 3yr vs Width (if available)
    ax = axes[1, 0]
    if auc_3yr_means:
        ax.errorbar(widths_sorted, auc_3yr_means, yerr=auc_3yr_stds,
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
        ax.set_ylabel('AUC at 3 Years', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'AUC data not available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Hidden Layer Width', fontsize=12)
    ax.set_title('Time-Dependent AUC (3yr) vs Hidden Width', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(widths_sorted)
    
    # Plot 4: Model Parameters vs Width
    ax = axes[1, 1]
    ax.bar(range(len(widths_sorted)), model_params, color='steelblue', alpha=0.7)
    ax.set_xlabel('Hidden Layer Width', fontsize=12)
    ax.set_ylabel('Total Parameters', fontsize=12)
    ax.set_title('Model Complexity vs Hidden Width', fontsize=14)
    ax.set_xticks(range(len(widths_sorted)))
    ax.set_xticklabels(widths_sorted)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add parameter counts as text
    for i, p in enumerate(model_params):
        ax.text(i, p + 500, f'{p:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'hidden_width_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plot saved: {fig_path}")
    
    # Create detailed box plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # C-index box plot
    ax = axes[0]
    c_index_data = [results['results_by_width'][w]['summary']['c_index_values'] for w in widths_sorted]
    bp = ax.boxplot(c_index_data, labels=widths_sorted, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Hidden Layer Width')
    ax.set_ylabel('C-index')
    ax.set_title('C-index Distribution by Width')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Intercept R² box plot
    ax = axes[1]
    r2_data = [results['results_by_width'][w]['summary']['intercept_R2_values'] for w in widths_sorted]
    bp = ax.boxplot(r2_data, labels=widths_sorted, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax.set_xlabel('Hidden Layer Width')
    ax.set_ylabel('R²')
    ax.set_title('Intercept R² Distribution by Width')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Slope R² box plot
    ax = axes[2]
    slope_data = [results['results_by_width'][w]['summary']['slope_R2_values'] for w in widths_sorted]
    bp = ax.boxplot(slope_data, labels=widths_sorted, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightyellow')
    ax.set_xlabel('Hidden Layer Width')
    ax.set_ylabel('R²')
    ax.set_title('Slope R² Distribution by Width')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    box_path = os.path.join(output_dir, 'hidden_width_boxplots.png')
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Box plots saved: {box_path}")


def generate_summary_table(results: Dict) -> pd.DataFrame:
    """Generate summary table for paper."""
    
    rows = []
    widths = sorted(results['results_by_width'].keys())
    
    for w in widths:
        data = results['results_by_width'][w]
        summary = data['summary']
        
        row = {
            'Width': w,
            'Traj Hidden Dims': str(data['traj_hidden_dims']),
            'Surv Hidden Dims': str(data['surv_hidden_dims']),
            'Total Params': data['runs'][0]['traj_params'] + data['runs'][0]['surv_params'],
            'C-index': f"{summary['c_index_mean']:.3f} ± {summary['c_index_std']:.3f}",
            'C-index Mean': summary['c_index_mean'],
            'Intercept R²': f"{summary['intercept_R2_mean']:.3f} ± {summary['intercept_R2_std']:.3f}",
            'Intercept R² Mean': summary['intercept_R2_mean'],
            'Slope R²': f"{summary['slope_R2_mean']:.3f} ± {summary['slope_R2_std']:.3f}",
            'Slope R² Mean': summary['slope_R2_mean'],
        }
        
        if 'AUC_3yr_mean' in summary:
            row['AUC 3yr'] = f"{summary['AUC_3yr_mean']:.3f} ± {summary['AUC_3yr_std']:.3f}"
            row['AUC 3yr Mean'] = summary['AUC_3yr_mean']
        
        if 'intercept_PICP_mean' in summary:
            row['Intercept PICP'] = f"{summary['intercept_PICP_mean']*100:.1f}%"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """Print summary table in readable format."""
    
    # Select columns for display
    display_cols = ['Width', 'Total Params', 'C-index', 'Intercept R²', 'Slope R²']
    if 'AUC 3yr' in df.columns:
        display_cols.append('AUC 3yr')
    
    print("\n" + "=" * 80)
    print("HIDDEN LAYER WIDTH SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    
    # Print header
    header = f"{'Width':>8} | {'Params':>10} | {'C-index':>18} | {'Intercept R²':>18} | {'Slope R²':>18}"
    print(header)
    print("-" * len(header))
    
    # Find best values for highlighting
    best_c_index = df['C-index Mean'].max()
    best_intercept = df['Intercept R² Mean'].max()
    best_slope = df['Slope R² Mean'].max()
    
    for _, row in df.iterrows():
        c_idx_str = row['C-index']
        int_r2_str = row['Intercept R²']
        slope_r2_str = row['Slope R²']
        
        # Add indicator for best values
        if row['C-index Mean'] == best_c_index:
            c_idx_str += " *"
        if row['Intercept R² Mean'] == best_intercept:
            int_r2_str += " *"
        if row['Slope R² Mean'] == best_slope:
            slope_r2_str += " *"
        
        print(f"{row['Width']:>8} | {row['Total Params']:>10,} | {c_idx_str:>18} | {int_r2_str:>18} | {slope_r2_str:>18}")
    
    print("-" * len(header))
    print("* indicates best performance for that metric")
    print("=" * 80)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PROGRESS: Hidden Layer Width Sensitivity Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width
  python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width --widths 32 64 128 256
  python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width --n-runs 5
  python PROGRESS_experiments.py --data-dir ./dataset --experiment hidden-width --quick-test
        """
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
        help='Output directory (default: {data-dir}/experiments_output)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='hidden-width',
        choices=['hidden-width'],
        help='Type of experiment to run (default: hidden-width)'
    )
    
    parser.add_argument(
        '--widths',
        type=int,
        nargs='+',
        default=[32, 64, 128, 256],
        help='Hidden layer widths to test (default: 32 64 128 256)'
    )
    
    parser.add_argument(
        '--n-runs',
        type=int,
        default=3,
        help='Number of runs per configuration (default: 3)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience (default: 15)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with reduced epochs and runs'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'experiments_output')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    global logger
    logger = setup_logging(args.output_dir)
    
    # Create base configuration
    base_config = PROGRESSConfig(
        num_epochs=10 if args.quick_test else args.epochs,
        batch_size=args.batch_size,
        patience=5 if args.quick_test else args.patience,
        device=args.device
    )
    
    # Create experiment configuration
    exp_config = HiddenWidthExperimentConfig(
        widths=args.widths,
        n_runs=1 if args.quick_test else args.n_runs
    )
    
    # Run experiment
    if args.experiment == 'hidden-width':
        try:
            results = run_hidden_width_experiment(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                base_config=base_config,
                exp_config=exp_config
            )
            logger.info("\nExperiment completed successfully!")
            return 0
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
