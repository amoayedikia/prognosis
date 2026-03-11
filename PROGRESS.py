#!/usr/bin/env python3
"""
PROGRESS: PRognostic Generalization from REsting Static Signatures

A dual-model framework for Alzheimer's Disease progression prediction from
baseline CSF biomarkers, as described in the PROGRESS paper.

================================================================================
FRAMEWORK OVERVIEW (from Methods Section 2)
================================================================================

The PROGRESS framework addresses the clinical challenge of predicting individual 
AD progression trajectories using only baseline (first-visit) CSF biomarkers.

Model 1: Probabilistic Trajectory Parameter Network (Section 2.3)
    - Input: Baseline CSF biomarkers (AÎ²42, p-tau, t-tau) + demographics
    - Output: Distribution of trajectory parameters (intercept Î±, slope Î², acceleration Î³)
    - Key features:
        * Heteroscedastic uncertainty estimation (Equation 6)
        * Biomarker attention mechanism for interpretability
        * Predicts full cognitive decline dynamics

Model 2: Deep Survival Analysis Network (Section 2.4)
    - Input: Same baseline features
    - Output: Time-to-conversion risk scores
    - Key features:
        * Cox proportional hazards with neural risk function
        * Calibrated survival curves
        * Time-dependent AUC evaluation

Training Protocol (Section 2.5):
    - Nested 5Ã—5 cross-validation for unbiased evaluation
    - AdamW optimizer with cosine annealing
    - Multi-component loss with calibration regularization

================================================================================
USAGE
================================================================================

# Basic usage with default settings:
python PROGRESS.py --data-dir ./dataset

# Full training with custom parameters:
python PROGRESS.py --data-dir ./dataset --epochs 100 --batch-size 32 --lr 1e-3

# Quick test run:
python PROGRESS.py --data-dir ./dataset --epochs 10 --quick-test

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
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # File handler
    log_file = os.path.join(output_dir, 'progress_training.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
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
    """
    Configuration for PROGRESS framework.
    
    All hyperparameters from paper Section 2.5.
    """
    
    # === Data Parameters ===
    sequence_length: int = 5              # L: visits per sequence
    min_visits_trajectory: int = 3        # Minimum visits for trajectory fitting
    
    # === Model 1: Trajectory Network Architecture ===
    traj_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    traj_dropout: float = 0.3
    traj_attention_heads: int = 4
    traj_use_batch_norm: bool = True
    
    # === Model 2: Survival Network Architecture ===
    surv_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    surv_dropout: float = 0.3
    surv_use_batch_norm: bool = True
    
    # === Training Parameters (Section 2.5) ===
    learning_rate: float = 1e-3           # Initial learning rate
    weight_decay: float = 1e-4            # L2 regularization
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15                    # Early stopping patience
    gradient_clip: float = 1.0            # Gradient clipping threshold
    
    # === Loss Weights (Equation 7) ===
    nll_weight: float = 1.0               # Î»â‚: NLL weight
    calibration_weight: float = 0.1       # Î»â‚‚: Calibration loss weight
    ranking_weight: float = 0.5           # Î»â‚ƒ: Ranking loss weight (survival)
    
    # === Uncertainty Estimation ===
    mc_dropout_samples: int = 50          # T: MC dropout samples
    
    # === Cross-Validation (Section 2.5) ===
    n_outer_folds: int = 5                # Outer CV folds
    n_inner_folds: int = 5                # Inner CV folds (hyperparameter tuning)
    
    # === Evaluation ===
    survival_horizons: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    
    # === Device ===
    device: str = 'auto'                  # 'auto', 'cuda', 'cpu', or 'mps'
    
    def get_device(self) -> torch.device:
        """Get PyTorch device based on configuration."""
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.device)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            k: v if not isinstance(v, list) else v.copy()
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PROGRESSConfig':
        """Create from dictionary."""
        return cls(**d)


# =============================================================================
# DATA PROCESSING
# =============================================================================

class PROGRESSDataset(Dataset):
    """
    Dataset for PROGRESS framework.
    
    Processes integrated NACC data for both:
    - Model 1: Trajectory parameter prediction
    - Model 2: Survival analysis
    
    From paper Section 2.2 (Feature Engineering):
    Static features x_i include:
    - CSF biomarkers: AÎ²42, p-tau, t-tau (harmonized)
    - Derived ratios: p-tau/AÎ²42, t-tau/p-tau
    - Demographics: age, sex, education
    - Baseline cognition: MMSE, CDR-SB
    """
    
    # NACC missing value codes
    MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}
    
    def __init__(self,
                 integrated_data: pd.DataFrame,
                 sequences_data: pd.DataFrame = None,
                 scaler: StandardScaler = None,
                 fit_scaler: bool = False,
                 config: PROGRESSConfig = None):
        """
        Initialize dataset.
        
        Args:
            integrated_data: Subject-level integrated dataset (from NACCDataIntegrator)
            sequences_data: ML sequences dataset (optional, for sequence features)
            scaler: Pre-fitted feature scaler
            fit_scaler: Whether to fit a new scaler
            config: Configuration object
        """
        self.config = config or PROGRESSConfig()
        self.integrated_data = integrated_data
        self.sequences_data = sequences_data
        
        # Get valid subjects
        self.subjects = self._get_valid_subjects()
        logger.info(f"Dataset: {len(self.subjects)} valid subjects")
        
        # Extract features and targets
        logger.info("Extracting baseline features...")
        self.baseline_features, self.feature_names = self._extract_baseline_features()
        
        logger.info("Computing trajectory parameters...")
        self.trajectory_params = self._compute_trajectory_parameters()
        
        logger.info("Extracting survival data...")
        self.survival_data = self._extract_survival_data()
        
        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()  # Robust to outliers in biomarkers
            self.baseline_features_scaled = self.scaler.fit_transform(self.baseline_features)
            logger.info("Fitted feature scaler")
        elif scaler is not None:
            self.scaler = scaler
            self.baseline_features_scaled = self.scaler.transform(self.baseline_features)
        else:
            self.scaler = None
            self.baseline_features_scaled = self.baseline_features
        
        # Convert to tensors
        self._create_tensors()
        
        # Log summary statistics
        self._log_summary()
    
    def _get_valid_subjects(self) -> List[str]:
        """Get subjects with valid data for both models."""
        valid_subjects = []
        
        for _, row in self.integrated_data.iterrows():
            naccid = row.get('NACCID')
            if naccid is None:
                continue
            
            # Check for essential CSF biomarkers
            abeta = row.get('ABETA_harm')
            ptau = row.get('PTAU_harm')
            ttau = row.get('TTAU_harm')
            
            # At least one biomarker must be present
            has_biomarker = any([
                self._is_valid(abeta),
                self._is_valid(ptau),
                self._is_valid(ttau)
            ])
            
            if not has_biomarker:
                continue
            
            # Check for clinical trajectory
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) >= 2:
                valid_subjects.append(naccid)
        
        return valid_subjects
    
    def _is_valid(self, value) -> bool:
        """Check if value is valid (not missing or NaN)."""
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
    
    def _extract_baseline_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Extract baseline feature vector for each subject.
        
        Features (Equation from paper Section 2.2):
        x_i = [AÎ²42, p-tau, t-tau, p-tau/AÎ²42, t-tau/p-tau, 
               age, sex, education, baseline_MMSE, baseline_CDR]
        """
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
            abeta = self._clean_value(row.get('ABETA_harm'), 500.0)  # Default to normal
            ptau = self._clean_value(row.get('PTAU_harm'), 50.0)
            ttau = self._clean_value(row.get('TTAU_harm'), 300.0)
            
            # Derived ratios (with safe division)
            if abeta > 0:
                ptau_abeta_ratio = ptau / abeta
            else:
                ptau_abeta_ratio = 0.1
            
            if ptau > 0:
                ttau_ptau_ratio = ttau / ptau
            else:
                ttau_ptau_ratio = 6.0
            
            # Demographics
            age = self._clean_value(row.get('AGE_AT_BASELINE'), 75.0)
            sex = self._clean_value(row.get('SEX'), 1.0)
            educ = self._clean_value(row.get('EDUC'), 16.0)
            
            # Baseline cognition from trajectory
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
        
        # Handle any remaining NaN with column medians
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
        """
        Compute trajectory parameters using quadratic regression.
        
        From paper Equation 4:
        y_{ij} = (Î±â‚€ + Î±_i) + (Î²â‚€ + Î²_i)t_{ij} + (Î³â‚€ + Î³_i)tÂ²_{ij} + Îµ_{ij}
        
        We fit individual-level parameters:
        - Î±_i: intercept (baseline severity)
        - Î²_i: slope (rate of decline)
        - Î³_i: acceleration (change in rate)
        
        Returns:
            Array of shape (n_subjects, 3) with [intercept, slope, acceleration]
        """
        trajectory_params = []
        valid_trajectory_count = 0
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            trajectory = row.get('clinical_trajectory', [])
            
            if not isinstance(trajectory, list) or len(trajectory) < self.config.min_visits_trajectory:
                # Insufficient data - use defaults
                trajectory_params.append([np.nan, np.nan, np.nan])
                continue
            
            # Extract time and CDR-SB values
            times = []
            cdr_values = []
            
            for visit in trajectory:
                t = visit.get('YearsFromBaseline')
                cdr = visit.get('CDRSUM')
                
                if self._is_valid(t) and self._is_valid(cdr):
                    # Validate CDR range
                    cdr = float(cdr)
                    if 0 <= cdr <= 18:
                        times.append(float(t))
                        cdr_values.append(cdr)
            
            if len(times) >= self.config.min_visits_trajectory:
                times = np.array(times)
                cdr_values = np.array(cdr_values)
                
                try:
                    # Fit quadratic: y = Î± + Î²t + Î³tÂ²
                    # Using numpy polyfit (returns [Î³, Î², Î±] for degree=2)
                    coeffs = np.polyfit(times, cdr_values, deg=2)
                    
                    # Reorder to [intercept, slope, acceleration]
                    alpha = coeffs[2]  # Intercept
                    beta = coeffs[1]   # Slope
                    gamma = coeffs[0]  # Acceleration
                    
                    # Validate parameters are reasonable
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
        
        # Impute missing with population medians
        for col in range(params_array.shape[1]):
            col_data = params_array[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                if np.isnan(median_val):
                    # Reasonable defaults
                    defaults = [1.0, 0.3, 0.02]  # [intercept, slope, acceleration]
                    median_val = defaults[col]
                params_array[mask, col] = median_val
        
        logger.info(f"  Valid trajectory fits: {valid_trajectory_count}/{len(self.subjects)}")
        logger.info(f"  Intercept (Î±): mean={params_array[:, 0].mean():.2f}, std={params_array[:, 0].std():.2f}")
        logger.info(f"  Slope (Î²): mean={params_array[:, 1].mean():.3f}, std={params_array[:, 1].std():.3f}")
        logger.info(f"  Acceleration (Î³): mean={params_array[:, 2].mean():.4f}, std={params_array[:, 2].std():.4f}")
        
        return params_array
    
    def _extract_survival_data(self) -> Dict[str, np.ndarray]:
        """
        Extract survival/time-to-event data for Model 2.
        
        From paper Section 2.4:
        - Event: Conversion from MCI to dementia
        - Time: Years until conversion or last follow-up (censoring)
        """
        times = []
        events = []
        
        for naccid in self.subjects:
            row = self.integrated_data[
                self.integrated_data['NACCID'] == naccid
            ].iloc[0]
            
            # Check for dementia conversion
            converted = row.get('converted_to_dementia', 0)
            
            if converted == 1:
                # Event occurred
                time_to_event = row.get('time_to_dementia')
                if not self._is_valid(time_to_event):
                    time_to_event = row.get('follow_up_years', 5.0)
                times.append(float(time_to_event))
                events.append(1)
            else:
                # Censored - use last follow-up time
                follow_up = row.get('follow_up_years')
                if not self._is_valid(follow_up):
                    # Estimate from trajectory
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
        
        # Handle missing times
        mask = np.isnan(times)
        if mask.any():
            times[mask] = np.nanmedian(times)
        
        # Ensure positive times
        times = np.maximum(times, 0.1)
        
        event_rate = events.sum() / len(events) * 100
        logger.info(f"  Events: {events.sum()}/{len(events)} ({event_rate:.1f}%)")
        logger.info(f"  Median follow-up: {np.median(times):.1f} years")
        
        return {'times': times, 'events': events}
    
    def _create_tensors(self):
        """Convert numpy arrays to PyTorch tensors."""
        self.X = torch.FloatTensor(self.baseline_features_scaled)
        self.Y_traj = torch.FloatTensor(self.trajectory_params)
        self.T = torch.FloatTensor(self.survival_data['times'])
        self.E = torch.LongTensor(self.survival_data['events'])
    
    def _log_summary(self):
        """Log dataset summary statistics."""
        logger.info("Dataset Summary:")
        logger.info(f"  Subjects: {len(self.subjects)}")
        logger.info(f"  Features: {self.X.shape[1]}")
        logger.info(f"  Feature names: {self.feature_names}")
    
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
        """Get list of subject IDs."""
        return self.subjects.copy()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names.copy()


# =============================================================================
# MODEL 1: TRAJECTORY PARAMETER NETWORK
# =============================================================================

class BiomarkerAttention(nn.Module):
    """
    Biomarker attention mechanism for interpretability.
    
    From paper Section 2.3.2:
    Learns which biomarkers are most predictive for each subject,
    providing clinical interpretability.
    
    a = softmax(W_QÂ·x Â· (W_KÂ·x)áµ€ / âˆšd_k) Â· W_VÂ·x
    """
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = max(1, input_dim // num_heads)
        self.scaled_dim = self.head_dim * num_heads
        
        # Project to scaled dimension
        self.input_proj = nn.Linear(input_dim, self.scaled_dim)
        
        # Query, Key, Value projections
        self.W_Q = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.W_K = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.W_V = nn.Linear(self.scaled_dim, self.scaled_dim)
        
        # Output projection back to input dim
        self.W_O = nn.Linear(self.scaled_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size = x.size(0)
        
        # Project input
        x_proj = self.input_proj(x)
        
        # Compute Q, K, V
        Q = self.W_Q(x_proj)
        K = self.W_K(x_proj)
        V = self.W_V(x_proj)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.bmm(attention_weights, V)
        
        # Reshape and project output
        attended = attended.view(batch_size, self.scaled_dim)
        output = self.W_O(attended)
        
        # Residual connection
        output = output + x
        
        return output, attention_weights.mean(dim=1)  # Average across heads


class TrajectoryParameterNetwork(nn.Module):
    """
    Probabilistic Trajectory Parameter Network (Model 1).
    
    From paper Section 2.3:
    f_Î¸: x â†’ (Î¼_Î¸, ÏƒÂ²_Î¸)
    
    Predicts distribution of trajectory parameters:
    - Î¼ = [Î¼_Î±, Î¼_Î², Î¼_Î³]: Mean of intercept, slope, acceleration
    - ÏƒÂ² = [ÏƒÂ²_Î±, ÏƒÂ²_Î², ÏƒÂ²_Î³]: Variance (aleatoric uncertainty)
    
    Uses heteroscedastic regression for uncertainty-aware predictions.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3,
                 num_attention_heads: int = 4,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = 3  # [intercept, slope, acceleration]
        
        # Biomarker attention
        self.attention = BiomarkerAttention(input_dim, num_attention_heads, dropout)
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())  # GELU activation (paper spec)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads for mean and log-variance
        # Separate heads for each parameter (better calibration)
        self.mean_head = nn.Linear(hidden_dims[-1], self.output_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], self.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization with small gains for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Baseline features (batch_size, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with:
            - 'mean': Predicted means (batch_size, 3)
            - 'log_var': Log variances (batch_size, 3)
            - 'std': Standard deviations (batch_size, 3)
            - 'attention': Attention weights (optional)
        """
        # Apply attention
        attended, attention_weights = self.attention(x)
        
        # Encode
        h = self.encoder(attended)
        
        # Predict mean and log-variance
        mean = self.mean_head(h)
        log_var = self.logvar_head(h)
        
        # Clamp log_var for numerical stability
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
        """
        Monte Carlo dropout for epistemic uncertainty estimation.
        
        From paper Section 2.3.3:
        Total uncertainty = Aleatoric (data noise) + Epistemic (model uncertainty)
        
        ÏƒÂ²_total = ÏƒÂ²_aleatoric + ÏƒÂ²_epistemic
        
        Args:
            x: Input features
            n_samples: Number of MC dropout samples
            
        Returns:
            Dictionary with mean and uncertainty estimates
        """
        self.train()  # Enable dropout
        
        all_means = []
        all_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                all_means.append(output['mean'])
                all_vars.append(torch.exp(output['log_var']))
        
        all_means = torch.stack(all_means)  # (n_samples, batch, 3)
        all_vars = torch.stack(all_vars)    # (n_samples, batch, 3)
        
        # Mean prediction (ensemble)
        mean_pred = all_means.mean(dim=0)
        
        # Aleatoric uncertainty (mean of predicted variances)
        aleatoric_var = all_vars.mean(dim=0)
        
        # Epistemic uncertainty (variance of means)
        epistemic_var = all_means.var(dim=0)
        
        # Total uncertainty
        total_var = aleatoric_var + epistemic_var
        
        self.eval()
        
        return {
            'mean': mean_pred,
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'epistemic_std': torch.sqrt(epistemic_var),
            'total_std': torch.sqrt(total_var)
        }


# =============================================================================
# MODEL 2: DEEP SURVIVAL NETWORK
# =============================================================================

class DeepSurvivalNetwork(nn.Module):
    """
    Deep Survival Network (Model 2).
    
    From paper Section 2.4:
    Implements Cox proportional hazards with neural network risk function:
    
    h(t|x) = hâ‚€(t) Â· exp(Ïˆ_Ï†(x))
    
    Where Ïˆ_Ï† is a neural network mapping baseline features to log-risk.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build risk network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())  # ReLU preserves proportional hazards interpretation
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final log-risk output (scalar)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.risk_network = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-risk score Ïˆ_Ï†(x).
        
        Args:
            x: Baseline features (batch_size, input_dim)
            
        Returns:
            Log-risk scores (batch_size, 1)
        """
        return self.risk_network(x)
    
    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get risk scores (exponentiated).
        
        Returns:
            Risk scores exp(Ïˆ_Ï†(x))
        """
        return torch.exp(self.forward(x))
    
    def predict_survival_function(self,
                                  x: torch.Tensor,
                                  times: np.ndarray,
                                  baseline_hazard: np.ndarray) -> np.ndarray:
        """
        Predict survival function S(t|x) = exp(-Hâ‚€(t)Â·exp(Ïˆ(x))).
        
        Args:
            x: Features
            times: Time points for evaluation
            baseline_hazard: Baseline cumulative hazard Hâ‚€(t)
            
        Returns:
            Survival probabilities (n_subjects, n_times)
        """
        self.eval()
        with torch.no_grad():
            log_risk = self.forward(x).cpu().numpy().squeeze()
        
        risk = np.exp(log_risk)
        
        # S(t|x) = exp(-Hâ‚€(t) Â· exp(Ïˆ(x)))
        survival = np.exp(-np.outer(risk, baseline_hazard))
        
        return survival


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class TrajectoryLoss(nn.Module):
    """
    Loss function for trajectory parameter prediction.
    
    From paper Equation 7:
    L_traj = L_NLL + Î»â‚||Î˜||Â² + Î»â‚‚L_calibration
    
    Where:
    - L_NLL: Negative log-likelihood (heteroscedastic Gaussian)
    - L_calibration: Prediction interval coverage loss
    """
    
    def __init__(self, calibration_weight: float = 0.1):
        super().__init__()
        self.calibration_weight = calibration_weight
    
    def forward(self,
                pred_mean: torch.Tensor,
                pred_log_var: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory loss.
        
        Args:
            pred_mean: Predicted means (batch, 3)
            pred_log_var: Predicted log variances (batch, 3)
            targets: True trajectory parameters (batch, 3)
            
        Returns:
            Dictionary with loss components
        """
        # Heteroscedastic Gaussian NLL
        pred_var = torch.exp(pred_log_var)
        
        # NLL = 0.5 * [(y - Î¼)Â²/ÏƒÂ² + log(ÏƒÂ²)]
        nll = 0.5 * (
            (targets - pred_mean) ** 2 / (pred_var + 1e-8) + 
            pred_log_var
        ).mean()
        
        # Calibration loss: encourage 95% prediction interval coverage
        pred_std = torch.sqrt(pred_var + 1e-8)
        z_scores = torch.abs((targets - pred_mean) / pred_std)
        
        # Proportion within 95% CI (z < 1.96)
        within_95 = (z_scores < 1.96).float().mean()
        
        # Loss penalizes deviation from 95% coverage
        calibration_loss = (within_95 - 0.95) ** 2
        
        # Total loss
        total = nll + self.calibration_weight * calibration_loss
        
        # Also compute MSE for monitoring
        mse = F.mse_loss(pred_mean, targets)
        
        return {
            'total': total,
            'nll': nll,
            'calibration': calibration_loss,
            'coverage_95': within_95,
            'mse': mse
        }


class CoxPartialLikelihoodLoss(nn.Module):
    """
    Cox partial likelihood loss for survival analysis.
    
    From paper Equation 9:
    L_cox = -âˆ‘_{i:Î´áµ¢=1} [Ïˆ(xáµ¢) - log(âˆ‘_{jâˆˆR(táµ¢)} exp(Ïˆ(xâ±¼)))]
    
    Where R(táµ¢) is the risk set at time táµ¢.
    """
    
    def __init__(self, ranking_weight: float = 0.5):
        super().__init__()
        self.ranking_weight = ranking_weight
    
    def forward(self,
                risk_scores: torch.Tensor,
                times: torch.Tensor,
                events: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Cox partial likelihood loss.
        
        Args:
            risk_scores: Log-risk scores Ïˆ(x) (batch,)
            times: Survival/censoring times (batch,)
            events: Event indicators Î´ (batch,)
            
        Returns:
            Dictionary with loss components
        """
        # Flatten risk scores
        risk_scores = risk_scores.view(-1)
        
        # Sort by time (descending)
        sorted_indices = torch.argsort(times, descending=True)
        sorted_risks = risk_scores[sorted_indices]
        sorted_events = events[sorted_indices].float()
        
        # Compute cumulative sum of exp(risk) in log space for numerical stability
        # Using log-sum-exp trick
        max_risk = sorted_risks.max()
        exp_risks = torch.exp(sorted_risks - max_risk)
        cumsum_exp = torch.cumsum(exp_risks, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
        
        # Cox partial likelihood
        # For each event, we want: risk_i - log(sum of risks in risk set)
        # The risk set at time t_i includes all subjects with time >= t_i
        # Since we sorted descending, cumsum gives us the risk set
        
        # Compute log partial likelihood
        log_lik = sorted_risks - log_cumsum
        
        # Only count events (multiply by event indicator)
        # Add small term to ensure gradient flow even with few events
        n_events = sorted_events.sum()
        
        if n_events > 0:
            cox_loss = -(log_lik * sorted_events).sum() / (n_events + 1e-8)
        else:
            # No events - use mean risk as regularization to maintain gradients
            cox_loss = risk_scores.mean() * 0.0 + 0.1 * (risk_scores ** 2).mean()
        
        # Pairwise ranking loss (concordance)
        ranking_loss = self._ranking_loss(risk_scores, times, events)
        
        # Total loss
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
        """
        Pairwise ranking loss for concordance.
        
        For pairs (i,j) where táµ¢ < tâ±¼ and Î´áµ¢=1:
        We want risk_i > risk_j (higher risk = shorter survival)
        """
        n = len(times)
        
        if n < 2:
            # Return zero with gradient connection
            return risk_scores.sum() * 0.0
        
        risk_scores = risk_scores.view(-1)
        
        # Compute all pairwise differences
        risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)  # (n, n)
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)  # (n, n)
        
        # Valid pairs: i had event before j
        valid_pairs = (events.unsqueeze(1) == 1) & (time_diff < 0)
        n_valid = valid_pairs.sum().float()
        
        if n_valid == 0:
            # No valid pairs - return zero with gradient connection
            return risk_scores.sum() * 0.0
        
        # Ranking violation: should have risk_i > risk_j, penalize if risk_i < risk_j
        # Using sigmoid for smooth gradients instead of ReLU
        violations = torch.sigmoid(-risk_diff + margin)
        
        ranking_loss = (violations * valid_pairs.float()).sum() / (n_valid + 1e-8)
        
        return ranking_loss


# =============================================================================
# METRICS
# =============================================================================

class PROGRESSMetrics:
    """
    Evaluation metrics for PROGRESS framework.
    
    From paper Section 2.6.
    
    Includes:
    - Regression metrics for trajectory prediction
    - Survival metrics for time-to-event prediction
    - Classification metrics (F1, precision, recall, accuracy, AUC) for clinical decisions
    """
    
    # Clinical thresholds for classification
    SLOPE_THRESHOLD = 0.5  # CDR-SB points/year - fast vs slow progression
    RISK_PERCENTILE = 75   # High risk if above this percentile
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True binary labels (0/1)
            y_pred: Predicted binary labels (0/1)
            y_prob: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary with accuracy, precision, recall, F1, specificity, AUC
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score, average_precision_score
        )
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity (true negative rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Negative predictive value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Positive predictive value (same as precision)
        metrics['ppv'] = metrics['precision']
        
        # AUC metrics (if probabilities provided)
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics['auc_roc'] = 0.5
                metrics['auc_pr'] = y_true.mean()
        
        return metrics
    
    @staticmethod
    def progressor_classification(slope_true: np.ndarray,
                                  slope_pred: np.ndarray,
                                  slope_std: np.ndarray = None,
                                  threshold: float = 0.5) -> Dict[str, float]:
        """
        Classify patients as fast vs slow progressors based on predicted slope.
        
        Clinical definition:
        - Fast progressor: slope > threshold (rapid cognitive decline)
        - Slow progressor: slope <= threshold
        
        Args:
            slope_true: True slope values (Î² parameter)
            slope_pred: Predicted slope values
            slope_std: Predicted uncertainty (optional, for probability estimation)
            threshold: CDR-SB points/year threshold for fast progression
            
        Returns:
            Dictionary with classification metrics
        """
        # Binary classification: fast (1) vs slow (0) progressor
        y_true = (slope_true > threshold).astype(int)
        y_pred = (slope_pred > threshold).astype(int)
        
        # Compute probability of being fast progressor using uncertainty
        if slope_std is not None:
            # P(slope > threshold) using normal CDF
            from scipy.stats import norm
            y_prob = 1 - norm.cdf(threshold, loc=slope_pred, scale=slope_std + 1e-8)
        else:
            # Use distance from threshold as pseudo-probability
            y_prob = 1 / (1 + np.exp(-(slope_pred - threshold)))
        
        metrics = PROGRESSMetrics.classification_metrics(y_true, y_pred, y_prob)
        
        # Add prefix for clarity
        return {f'progressor_{k}': v for k, v in metrics.items()}
    
    @staticmethod
    def conversion_classification(times: np.ndarray,
                                  events: np.ndarray,
                                  risk_scores: np.ndarray,
                                  horizon: float = 3.0,
                                  risk_threshold: float = None) -> Dict[str, float]:
        """
        Classify patients by conversion risk within a time horizon.
        
        Clinical question: Will this patient convert to dementia within X years?
        
        Args:
            times: Observed times
            events: Event indicators (1 = converted)
            risk_scores: Predicted risk scores
            horizon: Time horizon in years
            risk_threshold: Risk score threshold (default: median)
            
        Returns:
            Dictionary with classification metrics
        """
        # True labels: converted within horizon
        y_true = ((times <= horizon) & (events == 1)).astype(int)
        
        # Set threshold if not provided (use median or optimal)
        if risk_threshold is None:
            risk_threshold = np.median(risk_scores)
        
        # Predicted labels: high risk
        y_pred = (risk_scores > risk_threshold).astype(int)
        
        # Use risk scores as probabilities (normalized)
        risk_min, risk_max = risk_scores.min(), risk_scores.max()
        if risk_max > risk_min:
            y_prob = (risk_scores - risk_min) / (risk_max - risk_min)
        else:
            y_prob = np.ones_like(risk_scores) * 0.5
        
        metrics = PROGRESSMetrics.classification_metrics(y_true, y_pred, y_prob)
        
        # Add horizon-specific prefix
        return {f'conversion_{horizon:.0f}yr_{k}': v for k, v in metrics.items()}
    
    @staticmethod
    def risk_stratification(risk_scores: np.ndarray,
                           times: np.ndarray,
                           events: np.ndarray,
                           n_groups: int = 3) -> Dict[str, Any]:
        """
        Stratify patients into risk groups and evaluate separation.
        
        Args:
            risk_scores: Predicted risk scores
            times: Observed times
            events: Event indicators
            n_groups: Number of risk groups (default: 3 = low/medium/high)
            
        Returns:
            Dictionary with stratification metrics
        """
        # Create risk groups based on percentiles
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
                group_event_rate = events[mask].mean()
                group_median_time = np.median(times[mask])
                metrics[f'group_{g}_event_rate'] = float(group_event_rate)
                metrics[f'group_{g}_median_time'] = float(group_median_time)
        
        # Log-rank test between groups (if scipy available)
        try:
            from scipy.stats import chi2
            # Simplified log-rank: compare high vs low risk groups
            high_risk = risk_groups == (n_groups - 1)
            low_risk = risk_groups == 0
            
            if high_risk.sum() > 0 and low_risk.sum() > 0:
                high_events = events[high_risk].sum()
                low_events = events[low_risk].sum()
                high_n = high_risk.sum()
                low_n = low_risk.sum()
                
                # Expected events under null hypothesis
                total_events = high_events + low_events
                total_n = high_n + low_n
                expected_high = total_events * high_n / total_n
                expected_low = total_events * low_n / total_n
                
                # Chi-square statistic
                if expected_high > 0 and expected_low > 0:
                    chi2_stat = ((high_events - expected_high) ** 2 / expected_high +
                                (low_events - expected_low) ** 2 / expected_low)
                    p_value = 1 - chi2.cdf(chi2_stat, df=1)
                    metrics['log_rank_chi2'] = float(chi2_stat)
                    metrics['log_rank_p_value'] = float(p_value)
        except ImportError:
            pass
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray,
                               y_scores: np.ndarray,
                               metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores/probabilities
            metric: Metric to optimize ('f1', 'youden', 'accuracy')
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        thresholds = np.percentile(y_scores, np.arange(5, 96, 5))
        best_threshold = np.median(y_scores)
        best_value = 0
        
        for thresh in thresholds:
            y_pred = (y_scores > thresh).astype(int)
            
            if metric == 'f1':
                value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J = sensitivity + specificity - 1
                tn = ((y_true == 0) & (y_pred == 0)).sum()
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                value = sens + spec - 1
            else:  # accuracy
                value = accuracy_score(y_true, y_pred)
            
            if value > best_value:
                best_value = value
                best_threshold = thresh
        
        return best_threshold, best_value
    
    @staticmethod
    def trajectory_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_std: np.ndarray = None) -> Dict[str, float]:
        """
        Compute metrics for trajectory parameter prediction.
        
        Metrics:
        - RMSE: Root mean squared error
        - MAE: Mean absolute error
        - RÂ²: Coefficient of determination
        - Correlation: Pearson correlation
        - PICP: Prediction interval coverage probability (if std provided)
        - MPIW: Mean prediction interval width (if std provided)
        """
        param_names = ['intercept', 'slope', 'acceleration']
        metrics = {}
        
        for i, name in enumerate(param_names):
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]
            
            # RMSE
            metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(true_i, pred_i))
            
            # MAE
            metrics[f'{name}_MAE'] = mean_absolute_error(true_i, pred_i)
            
            # RÂ²
            if np.var(true_i) > 0:
                metrics[f'{name}_R2'] = r2_score(true_i, pred_i)
            else:
                metrics[f'{name}_R2'] = 0.0
            
            # Correlation
            if np.std(true_i) > 0 and np.std(pred_i) > 0:
                corr, p_val = stats.pearsonr(true_i, pred_i)
                metrics[f'{name}_correlation'] = corr
                metrics[f'{name}_p_value'] = p_val
            else:
                metrics[f'{name}_correlation'] = 0.0
                metrics[f'{name}_p_value'] = 1.0
            
            # Uncertainty metrics (if available)
            if y_std is not None:
                std_i = y_std[:, i]
                
                # PICP: Prediction Interval Coverage Probability
                lower = pred_i - 1.96 * std_i
                upper = pred_i + 1.96 * std_i
                coverage = np.mean((true_i >= lower) & (true_i <= upper))
                metrics[f'{name}_PICP'] = coverage
                
                # MPIW: Mean Prediction Interval Width
                mpiw = np.mean(2 * 1.96 * std_i)
                metrics[f'{name}_MPIW'] = mpiw
        
        return metrics
    
    @staticmethod
    def concordance_index(risk_scores: np.ndarray,
                         times: np.ndarray,
                         events: np.ndarray) -> float:
        """
        Compute Harrell's concordance index (C-index).
        
        C = P(risk_i > risk_j | t_i < t_j, Î´_i = 1)
        
        Higher risk should correspond to shorter survival time.
        """
        n = len(times)
        concordant = 0
        discordant = 0
        tied_risk = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if comparable
                if events[i] == 1 and times[i] < times[j]:
                    # i had event before j's time
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
                        
                elif events[j] == 1 and times[j] < times[i]:
                    # j had event before i's time
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
        """
        Time-dependent AUC at specific time horizon.
        
        AUC(t) = P(risk_i > risk_j | T_i â‰¤ t < T_j)
        """
        # Cases: experienced event by time t
        cases = (times <= horizon) & (events == 1)
        
        # Controls: event-free at time t
        controls = times > horizon
        
        n_cases = cases.sum()
        n_controls = controls.sum()
        
        if n_cases == 0 or n_controls == 0:
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
    def brier_score(survival_probs: np.ndarray,
                   times: np.ndarray,
                   events: np.ndarray,
                   eval_time: float) -> float:
        """
        Compute Brier score at specific time.
        
        BS(t) = (1/n) âˆ‘ (I(T_i > t) - S(t|x_i))Â²
        """
        n = len(times)
        
        # True status at eval_time
        y_true = (times > eval_time).astype(float)
        
        # Predicted survival probability at eval_time
        if survival_probs.ndim == 1:
            y_pred = survival_probs
        else:
            # Find closest time index
            y_pred = survival_probs[:, 0]  # Simplified
        
        # Brier score
        bs = np.mean((y_true - y_pred) ** 2)
        
        return bs


# =============================================================================
# TRAINER
# =============================================================================

class PROGRESSTrainer:
    """
    Trainer for PROGRESS framework.
    
    Handles training of both trajectory and survival models
    with proper validation and early stopping.
    """
    
    def __init__(self, config: PROGRESSConfig):
        self.config = config
        self.device = config.get_device()
        
        # Models (initialized later with correct input dim)
        self.traj_model = None
        self.surv_model = None
        
        # Loss functions
        self.traj_loss_fn = TrajectoryLoss(config.calibration_weight)
        self.surv_loss_fn = CoxPartialLikelihoodLoss(config.ranking_weight)
        
        # Training history
        self.history = {
            'traj_train_loss': [], 'traj_val_loss': [],
            'surv_train_loss': [], 'surv_val_loss': [],
            'traj_metrics': [], 'surv_metrics': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_models(self, input_dim: int):
        """Initialize models with correct input dimension."""
        # Trajectory model
        self.traj_model = TrajectoryParameterNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.traj_hidden_dims,
            dropout=self.config.traj_dropout,
            num_attention_heads=self.config.traj_attention_heads,
            use_batch_norm=self.config.traj_use_batch_norm
        ).to(self.device)
        
        # Survival model
        self.surv_model = DeepSurvivalNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.surv_hidden_dims,
            dropout=self.config.surv_dropout,
            use_batch_norm=self.config.surv_use_batch_norm
        ).to(self.device)
        
        traj_params = sum(p.numel() for p in self.traj_model.parameters())
        surv_params = sum(p.numel() for p in self.surv_model.parameters())
        
        logger.info(f"Trajectory model: {traj_params:,} parameters")
        logger.info(f"Survival model: {surv_params:,} parameters")
    
    def train_trajectory_model(self,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               num_epochs: int = None) -> Dict:
        """
        Train trajectory parameter network.
        
        Returns training results dictionary.
        """
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
        
        logger.info(f"Training Trajectory Network for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
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
            
            # Validation phase
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
            
            # Early stopping check
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.traj_model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}: "
                           f"Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if patience_counter >= self.config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.traj_model.load_state_dict(best_state)
        
        return {'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
    
    def train_survival_model(self,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            num_epochs: int = None) -> Dict:
        """
        Train survival network.
        
        Returns training results dictionary.
        """
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
        
        logger.info(f"Training Survival Network for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
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
                
                # Safety check: ensure loss requires grad and is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at epoch {epoch+1}, skipping batch")
                    continue
                
                if not loss.requires_grad:
                    # This shouldn't happen with the fix, but just in case
                    logger.warning(f"Loss doesn't require grad at epoch {epoch+1}")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.surv_model.parameters(),
                    self.config.gradient_clip
                )
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
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
            
            # Early stopping
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.surv_model.state_dict().items()}
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}: "
                           f"Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if patience_counter >= self.config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.surv_model.load_state_dict(best_state)
        
        return {'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate both models on test set.
        
        Returns dictionary with all metrics including:
        - Regression metrics (RMSE, MAE, RÂ², correlation)
        - Classification metrics (accuracy, precision, recall, F1, AUC)
        - Survival metrics (C-index, time-dependent AUC)
        """
        self.traj_model.eval()
        self.surv_model.eval()
        
        # Collect data
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
        
        # === Trajectory Model Evaluation ===
        traj_output = self.traj_model.predict_with_uncertainty(
            features, n_samples=self.config.mc_dropout_samples
        )
        
        traj_pred = traj_output['mean'].cpu().numpy()
        traj_std = traj_output['total_std'].cpu().numpy()
        
        # Regression metrics
        traj_metrics = PROGRESSMetrics.trajectory_metrics(
            traj_targets, traj_pred, traj_std
        )
        results['trajectory_regression'] = traj_metrics
        
        # Classification metrics: Fast vs Slow progressor
        # Based on slope (Î² parameter, index 1)
        progressor_metrics = PROGRESSMetrics.progressor_classification(
            slope_true=traj_targets[:, 1],
            slope_pred=traj_pred[:, 1],
            slope_std=traj_std[:, 1],
            threshold=0.5  # CDR-SB points/year
        )
        results['trajectory_classification'] = progressor_metrics
        
        # === Survival Model Evaluation ===
        self.surv_model.eval()
        with torch.no_grad():
            risk_scores = self.surv_model(features).cpu().numpy().squeeze()
        
        # C-index
        c_index = PROGRESSMetrics.concordance_index(risk_scores, times, events)
        results['survival'] = {'c_index': c_index}
        
        # Time-dependent AUC and classification at different horizons
        for horizon in self.config.survival_horizons:
            if times.max() >= horizon:
                # Time-dependent AUC
                auc = PROGRESSMetrics.time_dependent_auc(
                    risk_scores, times, events, horizon
                )
                results['survival'][f'AUC_{horizon:.0f}yr'] = auc
                
                # Classification metrics for conversion within horizon
                conversion_metrics = PROGRESSMetrics.conversion_classification(
                    times=times,
                    events=events,
                    risk_scores=risk_scores,
                    horizon=horizon
                )
                results['survival'].update(conversion_metrics)
        
        # Risk stratification analysis
        stratification = PROGRESSMetrics.risk_stratification(
            risk_scores, times, events, n_groups=3
        )
        results['risk_stratification'] = stratification
        
        # === Summary Classification Metrics ===
        # Find optimal threshold for conversion prediction
        y_true_3yr = ((times <= 3.0) & (events == 1)).astype(int)
        if len(np.unique(y_true_3yr)) > 1:
            opt_thresh, opt_f1 = PROGRESSMetrics.find_optimal_threshold(
                y_true_3yr, risk_scores, metric='f1'
            )
            results['optimal_threshold'] = {
                'threshold': float(opt_thresh),
                'f1_at_optimal': float(opt_f1)
            }
        
        return results
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'traj_model': self.traj_model.state_dict(),
            'surv_model': self.surv_model.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.traj_model.load_state_dict(checkpoint['traj_model'])
        self.surv_model.load_state_dict(checkpoint['surv_model'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded: {path}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_curves(history: Dict, output_dir: str):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trajectory model
    ax = axes[0]
    if history['traj_train_loss']:
        epochs = range(1, len(history['traj_train_loss']) + 1)
        ax.plot(epochs, history['traj_train_loss'], 'b-', label='Train', alpha=0.8)
        ax.plot(epochs, history['traj_val_loss'], 'r-', label='Validation', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model 1: Trajectory Parameter Network')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Survival model
    ax = axes[1]
    if history['surv_train_loss']:
        epochs = range(1, len(history['surv_train_loss']) + 1)
        ax.plot(epochs, history['surv_train_loss'], 'b-', label='Train', alpha=0.8)
        ax.plot(epochs, history['surv_val_loss'], 'r-', label='Validation', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model 2: Deep Survival Network')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved: {save_path}")


def plot_trajectory_predictions(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_std: np.ndarray,
                               output_dir: str):
    """Plot trajectory parameter predictions vs true values."""
    param_names = ['Intercept (Î±)', 'Slope (Î²)', 'Acceleration (Î³)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        std_i = y_std[:, i]
        
        # Scatter plot
        ax.scatter(true_i, pred_i, alpha=0.5, s=20)
        
        # Identity line
        lims = [
            min(true_i.min(), pred_i.min()),
            max(true_i.max(), pred_i.max())
        ]
        ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect prediction')
        
        # Compute metrics
        r2 = r2_score(true_i, pred_i)
        corr = np.corrcoef(true_i, pred_i)[0, 1]
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\nRÂ²={r2:.3f}, r={corr:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'trajectory_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Trajectory predictions plot saved: {save_path}")


def plot_survival_analysis(risk_scores: np.ndarray,
                          times: np.ndarray,
                          events: np.ndarray,
                          output_dir: str):
    """Plot survival analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Risk score distribution by event status
    ax = axes[0]
    risk_event = risk_scores[events == 1]
    risk_censored = risk_scores[events == 0]
    
    ax.hist(risk_censored, bins=30, alpha=0.6, label='Censored', density=True)
    ax.hist(risk_event, bins=30, alpha=0.6, label='Event', density=True)
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Density')
    ax.set_title('Risk Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Risk vs time scatter
    ax = axes[1]
    scatter = ax.scatter(times, risk_scores, c=events, cmap='RdYlBu_r', 
                        alpha=0.6, s=20)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Risk Score')
    ax.set_title('Risk Score vs Follow-up Time')
    plt.colorbar(scatter, ax=ax, label='Event (1=yes)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'survival_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Survival analysis plot saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_progress_pipeline(data_dir: str,
                         config: PROGRESSConfig = None,
                         output_dir: str = None) -> Dict:
    """
    Run complete PROGRESS training and evaluation pipeline.
    
    Args:
        data_dir: Directory containing data files
        config: Configuration object
        output_dir: Output directory (default: {data_dir}/progress_output)
        
    Returns:
        Dictionary with all results
    """
    if config is None:
        config = PROGRESSConfig()
    
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'progress_output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to output directory
    global logger
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("PROGRESS: PRognostic Generalization from REsting Static Signatures")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config.get_device()}")
    
    # === Load Data ===
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    integrated_path = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    sequences_path = os.path.join(data_dir, 'nacc_ml_sequences_cleaned.pkl')
    
    if not os.path.exists(integrated_path):
        raise FileNotFoundError(f"Integrated dataset not found: {integrated_path}")
    
    integrated_data = pd.read_pickle(integrated_path)
    logger.info(f"Loaded integrated dataset: {len(integrated_data)} subjects")
    
    sequences_data = None
    if os.path.exists(sequences_path):
        sequences_data = pd.read_pickle(sequences_path)
        logger.info(f"Loaded sequences dataset: {len(sequences_data)} sequences")
    
    # === Create Dataset ===
    logger.info("\n" + "=" * 70)
    logger.info("CREATING DATASET")
    logger.info("=" * 70)
    
    full_dataset = PROGRESSDataset(
        integrated_data=integrated_data,
        sequences_data=sequences_data,
        fit_scaler=True,
        config=config
    )
    
    # === Data Split ===
    logger.info("\n" + "=" * 70)
    logger.info("SPLITTING DATA")
    logger.info("=" * 70)
    
    n_samples = len(full_dataset)
    indices = np.arange(n_samples)
    events = full_dataset.survival_data['events']
    
    # Stratified split by event status
    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=events
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42, stratify=events[train_idx]
    )
    
    logger.info(f"Train: {len(train_idx)} ({100*len(train_idx)/n_samples:.1f}%)")
    logger.info(f"Validation: {len(val_idx)} ({100*len(val_idx)/n_samples:.1f}%)")
    logger.info(f"Test: {len(test_idx)} ({100*len(test_idx)/n_samples:.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # === Initialize Trainer ===
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING MODELS")
    logger.info("=" * 70)
    
    trainer = PROGRESSTrainer(config)
    trainer.setup_models(input_dim=full_dataset.X.shape[1])
    
    # === Train Model 1: Trajectory Network ===
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL 1: Trajectory Parameter Network")
    logger.info("=" * 70)
    
    traj_results = trainer.train_trajectory_model(train_loader, val_loader)
    logger.info(f"Best validation loss: {traj_results['best_val_loss']:.4f}")
    
    # === Train Model 2: Survival Network ===
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL 2: Deep Survival Network")
    logger.info("=" * 70)
    
    surv_results = trainer.train_survival_model(train_loader, val_loader)
    logger.info(f"Best validation loss: {surv_results['best_val_loss']:.4f}")
    
    # === Evaluation ===
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 70)
    
    test_results = trainer.evaluate(test_loader)
    
    # Print trajectory metrics
    logger.info("\nTrajectory Parameter Prediction (Regression):")
    if 'trajectory_regression' in test_results:
        for metric, value in test_results['trajectory_regression'].items():
            if not metric.endswith('p_value'):
                logger.info(f"  {metric}: {value:.4f}")
    
    # Print trajectory classification metrics
    logger.info("\nProgressor Classification (Fast vs Slow decline):")
    if 'trajectory_classification' in test_results:
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'specificity']
        for metric, value in test_results['trajectory_classification'].items():
            short_name = metric.replace('progressor_', '')
            if short_name in key_metrics:
                logger.info(f"  {short_name}: {value:.4f}")
    
    # Print survival metrics
    logger.info("\nSurvival Prediction:")
    if 'survival' in test_results:
        for metric, value in test_results['survival'].items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
    
    # Print risk stratification
    logger.info("\nRisk Stratification:")
    if 'risk_stratification' in test_results:
        strat = test_results['risk_stratification']
        logger.info(f"  Groups: {strat.get('n_groups', 3)}")
        logger.info(f"  Group sizes: {strat.get('group_sizes', [])}")
        for g in range(strat.get('n_groups', 3)):
            er = strat.get(f'group_{g}_event_rate', 0)
            logger.info(f"  Group {g} event rate: {er:.3f}")
        if 'log_rank_p_value' in strat:
            logger.info(f"  Log-rank p-value: {strat['log_rank_p_value']:.4f}")
    
    # Print optimal threshold
    if 'optimal_threshold' in test_results:
        opt = test_results['optimal_threshold']
        logger.info(f"\nOptimal Classification Threshold:")
        logger.info(f"  Threshold: {opt['threshold']:.4f}")
        logger.info(f"  F1 at optimal: {opt['f1_at_optimal']:.4f}")
    
    # === Save Results ===
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)
    
    # Save checkpoint
    trainer.save_checkpoint(os.path.join(output_dir, 'progress_models.pt'))
    
    # Save scaler
    with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(full_dataset.scaler, f)
    
    # Collect all predictions for visualization
    all_features = torch.cat([batch['features'] for batch in test_loader]).to(trainer.device)
    all_traj_targets = torch.cat([batch['trajectory_params'] for batch in test_loader]).numpy()
    all_times = torch.cat([batch['time'] for batch in test_loader]).numpy()
    all_events = torch.cat([batch['event'] for batch in test_loader]).numpy()
    
    # Get predictions
    trainer.traj_model.eval()
    trainer.surv_model.eval()
    
    with torch.no_grad():
        traj_output = trainer.traj_model.predict_with_uncertainty(
            all_features, n_samples=config.mc_dropout_samples
        )
        traj_pred = traj_output['mean'].cpu().numpy()
        traj_std = traj_output['total_std'].cpu().numpy()
        
        risk_scores = trainer.surv_model(all_features).cpu().numpy().squeeze()
    
    # Save predictions
    predictions = {
        'trajectory_true': all_traj_targets,
        'trajectory_pred': traj_pred,
        'trajectory_std': traj_std,
        'survival_times': all_times,
        'survival_events': all_events,
        'risk_scores': risk_scores
    }
    
    with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    
    # Generate plots
    plot_training_curves(trainer.history, output_dir)
    plot_trajectory_predictions(all_traj_targets, traj_pred, traj_std, output_dir)
    plot_survival_analysis(risk_scores, all_times, all_events, output_dir)
    
    # Save summary report
    report = {
        'config': config.to_dict(),
        'dataset': {
            'n_subjects': len(full_dataset),
            'n_features': full_dataset.X.shape[1],
            'feature_names': full_dataset.feature_names,
            'event_rate': float(events.mean())
        },
        'splits': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        },
        'training': {
            'trajectory': traj_results,
            'survival': surv_results
        },
        'test_results': test_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'progress_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("\nPROGRESS pipeline completed successfully!")
    
    return report


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PROGRESS: AD Progression Prediction Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python PROGRESS.py --data-dir ./dataset
  python PROGRESS.py --data-dir ./dataset --epochs 50 --batch-size 16
  python PROGRESS.py --data-dir ./dataset --quick-test
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing nacc_integrated_dataset.pkl and nacc_ml_sequences_cleaned.pkl'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: {data-dir}/progress_output)'
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
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
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
        help='Quick test run with reduced epochs'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PROGRESSConfig(
        num_epochs=10 if args.quick_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=5 if args.quick_test else args.patience,
        device=args.device
    )
    
    # Run pipeline
    try:
        results = run_progress_pipeline(
            data_dir=args.data_dir,
            config=config,
            output_dir=args.output_dir
        )
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
