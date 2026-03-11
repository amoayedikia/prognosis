#!/usr/bin/env python3
"""
run_baseline_comparison.py - Run Baseline Comparisons on NACC Data

This script runs all baseline comparison methods on your actual NACC integrated
dataset and compares them with the PROGRESS framework.

Usage:
    python run_baseline_comparison.py --data-dir ./dataset
    python run_baseline_comparison.py --data-dir ./dataset --quick-test
    
Required files in data-dir:
    - nacc_integrated_dataset.pkl (from NACCDataIntegrator)
    - nacc_ml_sequences_cleaned.pkl (optional)

Output:
    - baseline_results.json: Complete comparison results
    - baseline_comparison_tables.csv: Tables formatted for paper
    - baseline_plots/: Visualization directory

Author: Generated for PROGRESS paper
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, 
    ElasticNet, BayesianRidge, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, f1_score
)

# Optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Info: XGBoost not installed. Install with: pip install xgboost")

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
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
# DATA LOADING (SAME AS PROGRESS.py)
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


def load_nacc_data(data_dir: str) -> Dict[str, np.ndarray]:
    """
    Load NACC data and extract features, trajectory params, and survival data.
    
    This replicates the data loading from PROGRESS.py PROGRESSDataset.
    
    Args:
        data_dir: Directory containing nacc_integrated_dataset.pkl
        
    Returns:
        Dictionary with:
            - X: baseline features (n_samples, n_features)
            - y_traj: trajectory parameters (n_samples, 3)
            - times: survival times (n_samples,)
            - events: event indicators (n_samples,)
            - feature_names: list of feature names
    """
    logger.info("=" * 60)
    logger.info("Loading NACC Data")
    logger.info("=" * 60)
    
    integrated_path = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
    
    if not os.path.exists(integrated_path):
        raise FileNotFoundError(
            f"Integrated dataset not found: {integrated_path}\n"
            f"Please run NACCDataIntegrator.py first to create the integrated dataset."
        )
    
    integrated_data = pd.read_pickle(integrated_path)
    logger.info(f"Loaded integrated dataset: {len(integrated_data)} subjects")
    
    # Get valid subjects
    valid_subjects = []
    for _, row in integrated_data.iterrows():
        naccid = row.get('NACCID')
        if naccid is None:
            continue
        
        # Check for CSF biomarkers
        has_biomarker = any([
            is_valid(row.get('ABETA_harm')),
            is_valid(row.get('PTAU_harm')),
            is_valid(row.get('TTAU_harm'))
        ])
        
        if not has_biomarker:
            continue
        
        # Check for clinical trajectory
        trajectory = row.get('clinical_trajectory', [])
        if isinstance(trajectory, list) and len(trajectory) >= 2:
            valid_subjects.append(naccid)
    
    logger.info(f"Valid subjects: {len(valid_subjects)}")
    
    # Extract features
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
        
        # === Extract Features ===
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
        
        # === Extract Trajectory Parameters (Quadratic Fit) ===
        if isinstance(trajectory, list) and len(trajectory) >= 3:
            times_traj = []
            scores_traj = []
            for visit in trajectory:
                t = visit.get('YearsFromBaseline', 0)
                score = visit.get('CDRSUM')
                if is_valid(score) and is_valid(t):
                    times_traj.append(float(t))
                    scores_traj.append(float(score))
            
            if len(times_traj) >= 3:
                try:
                    times_arr = np.array(times_traj)
                    scores_arr = np.array(scores_traj)
                    design = np.column_stack([
                        np.ones(len(times_arr)),
                        times_arr,
                        times_arr ** 2
                    ])
                    coeffs, _, _, _ = np.linalg.lstsq(design, scores_arr, rcond=None)
                    trajectory_params_list.append(coeffs)
                except:
                    trajectory_params_list.append([np.nan, np.nan, np.nan])
            else:
                trajectory_params_list.append([np.nan, np.nan, np.nan])
        else:
            trajectory_params_list.append([np.nan, np.nan, np.nan])
        
        # === Extract Survival Data ===
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
    
    # Handle NaN in features (column median imputation)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            median_val = np.nanmedian(col_data)
            if np.isnan(median_val):
                median_val = 0.0
            X[mask, col] = median_val
    
    # Handle NaN in trajectory params
    for col in range(y_traj.shape[1]):
        col_data = y_traj[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            median_val = np.nanmedian(col_data)
            defaults = [1.0, 0.3, 0.02]
            if np.isnan(median_val):
                median_val = defaults[col]
            y_traj[mask, col] = median_val
    
    # Handle NaN in times
    mask = np.isnan(times)
    if mask.any():
        times[mask] = np.nanmedian(times)
    times = np.maximum(times, 0.1)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Dataset prepared:")
    logger.info(f"  Samples: {X.shape[0]}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Event rate: {events.sum()}/{len(events)} ({100*events.mean():.1f}%)")
    logger.info(f"  Median follow-up: {np.median(times):.1f} years")
    
    return {
        'X': X_scaled,
        'X_raw': X,
        'y_traj': y_traj,
        'times': times,
        'events': events,
        'feature_names': feature_names,
        'scaler': scaler
    }


# =============================================================================
# BASELINE MODELS
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


def compute_c_index(risk_scores: np.ndarray, times: np.ndarray,
                   events: np.ndarray) -> float:
    """Compute Harrell's C-index."""
    concordant = 0
    comparable = 0
    
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] < times[j] and events[i] == 1:
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] == 1:
                comparable += 1
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    
    return concordant / comparable if comparable > 0 else 0.5


def compute_time_dependent_auc(risk_scores: np.ndarray, times: np.ndarray,
                              events: np.ndarray, horizon: float) -> float:
    """Compute time-dependent AUC at a specific horizon."""
    y_true = (times <= horizon) & (events == 1)
    
    if y_true.sum() < 2 or (~y_true).sum() < 2:
        return 0.5
    
    try:
        return roc_auc_score(y_true, risk_scores)
    except:
        return 0.5


# =============================================================================
# MAIN COMPARISON RUNNER
# =============================================================================

def run_trajectory_baselines(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             epochs: int = 100) -> Dict:
    """Run all trajectory prediction baselines."""
    results = {}
    param_names = ['intercept', 'slope', 'acceleration']
    
    # 1. Linear Regression
    logger.info("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Linear Regression'] = {
        f'{param_names[i]}_r2': r2_score(y_test[:, i], y_pred[:, i])
        for i in range(3)
    }
    
    # 2. Ridge Regression
    logger.info("  Training Ridge Regression...")
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results['Ridge'] = {
        f'{param_names[i]}_r2': r2_score(y_test[:, i], y_pred[:, i])
        for i in range(3)
    }
    
    # 3. Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = {
        f'{param_names[i]}_r2': r2_score(y_test[:, i], y_pred[:, i])
        for i in range(3)
    }
    
    # 4. Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    gb_results = {}
    for i, name in enumerate(param_names):
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train, y_train[:, i])
        y_pred_i = gb.predict(X_test)
        gb_results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred_i)
    results['Gradient Boosting'] = gb_results
    
    # 5. XGBoost (if available)
    if HAS_XGBOOST:
        logger.info("  Training XGBoost...")
        xgb_results = {}
        for i, name in enumerate(param_names):
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
            model.fit(X_train, y_train[:, i])
            y_pred_i = model.predict(X_test)
            xgb_results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred_i)
        results['XGBoost'] = xgb_results
    
    # 6. SVR
    logger.info("  Training SVR...")
    svr_results = {}
    for i, name in enumerate(param_names):
        svr = SVR(kernel='rbf', C=1.0)
        svr.fit(X_train, y_train[:, i])
        y_pred_i = svr.predict(X_test)
        svr_results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred_i)
    results['SVR'] = svr_results
    
    # 7. Bayesian Ridge
    logger.info("  Training Bayesian Ridge...")
    br_results = {}
    for i, name in enumerate(param_names):
        br = BayesianRidge()
        br.fit(X_train, y_train[:, i])
        y_pred_i = br.predict(X_test)
        br_results[f'{name}_r2'] = r2_score(y_test[:, i], y_pred_i)
    results['Bayesian Ridge'] = br_results
    
    # 8. MLP Baseline
    logger.info("  Training MLP...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    mlp = MLPRegressor(X_train.shape[1], 3).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    mlp.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = mlp(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_test_t).cpu().numpy()
    
    results['MLP'] = {
        f'{param_names[i]}_r2': r2_score(y_test[:, i], y_pred[:, i])
        for i in range(3)
    }
    
    return results


def run_survival_baselines(X_train: np.ndarray, times_train: np.ndarray,
                          events_train: np.ndarray, X_test: np.ndarray,
                          times_test: np.ndarray, events_test: np.ndarray,
                          feature_names: List[str], epochs: int = 100) -> Dict:
    """Run all survival analysis baselines."""
    results = {}
    
    # 1. Cox PH (lifelines)
    if HAS_LIFELINES:
        logger.info("  Training Cox PH...")
        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train['T'] = times_train
        df_train['E'] = events_train
        
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test['T'] = times_test
        df_test['E'] = events_test
        
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df_train, duration_col='T', event_col='E')
            
            risk_scores = -cph.predict_partial_hazard(df_test).values.flatten()
            c_index = compute_c_index(risk_scores, times_test, events_test)
            
            results['Cox PH'] = {
                'c_index': c_index,
                'auc_2yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 2.0),
                'auc_3yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 3.0),
                'auc_5yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 5.0),
            }
        except Exception as e:
            logger.warning(f"Cox PH failed: {e}")
    
    # 2. Random Survival Forest (scikit-survival)
    if HAS_SKSURV:
        logger.info("  Training Random Survival Forest...")
        y_train_surv = np.array([(bool(e), t) for e, t in zip(events_train, times_train)],
                               dtype=[('event', bool), ('time', float)])
        y_test_surv = np.array([(bool(e), t) for e, t in zip(events_test, times_test)],
                              dtype=[('event', bool), ('time', float)])
        
        try:
            rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rsf.fit(X_train, y_train_surv)
            risk_scores = rsf.predict(X_test)
            
            c_index, _, _, _, _ = concordance_index_censored(
                events_test.astype(bool), times_test, risk_scores
            )
            
            results['Random Survival Forest'] = {
                'c_index': c_index,
                'auc_2yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 2.0),
                'auc_3yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 3.0),
                'auc_5yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 5.0),
            }
        except Exception as e:
            logger.warning(f"RSF failed: {e}")
        
        # 3. Cox-Lasso
        logger.info("  Training Cox-Lasso...")
        try:
            cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True)
            cox_lasso.fit(X_train, y_train_surv)
            risk_scores = cox_lasso.predict(X_test)
            
            c_index, _, _, _, _ = concordance_index_censored(
                events_test.astype(bool), times_test, risk_scores
            )
            
            results['Cox-Lasso'] = {
                'c_index': c_index,
                'auc_2yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 2.0),
                'auc_3yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 3.0),
                'auc_5yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 5.0),
            }
        except Exception as e:
            logger.warning(f"Cox-Lasso failed: {e}")
        
        # 4. Gradient Boosting Survival
        logger.info("  Training Gradient Boosting Survival...")
        try:
            gbs = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=5, random_state=42)
            gbs.fit(X_train, y_train_surv)
            risk_scores = gbs.predict(X_test)
            
            c_index, _, _, _, _ = concordance_index_censored(
                events_test.astype(bool), times_test, risk_scores
            )
            
            results['GB Survival'] = {
                'c_index': c_index,
                'auc_2yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 2.0),
                'auc_3yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 3.0),
                'auc_5yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 5.0),
            }
        except Exception as e:
            logger.warning(f"GBS failed: {e}")
    
    # 5. DeepSurv Baseline
    logger.info("  Training DeepSurv baseline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    times_train_t = torch.FloatTensor(times_train).to(device)
    events_train_t = torch.FloatTensor(events_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    deepsurv = DeepSurvBaseline(X_train.shape[1]).to(device)
    optimizer = optim.AdamW(deepsurv.parameters(), lr=1e-3)
    
    deepsurv.train()
    for epoch in range(epochs):
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
        
        loss.backward()
        optimizer.step()
    
    deepsurv.eval()
    with torch.no_grad():
        risk_scores = deepsurv(X_test_t).cpu().numpy().squeeze()
    
    c_index = compute_c_index(risk_scores, times_test, events_test)
    
    results['DeepSurv (baseline)'] = {
        'c_index': c_index,
        'auc_2yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 2.0),
        'auc_3yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 3.0),
        'auc_5yr': compute_time_dependent_auc(risk_scores, times_test, events_test, 5.0),
    }
    
    return results


def run_all_baselines(data_dir: str, output_dir: str = None,
                     quick_test: bool = False) -> Dict:
    """
    Run complete baseline comparison on NACC data.
    
    Args:
        data_dir: Directory containing nacc_integrated_dataset.pkl
        output_dir: Output directory for results
        quick_test: If True, use reduced epochs
        
    Returns:
        Dictionary with all comparison results
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'baseline_comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("BASELINE COMPARISON FOR PROGRESS FRAMEWORK")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quick test mode: {quick_test}")
    
    # Load data
    data = load_nacc_data(data_dir)
    
    X = data['X']
    y_traj = data['y_traj']
    times = data['times']
    events = data['events']
    feature_names = data['feature_names']
    
    # Split data
    logger.info("\n" + "=" * 60)
    logger.info("Splitting Data (80/20)")
    logger.info("=" * 60)
    
    (X_train, X_test, y_traj_train, y_traj_test,
     times_train, times_test, events_train, events_test) = train_test_split(
        X, y_traj, times, events, test_size=0.2, random_state=42, stratify=events
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Train event rate: {events_train.mean()*100:.1f}%")
    logger.info(f"Test event rate: {events_test.mean()*100:.1f}%")
    
    epochs = 50 if quick_test else 100
    
    # Run trajectory baselines
    logger.info("\n" + "=" * 60)
    logger.info("TRAJECTORY PREDICTION BASELINES")
    logger.info("=" * 60)
    
    traj_results = run_trajectory_baselines(
        X_train, y_traj_train, X_test, y_traj_test, epochs=epochs
    )
    
    # Run survival baselines
    logger.info("\n" + "=" * 60)
    logger.info("SURVIVAL PREDICTION BASELINES")
    logger.info("=" * 60)
    
    surv_results = run_survival_baselines(
        X_train, times_train, events_train,
        X_test, times_test, events_test,
        feature_names, epochs=epochs
    )
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("TRAJECTORY PREDICTION RESULTS")
    logger.info("=" * 70)
    
    print("\n{:<25} {:>12} {:>12} {:>12}".format(
        "Method", "Intercept R²", "Slope R²", "Accel R²"
    ))
    print("-" * 65)
    
    for method, metrics in traj_results.items():
        int_r2 = metrics.get('intercept_r2', 0)
        slope_r2 = metrics.get('slope_r2', 0)
        accel_r2 = metrics.get('acceleration_r2', 0)
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            method, int_r2, slope_r2, accel_r2
        ))
    
    logger.info("\n" + "=" * 70)
    logger.info("SURVIVAL PREDICTION RESULTS")
    logger.info("=" * 70)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "C-index", "AUC-2yr", "AUC-3yr", "AUC-5yr"
    ))
    print("-" * 70)
    
    for method, metrics in surv_results.items():
        c_idx = metrics.get('c_index', 0)
        auc2 = metrics.get('auc_2yr', 0)
        auc3 = metrics.get('auc_3yr', 0)
        auc5 = metrics.get('auc_5yr', 0)
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            method, c_idx, auc2, auc3, auc5
        ))
    
    # Save results
    results = {
        'trajectory': traj_results,
        'survival': surv_results,
        'dataset_info': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'event_rate_train': float(events_train.mean()),
            'event_rate_test': float(events_test.mean()),
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(output_dir, 'baseline_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Create comparison DataFrame for paper
    traj_df = pd.DataFrame([
        {
            'Method': method,
            'Intercept R²': metrics.get('intercept_r2', 0),
            'Slope R²': metrics.get('slope_r2', 0),
            'Acceleration R²': metrics.get('acceleration_r2', 0)
        }
        for method, metrics in traj_results.items()
    ])
    
    surv_df = pd.DataFrame([
        {
            'Method': method,
            'C-index': metrics.get('c_index', 0),
            'AUC-2yr': metrics.get('auc_2yr', 0),
            'AUC-3yr': metrics.get('auc_3yr', 0),
            'AUC-5yr': metrics.get('auc_5yr', 0)
        }
        for method, metrics in surv_results.items()
    ])
    
    traj_csv = os.path.join(output_dir, 'trajectory_comparison.csv')
    surv_csv = os.path.join(output_dir, 'survival_comparison.csv')
    
    traj_df.to_csv(traj_csv, index=False)
    surv_df.to_csv(surv_csv, index=False)
    
    logger.info(f"Comparison tables saved to:")
    logger.info(f"  - {traj_csv}")
    logger.info(f"  - {surv_csv}")
    
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON COMPLETED")
    logger.info("=" * 70)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Baseline Comparisons on NACC Data'
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
        help='Output directory (default: {data-dir}/baseline_comparison_results)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with reduced epochs'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_all_baselines(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            quick_test=args.quick_test
        )
        return 0
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
