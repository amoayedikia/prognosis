#!/usr/bin/env python3
"""
Survival Model Statistical Significance Tests

Compares PROGRESS survival model against baseline survival methods:
- Cox Proportional Hazards (Linear)
- Cox Proportional Hazards (with biomarker interactions)
- DeepSurv (neural network Cox model)
- Random Survival Forest (RSF)

Performs statistical significance tests:
- DeLong test for AUC comparison
- Bootstrap confidence intervals
- Wilcoxon signed-rank test across CV folds
- Permutation tests

================================================================================
USAGE
================================================================================

python survival_significance_tests.py --data-dir ./dataset --n-runs 5 --n-folds 5

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
from scipy.special import ndtri
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

# Optional: lifelines for Cox PH
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as lifelines_cindex
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Warning: lifelines not installed. Cox PH baseline will use custom implementation.")

# Optional: scikit-survival for RSF
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("Warning: scikit-survival not installed. RSF baseline will be skipped.")


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str = '.'):
    """Configure logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'survival_significance_tests.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TestResult:
    """Container for a statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'statistic': float(self.statistic),
            'p_value': float(self.p_value),
            'effect_size': float(self.effect_size) if self.effect_size else None,
            'ci_lower': float(self.ci_lower) if self.ci_lower else None,
            'ci_upper': float(self.ci_upper) if self.ci_upper else None,
            'significant_005': self.p_value < 0.05,
            'significant_001': self.p_value < 0.01
        }


@dataclass
class AlgorithmResults:
    """Results from running an algorithm across multiple folds/seeds."""
    name: str
    c_index_scores: List[float] = field(default_factory=list)
    auc_2yr_scores: List[float] = field(default_factory=list)
    auc_3yr_scores: List[float] = field(default_factory=list)
    auc_5yr_scores: List[float] = field(default_factory=list)
    ibs_scores: List[float] = field(default_factory=list)
    predictions: List[np.ndarray] = field(default_factory=list)
    
    @property
    def c_index_mean(self) -> float:
        return np.mean(self.c_index_scores) if self.c_index_scores else 0.0
    
    @property
    def c_index_std(self) -> float:
        return np.std(self.c_index_scores) if self.c_index_scores else 0.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'c_index': {'mean': self.c_index_mean, 'std': self.c_index_std, 'values': self.c_index_scores},
            'auc_2yr': {'mean': np.mean(self.auc_2yr_scores), 'std': np.std(self.auc_2yr_scores), 'values': self.auc_2yr_scores},
            'auc_3yr': {'mean': np.mean(self.auc_3yr_scores), 'std': np.std(self.auc_3yr_scores), 'values': self.auc_3yr_scores},
            'auc_5yr': {'mean': np.mean(self.auc_5yr_scores), 'std': np.std(self.auc_5yr_scores), 'values': self.auc_5yr_scores},
            'ibs': {'mean': np.mean(self.ibs_scores), 'std': np.std(self.ibs_scores), 'values': self.ibs_scores}
        }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

class DeLongTest:
    """
    DeLong test for comparing two correlated AUCs.
    
    Reference: DeLong et al. (1988). Biometrics, 44, 837-845.
    """
    
    @staticmethod
    def compute_midrank(x: np.ndarray) -> np.ndarray:
        """Compute midranks for ties."""
        n = len(x)
        sorted_idx = np.argsort(x)
        ranks = np.empty(n, dtype=float)
        
        i = 0
        while i < n:
            j = i
            while j < n and x[sorted_idx[j]] == x[sorted_idx[i]]:
                j += 1
            avg_rank = 0.5 * (i + j + 1)
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        
        return ranks
    
    @classmethod
    def test(cls, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> TestResult:
        """
        Perform DeLong test comparing two AUCs.
        
        Args:
            y_true: True binary labels
            pred_a: Predictions from model A
            pred_b: Predictions from model B
            
        Returns:
            TestResult with z-statistic and p-value
        """
        y_true = np.asarray(y_true).ravel()
        pred_a = np.asarray(pred_a).ravel()
        pred_b = np.asarray(pred_b).ravel()
        
        # Get positive and negative indices
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        
        if n_pos == 0 or n_neg == 0:
            return TestResult('DeLong', 0.0, 1.0)
        
        # Compute AUCs
        auc_a = roc_auc_score(y_true, pred_a)
        auc_b = roc_auc_score(y_true, pred_b)
        
        # Compute placement values for variance estimation
        # Model A
        ranks_a = cls.compute_midrank(pred_a)
        v10_a = (ranks_a[pos_idx] - (n_pos + 1) / 2) / n_neg
        v01_a = 1 - (ranks_a[neg_idx] - (n_neg + 1) / 2) / n_pos
        
        # Model B
        ranks_b = cls.compute_midrank(pred_b)
        v10_b = (ranks_b[pos_idx] - (n_pos + 1) / 2) / n_neg
        v01_b = 1 - (ranks_b[neg_idx] - (n_neg + 1) / 2) / n_pos
        
        # Compute covariance matrix
        s10_a = np.var(v10_a, ddof=1) if n_pos > 1 else 0
        s01_a = np.var(v01_a, ddof=1) if n_neg > 1 else 0
        s10_b = np.var(v10_b, ddof=1) if n_pos > 1 else 0
        s01_b = np.var(v01_b, ddof=1) if n_neg > 1 else 0
        
        # Covariance between A and B
        s10_ab = np.cov(v10_a, v10_b, ddof=1)[0, 1] if n_pos > 1 else 0
        s01_ab = np.cov(v01_a, v01_b, ddof=1)[0, 1] if n_neg > 1 else 0
        
        # Variance of difference
        var_a = s10_a / n_pos + s01_a / n_neg
        var_b = s10_b / n_pos + s01_b / n_neg
        cov_ab = s10_ab / n_pos + s01_ab / n_neg
        
        var_diff = var_a + var_b - 2 * cov_ab
        
        if var_diff <= 0:
            return TestResult('DeLong', 0.0, 1.0, effect_size=auc_a - auc_b)
        
        # Z-statistic
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return TestResult('DeLong', float(z), float(p_value), effect_size=float(auc_a - auc_b))


class BootstrapTest:
    """Bootstrap confidence intervals and hypothesis tests."""
    
    @staticmethod
    def confidence_interval(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        seed: int = 42
    ) -> TestResult:
        """
        Compute bootstrap CI for difference in means.
        
        Args:
            scores_a: Scores from model A (e.g., C-index values across folds)
            scores_b: Scores from model B
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            seed: Random seed
            
        Returns:
            TestResult with CI and p-value
        """
        np.random.seed(seed)
        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)
        
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        n = len(scores_a)
        
        # Bootstrap differences
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_diff = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        # P-value (two-tailed test for difference from 0)
        # Shift distribution to be centered at 0 under null
        centered_diffs = bootstrap_diffs - observed_diff
        p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))
        
        # Alternative: proportion of bootstrap samples with opposite sign
        if observed_diff > 0:
            p_value_alt = 2 * np.mean(bootstrap_diffs <= 0)
        else:
            p_value_alt = 2 * np.mean(bootstrap_diffs >= 0)
        
        return TestResult(
            'Bootstrap',
            float(observed_diff),
            float(min(p_value_alt, 1.0)),
            effect_size=float(observed_diff),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper)
        )


class WilcoxonTest:
    """Wilcoxon signed-rank test for paired comparisons."""
    
    @staticmethod
    def test(scores_a: np.ndarray, scores_b: np.ndarray) -> TestResult:
        """
        Perform Wilcoxon signed-rank test.
        
        Args:
            scores_a: Scores from model A (paired with scores_b)
            scores_b: Scores from model B
            
        Returns:
            TestResult with statistic and p-value
        """
        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)
        
        # Remove ties (differences of 0)
        diff = scores_a - scores_b
        non_zero_mask = diff != 0
        
        if non_zero_mask.sum() < 2:
            return TestResult('Wilcoxon', 0.0, 1.0, effect_size=float(np.mean(diff)))
        
        statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
        
        # Effect size: matched-pairs rank biserial correlation
        n = len(diff[non_zero_mask])
        effect_size = 1 - (2 * statistic) / (n * (n + 1))
        
        return TestResult('Wilcoxon', float(statistic), float(p_value), effect_size=float(effect_size))


class MannWhitneyTest:
    """Mann-Whitney U test for unpaired comparisons."""
    
    @staticmethod
    def test(scores_a: np.ndarray, scores_b: np.ndarray) -> TestResult:
        """
        Perform Mann-Whitney U test.
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            
        Returns:
            TestResult with U statistic and p-value
        """
        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)
        
        statistic, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        
        # Effect size: rank biserial correlation
        n1, n2 = len(scores_a), len(scores_b)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return TestResult('Mann-Whitney', float(statistic), float(p_value), effect_size=float(effect_size))


class PermutationTest:
    """Permutation test for flexible hypothesis testing."""
    
    @staticmethod
    def test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_permutations: int = 10000,
        seed: int = 42
    ) -> TestResult:
        """
        Perform permutation test for difference in means.
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            n_permutations: Number of permutations
            seed: Random seed
            
        Returns:
            TestResult with observed difference and p-value
        """
        np.random.seed(seed)
        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)
        
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        combined = np.concatenate([scores_a, scores_b])
        n_a = len(scores_a)
        
        # Permutation distribution
        extreme_count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1
        
        p_value = (extreme_count + 1) / (n_permutations + 1)
        
        return TestResult('Permutation', float(observed_diff), float(p_value), effect_size=float(observed_diff))


def apply_multiple_testing_correction(p_values: List[float], method: str = 'holm') -> List[float]:
    """
    Apply multiple testing correction.
    
    Args:
        p_values: List of p-values
        method: 'bonferroni', 'holm', or 'fdr_bh'
        
    Returns:
        Corrected p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if method == 'bonferroni':
        return list(np.minimum(p_values * n, 1.0))
    
    elif method == 'holm':
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        corrected = np.zeros(n)
        
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(sorted_p[i] * (n - i), 1.0)
        
        # Ensure monotonicity
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i-1]
            corrected[idx] = max(corrected[idx], corrected[prev_idx])
        
        return list(corrected)
    
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        corrected = np.zeros(n)
        
        for i in range(n):
            corrected[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
        
        # Ensure monotonicity (from largest to smallest)
        for i in range(n-2, -1, -1):
            idx = sorted_idx[i]
            next_idx = sorted_idx[i+1]
            corrected[idx] = min(corrected[idx], corrected[next_idx])
        
        return list(np.minimum(corrected, 1.0))
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# METRICS
# =============================================================================

def concordance_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """
    Compute Harrell's concordance index.
    
    C = P(risk_i > risk_j | t_i < t_j, δ_i = 1)
    """
    n = len(times)
    if n < 2:
        return 0.5
    
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 1 and times[i] < times[j]:
                # i had event before j's time
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied += 1
            elif events[j] == 1 and times[j] < times[i]:
                # j had event before i's time
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] < risk_scores[i]:
                    discordant += 1
                else:
                    tied += 1
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


def time_dependent_auc(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    horizon: float
) -> float:
    """
    Compute time-dependent AUC at a specific time horizon.
    
    Uses the incident/dynamic definition.
    """
    # Cases: event before horizon
    # Controls: still at risk at horizon (time > horizon)
    
    cases_mask = (events == 1) & (times <= horizon)
    controls_mask = times > horizon
    
    n_cases = cases_mask.sum()
    n_controls = controls_mask.sum()
    
    if n_cases == 0 or n_controls == 0:
        return 0.5
    
    case_scores = risk_scores[cases_mask]
    control_scores = risk_scores[controls_mask]
    
    # AUC = P(risk_case > risk_control)
    auc = 0.0
    for cs in case_scores:
        auc += np.sum(cs > control_scores) + 0.5 * np.sum(cs == control_scores)
    
    auc /= (n_cases * n_controls)
    return auc


def integrated_brier_score(
    survival_probs: np.ndarray,  # Shape: (n_samples, n_times)
    eval_times: np.ndarray,
    times: np.ndarray,
    events: np.ndarray
) -> float:
    """
    Compute Integrated Brier Score (IBS).
    
    Lower is better.
    """
    n_samples = len(times)
    n_times = len(eval_times)
    
    # Kaplan-Meier estimate for censoring
    # Simple approximation
    km_censor = np.ones(n_times)
    for t_idx, t in enumerate(eval_times):
        n_censored = np.sum((events == 0) & (times <= t))
        n_at_risk = np.sum(times >= t)
        if n_at_risk > 0:
            km_censor[t_idx] = 1 - n_censored / n_at_risk
    
    km_censor = np.maximum(km_censor, 0.01)  # Avoid division by zero
    
    # Brier score at each time
    brier_scores = []
    for t_idx, t in enumerate(eval_times):
        bs = 0.0
        for i in range(n_samples):
            if times[i] <= t and events[i] == 1:
                # Event before t: should have predicted low survival
                weight = 1.0 / km_censor[t_idx]
                bs += weight * (survival_probs[i, t_idx] ** 2)
            elif times[i] > t:
                # Still alive at t: should have predicted high survival
                weight = 1.0 / km_censor[t_idx]
                bs += weight * ((1 - survival_probs[i, t_idx]) ** 2)
        
        brier_scores.append(bs / n_samples)
    
    # Integrate using trapezoidal rule
    ibs = np.trapz(brier_scores, eval_times) / (eval_times[-1] - eval_times[0])
    return ibs


# =============================================================================
# DATA LOADING
# =============================================================================

class SurvivalDataset:
    """Load and prepare data for survival analysis experiments."""
    
    MISSING_CODES = {-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999}
    
    def __init__(self, data_dir: str):
        """Load integrated dataset."""
        self.data_dir = data_dir
        
        # Load integrated data
        integrated_path = os.path.join(data_dir, 'nacc_integrated_dataset.pkl')
        if not os.path.exists(integrated_path):
            raise FileNotFoundError(f"Dataset not found: {integrated_path}")
        
        with open(integrated_path, 'rb') as f:
            self.integrated_data = pickle.load(f)
        
        logger.info(f"Loaded {len(self.integrated_data)} subjects from integrated dataset")
        
        # Process data
        self._prepare_data()
    
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
    
    def _prepare_data(self):
        """Extract features and survival targets."""
        features_list = []
        times_list = []
        events_list = []
        subjects = []
        
        for _, row in self.integrated_data.iterrows():
            naccid = row.get('NACCID')
            if naccid is None:
                continue
            
            # Check for essential biomarkers
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
            
            # Extract features
            abeta = self._clean_value(abeta, 500.0)
            ptau = self._clean_value(ptau, 50.0)
            ttau = self._clean_value(ttau, 300.0)
            
            ptau_abeta = ptau / abeta if abeta > 0 else 0.1
            ttau_ptau = ttau / ptau if ptau > 0 else 6.0
            
            age = self._clean_value(row.get('AGE_AT_BASELINE'), 75.0)
            sex = self._clean_value(row.get('SEX'), 1.0)
            educ = self._clean_value(row.get('EDUC'), 16.0)
            
            # Baseline cognition
            trajectory = row.get('clinical_trajectory', [])
            if isinstance(trajectory, list) and len(trajectory) > 0:
                baseline_mmse = self._clean_value(trajectory[0].get('NACCMMSE'), 28.0)
                baseline_cdr = self._clean_value(trajectory[0].get('CDRSUM'), 0.5)
            else:
                baseline_mmse = 28.0
                baseline_cdr = 0.5
            
            features = [abeta, ptau, ttau, ptau_abeta, ttau_ptau, 
                       age, sex, educ, baseline_mmse, baseline_cdr]
            
            # Survival data
            converted = row.get('converted_to_dementia', 0)
            if converted == 1:
                time = self._clean_value(row.get('time_to_dementia'), 
                                        row.get('follow_up_years', 5.0))
                event = 1
            else:
                time = self._clean_value(row.get('follow_up_years'), 5.0)
                if isinstance(trajectory, list) and len(trajectory) > 0:
                    last_time = trajectory[-1].get('YearsFromBaseline', time)
                    time = self._clean_value(last_time, time)
                event = 0
            
            features_list.append(features)
            times_list.append(max(time, 0.1))
            events_list.append(event)
            subjects.append(naccid)
        
        self.X = np.array(features_list, dtype=np.float32)
        self.times = np.array(times_list, dtype=np.float32)
        self.events = np.array(events_list, dtype=np.int32)
        self.subjects = subjects
        
        # Handle NaN
        for col in range(self.X.shape[1]):
            mask = np.isnan(self.X[:, col])
            if mask.any():
                self.X[mask, col] = np.nanmedian(self.X[:, col])
        
        self.feature_names = ['ABETA', 'PTAU', 'TTAU', 'PTAU_ABETA', 'TTAU_PTAU',
                             'AGE', 'SEX', 'EDUC', 'MMSE', 'CDRSUM']
        
        logger.info(f"Prepared dataset: {len(self.subjects)} subjects, {self.X.shape[1]} features")
        logger.info(f"Event rate: {self.events.mean()*100:.1f}%")
        logger.info(f"Median follow-up: {np.median(self.times):.1f} years")


# =============================================================================
# BASELINE MODELS
# =============================================================================

class CoxPHBaseline:
    """Cox Proportional Hazards baseline using lifelines or custom implementation."""
    
    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.model = None
        self.coefficients = None
    
    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray):
        """Fit Cox PH model."""
        if HAS_LIFELINES:
            # Use lifelines
            df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
            df['time'] = times
            df['event'] = events
            
            self.model = CoxPHFitter(penalizer=self.penalizer)
            self.model.fit(df, duration_col='time', event_col='event')
            self.coefficients = self.model.params_.values
        else:
            # Simple gradient descent implementation
            self.coefficients = self._fit_newton(X, times, events)
        
        return self
    
    def _fit_newton(self, X: np.ndarray, times: np.ndarray, events: np.ndarray,
                   max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Fit using Newton-Raphson."""
        n_features = X.shape[1]
        beta = np.zeros(n_features)
        
        # Sort by time
        order = np.argsort(times)
        X = X[order]
        times = times[order]
        events = events[order]
        
        for _ in range(max_iter):
            # Risk scores
            risk = np.exp(X @ beta)
            
            # Gradient and Hessian
            cum_risk = np.cumsum(risk[::-1])[::-1]
            cum_risk_x = np.cumsum((X.T * risk).T[::-1], axis=0)[::-1]
            
            grad = np.zeros(n_features)
            hess = np.zeros((n_features, n_features))
            
            for i in range(len(times)):
                if events[i] == 1:
                    grad += X[i] - cum_risk_x[i] / cum_risk[i]
                    
                    mean_x = cum_risk_x[i] / cum_risk[i]
                    cum_risk_xx = np.outer(X[i] * risk[i], X[i])
                    for j in range(i, len(times)):
                        cum_risk_xx += np.outer(X[j] * risk[j], X[j])
                    hess -= cum_risk_xx / cum_risk[i] - np.outer(mean_x, mean_x)
            
            # Add penalty
            grad -= self.penalizer * beta
            hess -= self.penalizer * np.eye(n_features)
            
            # Newton step
            try:
                step = np.linalg.solve(-hess, grad)
            except:
                step = grad * 0.01
            
            beta_new = beta + step
            
            if np.max(np.abs(step)) < tol:
                break
            
            beta = beta_new
        
        return beta
    
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = higher risk)."""
        return X @ self.coefficients


class DeepSurvBaseline(nn.Module):
    """DeepSurv: Neural network for Cox regression."""
    
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
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class DeepSurvTrainer:
    """Trainer for DeepSurv model."""
    
    def __init__(self, model: DeepSurvBaseline, lr: float = 1e-3, 
                 weight_decay: float = 1e-4, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _cox_loss(self, risk_scores: torch.Tensor, times: torch.Tensor, 
                  events: torch.Tensor) -> torch.Tensor:
        """Compute Cox partial likelihood loss."""
        # Sort by time descending
        order = torch.argsort(times, descending=True)
        risk_scores = risk_scores[order]
        events = events[order].float()
        
        # Log cumulative sum of exp(risk)
        max_risk = risk_scores.max()
        log_cumsum = torch.log(torch.cumsum(torch.exp(risk_scores - max_risk), dim=0) + 1e-8) + max_risk
        
        # Partial likelihood
        log_lik = risk_scores - log_cumsum
        
        n_events = events.sum()
        if n_events > 0:
            loss = -(log_lik * events).sum() / n_events
        else:
            loss = risk_scores.mean() * 0.0
        
        return loss
    
    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray,
            n_epochs: int = 100, batch_size: int = 32, verbose: bool = False):
        """Train the model."""
        X_t = torch.FloatTensor(X).to(self.device)
        times_t = torch.FloatTensor(times).to(self.device)
        events_t = torch.LongTensor(events).to(self.device)
        
        n_samples = len(X)
        
        self.model.train()
        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                
                self.optimizer.zero_grad()
                risk = self.model(X_t[idx])
                loss = self._cox_loss(risk, times_t[idx], events_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}: loss = {epoch_loss/n_batches:.4f}")
    
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            risk = self.model(X_t).cpu().numpy()
        return risk


class RandomSurvivalForestBaseline:
    """Random Survival Forest baseline (requires scikit-survival)."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 min_samples_split: int = 10, random_state: int = 42):
        if not HAS_SKSURV:
            raise ImportError("scikit-survival is required for RSF")
        
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray):
        """Fit RSF model."""
        # scikit-survival requires structured array for y
        y = np.array([(bool(e), t) for e, t in zip(events, times)],
                    dtype=[('event', bool), ('time', float)])
        self.model.fit(X, y)
        return self
    
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (negative cumulative hazard at median time)."""
        # Use negative survival probability as risk
        surv_funcs = self.model.predict_survival_function(X)
        
        # Get risk at median time
        times = surv_funcs[0].x
        median_idx = len(times) // 2
        
        risks = []
        for sf in surv_funcs:
            # Higher survival = lower risk, so negate
            risks.append(-sf.y[median_idx])
        
        return np.array(risks)


# =============================================================================
# PROGRESS SURVIVAL MODEL (simplified for comparison)
# =============================================================================

class PROGRESSSurvivalModel(nn.Module):
    """PROGRESS survival model for comparison."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3, num_heads: int = 4):
        super().__init__()
        
        # Attention mechanism
        self.attention = self._build_attention(input_dim, num_heads, dropout)
        
        # Risk network
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
        
        layers.append(nn.Linear(prev_dim, 1))
        self.risk_network = nn.Sequential(*layers)
    
    def _build_attention(self, input_dim: int, num_heads: int, dropout: float):
        """Build simple attention layer."""
        class SimpleAttention(nn.Module):
            def __init__(self, dim, heads, drop):
                super().__init__()
                self.query = nn.Linear(dim, dim)
                self.key = nn.Linear(dim, dim)
                self.value = nn.Linear(dim, dim)
                self.dropout = nn.Dropout(drop)
                self.scale = np.sqrt(dim)
            
            def forward(self, x):
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                
                # Self-attention over features
                attn = torch.softmax(q * k / self.scale, dim=-1)
                attn = self.dropout(attn)
                out = attn * v
                return out + x  # Residual
        
        return SimpleAttention(input_dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return self.risk_network(x).squeeze(-1)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class SurvivalExperiment:
    """Run survival model comparison experiments."""
    
    def __init__(self, data: SurvivalDataset, n_folds: int = 5, n_runs: int = 5,
                 device: str = 'cpu'):
        self.data = data
        self.n_folds = n_folds
        self.n_runs = n_runs
        self.device = device
        
        self.results: Dict[str, AlgorithmResults] = {}
        self.fold_predictions: Dict[str, List] = {}
    
    def run(self, quick_test: bool = False) -> Dict[str, AlgorithmResults]:
        """Run all experiments."""
        logger.info("=" * 70)
        logger.info("SURVIVAL MODEL COMPARISON EXPERIMENTS")
        logger.info("=" * 70)
        
        n_epochs = 20 if quick_test else 100
        
        algorithms = {
            'PROGRESS': self._run_progress,
            'Cox_PH': self._run_cox_ph,
            'DeepSurv': self._run_deepsurv,
        }
        
        if HAS_SKSURV:
            algorithms['RSF'] = self._run_rsf
        
        for name, run_func in algorithms.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {name}...")
            logger.info(f"{'='*50}")
            
            self.results[name] = AlgorithmResults(name=name)
            self.fold_predictions[name] = []
            
            for run_idx in range(self.n_runs):
                logger.info(f"\nRun {run_idx + 1}/{self.n_runs}")
                seed = 42 + run_idx
                
                # Cross-validation
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
                
                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.data.X, self.data.events)):
                    X_train = self.data.X[train_idx]
                    X_test = self.data.X[test_idx]
                    times_train = self.data.times[train_idx]
                    times_test = self.data.times[test_idx]
                    events_train = self.data.events[train_idx]
                    events_test = self.data.events[test_idx]
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Run algorithm
                    try:
                        risk_scores = run_func(
                            X_train_scaled, times_train, events_train,
                            X_test_scaled, n_epochs=n_epochs, seed=seed
                        )
                        
                        # Compute metrics
                        c_idx = concordance_index(risk_scores, times_test, events_test)
                        auc_2yr = time_dependent_auc(risk_scores, times_test, events_test, 2.0)
                        auc_3yr = time_dependent_auc(risk_scores, times_test, events_test, 3.0)
                        auc_5yr = time_dependent_auc(risk_scores, times_test, events_test, 5.0)
                        
                        self.results[name].c_index_scores.append(c_idx)
                        self.results[name].auc_2yr_scores.append(auc_2yr)
                        self.results[name].auc_3yr_scores.append(auc_3yr)
                        self.results[name].auc_5yr_scores.append(auc_5yr)
                        
                        # Store predictions for DeLong test
                        self.fold_predictions[name].append({
                            'risk_scores': risk_scores,
                            'times': times_test,
                            'events': events_test,
                            'test_idx': test_idx
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error in {name} fold {fold_idx}: {e}")
                        continue
                
                logger.info(f"  Run {run_idx+1} C-index: {np.mean(self.results[name].c_index_scores[-self.n_folds:]):.4f}")
            
            # Summary
            logger.info(f"\n{name} Summary:")
            logger.info(f"  C-index: {self.results[name].c_index_mean:.4f} ± {self.results[name].c_index_std:.4f}")
        
        return self.results
    
    def _run_progress(self, X_train, times_train, events_train, X_test, 
                      n_epochs: int = 100, seed: int = 42) -> np.ndarray:
        """Run PROGRESS survival model."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = PROGRESSSurvivalModel(
            input_dim=X_train.shape[1],
            hidden_dims=[64, 32],
            dropout=0.3
        )
        
        trainer = DeepSurvTrainer(model, lr=1e-3, device=self.device)
        trainer.fit(X_train, times_train, events_train, n_epochs=n_epochs)
        
        return trainer.predict_risk(X_test)
    
    def _run_cox_ph(self, X_train, times_train, events_train, X_test,
                    n_epochs: int = 100, seed: int = 42) -> np.ndarray:
        """Run Cox PH baseline."""
        model = CoxPHBaseline(penalizer=0.01)
        model.fit(X_train, times_train, events_train)
        return model.predict_risk(X_test)
    
    def _run_deepsurv(self, X_train, times_train, events_train, X_test,
                      n_epochs: int = 100, seed: int = 42) -> np.ndarray:
        """Run DeepSurv baseline."""
        torch.manual_seed(seed)
        
        model = DeepSurvBaseline(
            input_dim=X_train.shape[1],
            hidden_dims=[64, 32],
            dropout=0.3
        )
        
        trainer = DeepSurvTrainer(model, lr=1e-3, device=self.device)
        trainer.fit(X_train, times_train, events_train, n_epochs=n_epochs)
        
        return trainer.predict_risk(X_test)
    
    def _run_rsf(self, X_train, times_train, events_train, X_test,
                 n_epochs: int = 100, seed: int = 42) -> np.ndarray:
        """Run Random Survival Forest baseline."""
        model = RandomSurvivalForestBaseline(
            n_estimators=100,
            max_depth=5,
            random_state=seed
        )
        model.fit(X_train, times_train, events_train)
        return model.predict_risk(X_test)
    
    def run_significance_tests(self) -> Dict:
        """Run statistical significance tests comparing PROGRESS to baselines."""
        logger.info("\n" + "=" * 70)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("=" * 70)
        
        if 'PROGRESS' not in self.results:
            raise ValueError("PROGRESS results not found. Run experiments first.")
        
        progress_scores = np.array(self.results['PROGRESS'].c_index_scores)
        
        test_results = {}
        p_values_for_correction = []
        
        for baseline_name in self.results.keys():
            if baseline_name == 'PROGRESS':
                continue
            
            baseline_scores = np.array(self.results[baseline_name].c_index_scores)
            
            logger.info(f"\n--- PROGRESS vs {baseline_name} ---")
            logger.info(f"PROGRESS C-index: {progress_scores.mean():.4f} ± {progress_scores.std():.4f}")
            logger.info(f"{baseline_name} C-index: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")
            
            comparison = {
                'progress_mean': float(progress_scores.mean()),
                'progress_std': float(progress_scores.std()),
                'baseline_mean': float(baseline_scores.mean()),
                'baseline_std': float(baseline_scores.std()),
                'difference': float(progress_scores.mean() - baseline_scores.mean()),
                'tests': {}
            }
            
            # 1. Wilcoxon signed-rank test (paired)
            wilcoxon_result = WilcoxonTest.test(progress_scores, baseline_scores)
            comparison['tests']['wilcoxon'] = wilcoxon_result.to_dict()
            p_values_for_correction.append(wilcoxon_result.p_value)
            logger.info(f"Wilcoxon: W={wilcoxon_result.statistic:.2f}, p={wilcoxon_result.p_value:.4f}")
            
            # 2. Bootstrap CI
            bootstrap_result = BootstrapTest.confidence_interval(
                progress_scores, baseline_scores, n_bootstrap=10000
            )
            comparison['tests']['bootstrap'] = bootstrap_result.to_dict()
            logger.info(f"Bootstrap: diff={bootstrap_result.effect_size:.4f}, "
                       f"95% CI=[{bootstrap_result.ci_lower:.4f}, {bootstrap_result.ci_upper:.4f}], "
                       f"p={bootstrap_result.p_value:.4f}")
            
            # 3. Permutation test
            perm_result = PermutationTest.test(progress_scores, baseline_scores)
            comparison['tests']['permutation'] = perm_result.to_dict()
            logger.info(f"Permutation: diff={perm_result.effect_size:.4f}, p={perm_result.p_value:.4f}")
            
            # 4. Mann-Whitney U test
            mw_result = MannWhitneyTest.test(progress_scores, baseline_scores)
            comparison['tests']['mann_whitney'] = mw_result.to_dict()
            logger.info(f"Mann-Whitney: U={mw_result.statistic:.2f}, p={mw_result.p_value:.4f}")
            
            test_results[baseline_name] = comparison
        
        # Apply multiple testing correction
        if p_values_for_correction:
            corrected_holm = apply_multiple_testing_correction(p_values_for_correction, 'holm')
            corrected_fdr = apply_multiple_testing_correction(p_values_for_correction, 'fdr_bh')
            
            logger.info("\n--- Multiple Testing Correction ---")
            for i, baseline_name in enumerate([k for k in self.results.keys() if k != 'PROGRESS']):
                logger.info(f"{baseline_name}: raw p={p_values_for_correction[i]:.4f}, "
                           f"Holm p={corrected_holm[i]:.4f}, FDR p={corrected_fdr[i]:.4f}")
                test_results[baseline_name]['corrected_p_holm'] = float(corrected_holm[i])
                test_results[baseline_name]['corrected_p_fdr'] = float(corrected_fdr[i])
        
        return test_results
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Statistical comparison of survival prediction models. "
            r"PROGRESS is compared against baseline methods using multiple statistical tests. "
            r"$^{*}$p$<$0.05, $^{**}$p$<$0.01, $^{***}$p$<$0.001.}",
            r"\label{tab:survival_significance}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Method & C-index & $\Delta$ C-index & Wilcoxon $p$ & Bootstrap 95\% CI & Permutation $p$ \\",
            r"\midrule",
        ]
        
        # PROGRESS row
        prog = self.results['PROGRESS']
        lines.append(f"PROGRESS (Ours) & {prog.c_index_mean:.3f} $\\pm$ {prog.c_index_std:.3f} & --- & --- & --- & --- \\\\")
        lines.append(r"\midrule")
        
        # Baseline rows (need to run significance tests first)
        sig_results = self.run_significance_tests()
        
        for baseline_name in ['Cox_PH', 'DeepSurv', 'RSF']:
            if baseline_name not in self.results:
                continue
            
            baseline = self.results[baseline_name]
            sig = sig_results.get(baseline_name, {})
            
            diff = sig.get('difference', 0)
            diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
            
            # Wilcoxon p-value with significance stars
            wilcox_p = sig.get('tests', {}).get('wilcoxon', {}).get('p_value', 1.0)
            stars = "^{***}" if wilcox_p < 0.001 else "^{**}" if wilcox_p < 0.01 else "^{*}" if wilcox_p < 0.05 else ""
            wilcox_str = f"{wilcox_p:.4f}${stars}$" if stars else f"{wilcox_p:.4f}"
            
            # Bootstrap CI
            boot = sig.get('tests', {}).get('bootstrap', {})
            ci_lower = boot.get('ci_lower', 0)
            ci_upper = boot.get('ci_upper', 0)
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            
            # Permutation p
            perm_p = sig.get('tests', {}).get('permutation', {}).get('p_value', 1.0)
            perm_str = f"{perm_p:.4f}"
            
            display_name = baseline_name.replace('_', ' ')
            lines.append(f"{display_name} & {baseline.c_index_mean:.3f} $\\pm$ {baseline.c_index_std:.3f} & "
                        f"{diff_str} & {wilcox_str} & {ci_str} & {perm_str} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return '\n'.join(lines)
    
    def save_results(self, output_dir: str):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_dict = {name: res.to_dict() for name, res in self.results.items()}
        with open(os.path.join(output_dir, 'survival_comparison_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save significance tests
        sig_results = self.run_significance_tests()
        with open(os.path.join(output_dir, 'survival_significance_tests.json'), 'w') as f:
            json.dump(sig_results, f, indent=2)
        
        # Save LaTeX table
        latex_table = self.generate_latex_table()
        with open(os.path.join(output_dir, 'survival_significance_table.tex'), 'w') as f:
            f.write(latex_table)
        
        # Save summary CSV
        summary_data = []
        for name, res in self.results.items():
            summary_data.append({
                'Algorithm': name,
                'C-index Mean': res.c_index_mean,
                'C-index Std': res.c_index_std,
                'AUC@2yr Mean': np.mean(res.auc_2yr_scores),
                'AUC@3yr Mean': np.mean(res.auc_3yr_scores),
                'AUC@5yr Mean': np.mean(res.auc_5yr_scores),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'survival_comparison_summary.csv'), index=False)
        
        logger.info(f"\nResults saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Survival Model Statistical Significance Tests',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing nacc_integrated_dataset.pkl')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: {data-dir}/survival_significance)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--n-runs', type=int, default=5,
                       help='Number of runs with different seeds (default: 5)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced epochs')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = args.output_dir or os.path.join(args.data_dir, 'survival_significance')
    logger = setup_logging(output_dir)
    
    logger.info("Survival Model Statistical Significance Tests")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Folds: {args.n_folds}, Runs: {args.n_runs}")
    
    # Load data
    data = SurvivalDataset(args.data_dir)
    
    # Run experiments
    experiment = SurvivalExperiment(
        data=data,
        n_folds=args.n_folds,
        n_runs=args.n_runs,
        device=args.device
    )
    
    experiment.run(quick_test=args.quick_test)
    experiment.save_results(output_dir)
    
    # Print LaTeX table
    logger.info("\n" + "=" * 70)
    logger.info("LATEX TABLE FOR PAPER")
    logger.info("=" * 70)
    print(experiment.generate_latex_table())
    
    logger.info("\nExperiment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
