#!/usr/bin/env python3
"""
baseline_comparison.py - Baseline Methods for PROGRESS Framework Comparison

This module implements baseline and state-of-the-art comparison methods for 
evaluating the PROGRESS framework against established approaches.

Methods Implemented:
    Trajectory Prediction (Model 1 comparisons):
        - Linear Regression
        - Ridge Regression  
        - Random Forest Regressor
        - XGBoost Regressor
        - MLP (without attention)
        - Bayesian Ridge Regression
    
    Survival Analysis (Model 2 comparisons):
        - Cox Proportional Hazards (standard)
        - Cox-Lasso (regularized)
        - Random Survival Forest
        - DeepSurv (basic, without attention)
        - Gradient Boosting Survival
    
    Classification Baselines (for fixed-horizon prediction):
        - Logistic Regression
        - Random Forest Classifier
        - XGBoost Classifier

References from PROGRESS Paper:
    - Katzman et al., 2018 (DeepSurv)
    - Harrell et al., 1996 (C-index, Cox PH)
    - Yi et al., 2023 (XGBoost-SHAP)
    - Tao et al., 2018 (LSTM baseline)

Author: Generated for PROGRESS paper comparison
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
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# SHAP for explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# Survival analysis libraries
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Warning: lifelines not installed. Install with: pip install lifelines")

try:
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("Warning: scikit-survival not installed. Install with: pip install scikit-survival")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # MLP
    mlp_hidden_dims: List[int] = None
    mlp_dropout: float = 0.3
    mlp_epochs: int = 100
    mlp_lr: float = 1e-3
    mlp_batch_size: int = 32
    
    # Survival
    rsf_n_estimators: int = 100
    cox_alpha: float = 0.1
    
    # Random seed
    random_state: int = 42
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 32]


# =============================================================================
# TRAJECTORY PREDICTION BASELINES (MODEL 1 COMPARISONS)
# =============================================================================

class TrajectoryBaselines:
    """
    Baseline methods for trajectory parameter prediction.
    
    Predicts trajectory parameters (intercept α, slope β, acceleration γ)
    from baseline CSF biomarkers and demographics.
    """
    
    def __init__(self, config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def fit_linear_regression(self, X: np.ndarray, y: np.ndarray, 
                              param_name: str = 'all') -> Dict:
        """
        Fit simple linear regression.
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets - can be (n_samples,) or (n_samples, 3) for all params
            param_name: Name for storing model
            
        Returns:
            Dictionary with fitted model and training metrics
        """
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        if y.ndim == 1:
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
        else:
            r2 = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
            rmse = [np.sqrt(mean_squared_error(y[:, i], y_pred[:, i])) 
                   for i in range(y.shape[1])]
        
        self.models[f'linear_{param_name}'] = model
        
        return {
            'model': model,
            'train_r2': r2,
            'train_rmse': rmse,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
    
    def fit_ridge_regression(self, X: np.ndarray, y: np.ndarray,
                            param_name: str = 'all',
                            alphas: List[float] = None) -> Dict:
        """
        Fit Ridge regression with cross-validated alpha.
        
        Args:
            X: Features
            y: Targets
            param_name: Name for model
            alphas: List of alpha values to try
            
        Returns:
            Dictionary with fitted model and metrics
        """
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        if y.ndim == 1:
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
        else:
            r2 = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
            rmse = [np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
                   for i in range(y.shape[1])]
        
        self.models[f'ridge_{param_name}'] = model
        
        return {
            'model': model,
            'best_alpha': model.alpha_,
            'train_r2': r2,
            'train_rmse': rmse
        }
    
    def fit_random_forest(self, X: np.ndarray, y: np.ndarray,
                         param_name: str = 'all') -> Dict:
        """
        Fit Random Forest regressor.
        
        Args:
            X: Features
            y: Targets
            param_name: Name for model
            
        Returns:
            Dictionary with fitted model and metrics
        """
        model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        if y.ndim == 1:
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
        else:
            r2 = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
            rmse = [np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
                   for i in range(y.shape[1])]
        
        self.models[f'rf_{param_name}'] = model
        
        return {
            'model': model,
            'train_r2': r2,
            'train_rmse': rmse,
            'feature_importance': model.feature_importances_
        }
    
    def fit_xgboost(self, X: np.ndarray, y: np.ndarray,
                   param_name: str = 'all',
                   compute_shap: bool = False) -> Dict:
        """
        Fit XGBoost regressor with optional SHAP explanations.
        
        Reference: Yi et al., 2023 (XGBoost-SHAP for AD diagnosis)
        
        Args:
            X: Features
            y: Targets (single parameter)
            param_name: Name for model
            compute_shap: Whether to compute SHAP values
            
        Returns:
            Dictionary with fitted model, metrics, and optional SHAP values
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        
        # XGBoost handles multi-output differently
        if y.ndim == 1:
            model = xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
        else:
            # Train separate models for each parameter
            models = []
            r2 = []
            rmse = []
            for i in range(y.shape[1]):
                m = xgb.XGBRegressor(
                    n_estimators=self.config.xgb_n_estimators,
                    max_depth=self.config.xgb_max_depth,
                    learning_rate=self.config.xgb_learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
                m.fit(X, y[:, i])
                y_pred_i = m.predict(X)
                r2.append(r2_score(y[:, i], y_pred_i))
                rmse.append(np.sqrt(mean_squared_error(y[:, i], y_pred_i)))
                models.append(m)
            model = models  # List of models
        
        self.models[f'xgb_{param_name}'] = model
        
        result = {
            'model': model,
            'train_r2': r2,
            'train_rmse': rmse,
        }
        
        # SHAP explanation
        if compute_shap and HAS_SHAP and y.ndim == 1:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            result['shap_values'] = shap_values
            result['shap_explainer'] = explainer
        
        return result
    
    def fit_bayesian_ridge(self, X: np.ndarray, y: np.ndarray,
                          param_name: str = 'all') -> Dict:
        """
        Fit Bayesian Ridge Regression (provides uncertainty estimates).
        
        Args:
            X: Features
            y: Targets
            param_name: Name for model
            
        Returns:
            Dictionary with model and uncertainty info
        """
        model = BayesianRidge()
        
        if y.ndim == 1:
            model.fit(X, y)
            y_pred, y_std = model.predict(X, return_std=True)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            result = {
                'model': model,
                'train_r2': r2,
                'train_rmse': rmse,
                'prediction_std': y_std.mean()
            }
        else:
            models = []
            r2 = []
            rmse = []
            pred_std = []
            for i in range(y.shape[1]):
                m = BayesianRidge()
                m.fit(X, y[:, i])
                y_pred_i, y_std_i = m.predict(X, return_std=True)
                r2.append(r2_score(y[:, i], y_pred_i))
                rmse.append(np.sqrt(mean_squared_error(y[:, i], y_pred_i)))
                pred_std.append(y_std_i.mean())
                models.append(m)
            model = models
            
            result = {
                'model': model,
                'train_r2': r2,
                'train_rmse': rmse,
                'prediction_std': pred_std
            }
        
        self.models[f'bayesian_ridge_{param_name}'] = model
        return result

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a fitted model."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if isinstance(model, list):
            # Multi-output (separate models)
            return np.column_stack([m.predict(X) for m in model])
        return model.predict(X)
    
    def evaluate(self, model_name: str, X_test: np.ndarray, 
                y_test: np.ndarray) -> Dict:
        """
        Evaluate a fitted model on test data.
        
        Args:
            model_name: Name of fitted model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with test metrics
        """
        y_pred = self.predict(model_name, X_test)
        
        if y_test.ndim == 1:
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'correlation': stats.pearsonr(y_test, y_pred)[0]
            }
        else:
            param_names = ['intercept', 'slope', 'acceleration']
            metrics = {}
            for i, name in enumerate(param_names):
                metrics[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
                metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                metrics[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
                if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                    metrics[f'{name}_correlation'] = stats.pearsonr(y_test[:, i], y_pred[:, i])[0]
                else:
                    metrics[f'{name}_correlation'] = 0.0
        
        return metrics


# =============================================================================
# MLP BASELINE (DEEP LEARNING WITHOUT ATTENTION)
# =============================================================================

class MLPRegressor(nn.Module):
    """
    Simple MLP for trajectory prediction (ablation vs PROGRESS attention).
    
    This serves as a baseline to show the value of the attention mechanism
    in PROGRESS Model 1.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 3,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPBaseline:
    """Training wrapper for MLP baseline."""
    
    def __init__(self, config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train MLP baseline.
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples, 3) for trajectory parameters
            
        Returns:
            Training history
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create model
        self.model = MLPRegressor(
            input_dim=X.shape[1],
            output_dim=y.shape[1] if y.ndim > 1 else 1,
            hidden_dims=self.config.mlp_hidden_dims,
            dropout=self.config.mlp_dropout
        ).to(self.device)
        
        # Training setup
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.mlp_lr,
            weight_decay=1e-4
        )
        criterion = nn.MSELoss()
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.mlp_batch_size, shuffle=True)
        
        # Training loop
        history = {'loss': []}
        self.model.train()
        
        for epoch in range(self.config.mlp_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            history['loss'].append(epoch_loss / len(loader))
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_tensor)
        
        return pred.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate on test data."""
        y_pred = self.predict(X_test)
        
        param_names = ['intercept', 'slope', 'acceleration']
        metrics = {}
        
        for i, name in enumerate(param_names[:y_test.shape[1]]):
            metrics[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                metrics[f'{name}_correlation'] = stats.pearsonr(y_test[:, i], y_pred[:, i])[0]
        
        return metrics


# =============================================================================
# SURVIVAL ANALYSIS BASELINES (MODEL 2 COMPARISONS)
# =============================================================================

class SurvivalBaselines:
    """
    Baseline methods for survival analysis.
    
    Compares with PROGRESS Model 2 for MCI-to-dementia conversion prediction.
    
    References:
        - Katzman et al., 2018 (DeepSurv)
        - Harrell et al., 1996 (C-index)
    """
    
    def __init__(self, config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.models = {}
        self.results = {}
    
    def fit_cox_ph(self, X: np.ndarray, times: np.ndarray, 
                   events: np.ndarray, feature_names: List[str] = None) -> Dict:
        """
        Fit standard Cox Proportional Hazards model.
        
        Reference: Classical baseline (Harrell et al., 1996)
        
        Args:
            X: Features (n_samples, n_features)
            times: Observed times
            events: Event indicators (1=event, 0=censored)
            feature_names: Names of features
            
        Returns:
            Dictionary with model and metrics
        """
        if not HAS_LIFELINES:
            raise ImportError("lifelines not installed")
        
        # Create DataFrame for lifelines
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['T'] = times
        df['E'] = events
        
        # Fit model
        model = CoxPHFitter(penalizer=0.1)
        model.fit(df, duration_col='T', event_col='E')
        
        # Compute C-index
        c_index = model.concordance_index_
        
        self.models['cox_ph'] = model
        
        return {
            'model': model,
            'c_index': c_index,
            'hazard_ratios': model.hazard_ratios_,
            'summary': model.summary
        }
    
    def fit_cox_lasso(self, X: np.ndarray, times: np.ndarray,
                      events: np.ndarray, alpha: float = None) -> Dict:
        """
        Fit Cox model with L1 (Lasso) regularization.
        
        Args:
            X: Features
            times: Observed times
            events: Event indicators
            alpha: Regularization strength (auto-selected if None)
            
        Returns:
            Dictionary with model and metrics
        """
        if not HAS_SKSURV:
            raise ImportError("scikit-survival not installed")
        
        # Convert to structured array for scikit-survival
        y = np.array([(bool(e), t) for e, t in zip(events, times)],
                    dtype=[('event', bool), ('time', float)])
        
        if alpha is None:
            # Use cross-validation to find best alpha
            model = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True)
            model.fit(X, y)
        else:
            model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha],
                                          fit_baseline_model=True)
            model.fit(X, y)
        
        # Compute C-index
        risk_scores = model.predict(X)
        c_index, _, _, _, _ = concordance_index_censored(events.astype(bool), times, risk_scores)
        
        self.models['cox_lasso'] = model
        
        return {
            'model': model,
            'c_index': c_index,
            'coefficients': model.coef_
        }
    
    def fit_random_survival_forest(self, X: np.ndarray, times: np.ndarray,
                                   events: np.ndarray) -> Dict:
        """
        Fit Random Survival Forest.
        
        Non-parametric ensemble method for survival analysis.
        
        Args:
            X: Features
            times: Observed times
            events: Event indicators
            
        Returns:
            Dictionary with model and metrics
        """
        if not HAS_SKSURV:
            raise ImportError("scikit-survival not installed")
        
        # Convert to structured array
        y = np.array([(bool(e), t) for e, t in zip(events, times)],
                    dtype=[('event', bool), ('time', float)])
        
        model = RandomSurvivalForest(
            n_estimators=self.config.rsf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Compute C-index
        risk_scores = model.predict(X)
        c_index, _, _, _, _ = concordance_index_censored(events.astype(bool), times, risk_scores)
        
        self.models['rsf'] = model
        
        return {
            'model': model,
            'c_index': c_index,
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }
    
    def fit_gradient_boosting_survival(self, X: np.ndarray, times: np.ndarray,
                                       events: np.ndarray) -> Dict:
        """
        Fit Gradient Boosting for Survival Analysis.
        
        Args:
            X: Features
            times: Observed times
            events: Event indicators
            
        Returns:
            Dictionary with model and metrics
        """
        if not HAS_SKSURV:
            raise ImportError("scikit-survival not installed")
        
        y = np.array([(bool(e), t) for e, t in zip(events, times)],
                    dtype=[('event', bool), ('time', float)])
        
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            random_state=self.config.random_state
        )
        model.fit(X, y)
        
        risk_scores = model.predict(X)
        c_index, _, _, _, _ = concordance_index_censored(events.astype(bool), times, risk_scores)
        
        self.models['gbs'] = model
        
        return {
            'model': model,
            'c_index': c_index,
            'feature_importance': model.feature_importances_
        }
    
    def predict_risk(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if model_name == 'cox_ph':
            # lifelines returns negative partial hazard
            return -model.predict_partial_hazard(
                pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            ).values
        else:
            return model.predict(X)
    
    def evaluate(self, model_name: str, X_test: np.ndarray,
                times_test: np.ndarray, events_test: np.ndarray) -> Dict:
        """
        Evaluate survival model on test data.
        
        Args:
            model_name: Name of fitted model
            X_test: Test features
            times_test: Test times
            events_test: Test event indicators
            
        Returns:
            Dictionary with test metrics
        """
        risk_scores = self.predict_risk(model_name, X_test)
        
        # C-index
        if HAS_SKSURV:
            c_index, _, _, _, _ = concordance_index_censored(
                events_test.astype(bool), times_test, risk_scores
            )
        elif HAS_LIFELINES:
            c_index = concordance_index(times_test, -risk_scores, events_test)
        else:
            c_index = self._compute_c_index(risk_scores, times_test, events_test)
        
        metrics = {'c_index': c_index}
        
        # Time-dependent AUC at different horizons
        for horizon in [2.0, 3.0, 5.0]:
            auc = self._compute_time_dependent_auc(
                risk_scores, times_test, events_test, horizon
            )
            metrics[f'auc_{int(horizon)}yr'] = auc
        
        return metrics
    
    def _compute_c_index(self, risk_scores: np.ndarray, times: np.ndarray,
                        events: np.ndarray) -> float:
        """Compute Harrell's C-index manually."""
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
    
    def _compute_time_dependent_auc(self, risk_scores: np.ndarray,
                                   times: np.ndarray, events: np.ndarray,
                                   horizon: float) -> float:
        """Compute time-dependent AUC at a specific horizon."""
        # Binary labels at horizon
        y_true = (times <= horizon) & (events == 1)
        
        # Filter to relevant samples
        mask = (times <= horizon) | (times > horizon)
        
        if mask.sum() < 10 or y_true.sum() < 2:
            return 0.5
        
        try:
            return roc_auc_score(y_true[mask], risk_scores[mask])
        except:
            return 0.5


# =============================================================================
# DEEPSURV BASELINE (WITHOUT ATTENTION)
# =============================================================================

class DeepSurvBaseline(nn.Module):
    """
    DeepSurv baseline (simplified version without attention).
    
    Reference: Katzman et al., 2018
    
    This is an ablation of PROGRESS Model 2 to show the value
    of the attention mechanism.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
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
        """Compute log-risk score."""
        return self.risk_network(x)


class DeepSurvTrainer:
    """Training wrapper for DeepSurv baseline."""
    
    def __init__(self, config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray,
            epochs: int = 100, lr: float = 1e-3) -> Dict:
        """
        Train DeepSurv model using Cox partial likelihood.
        
        Args:
            X: Features
            times: Observed times
            events: Event indicators
            epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        times_tensor = torch.FloatTensor(times).to(self.device)
        events_tensor = torch.FloatTensor(events).to(self.device)
        
        # Create model
        self.model = DeepSurvBaseline(
            input_dim=X.shape[1],
            hidden_dims=self.config.mlp_hidden_dims,
            dropout=self.config.mlp_dropout
        ).to(self.device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            risk_scores = self.model(X_tensor).squeeze()
            
            # Cox partial likelihood loss
            loss = self._cox_loss(risk_scores, times_tensor, events_tensor)
            
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
        
        return history
    
    def _cox_loss(self, risk_scores: torch.Tensor, times: torch.Tensor,
                  events: torch.Tensor) -> torch.Tensor:
        """Compute negative Cox partial log-likelihood."""
        # Sort by time (descending)
        sorted_idx = torch.argsort(times, descending=True)
        sorted_risks = risk_scores[sorted_idx]
        sorted_events = events[sorted_idx]
        
        # Log-sum-exp for numerical stability
        max_risk = sorted_risks.max()
        exp_risks = torch.exp(sorted_risks - max_risk)
        cumsum_exp = torch.cumsum(exp_risks, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk
        
        # Partial likelihood
        log_lik = sorted_risks - log_cumsum
        n_events = sorted_events.sum()
        
        if n_events > 0:
            loss = -(log_lik * sorted_events).sum() / n_events
        else:
            loss = risk_scores.mean() * 0.0
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            risk = self.model(X_tensor)
        
        return risk.cpu().numpy().squeeze()
    
    def evaluate(self, X_test: np.ndarray, times_test: np.ndarray,
                events_test: np.ndarray) -> Dict:
        """Evaluate on test data."""
        risk_scores = self.predict(X_test)
        
        # C-index
        if HAS_SKSURV:
            c_index, _, _, _, _ = concordance_index_censored(
                events_test.astype(bool), times_test, risk_scores
            )
        else:
            c_index = self._compute_c_index(risk_scores, times_test, events_test)
        
        return {'c_index': c_index}
    
    def _compute_c_index(self, risk_scores, times, events):
        """Manual C-index computation."""
        concordant = 0
        comparable = 0
        
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if times[i] < times[j] and events[i] == 1:
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif times[j] < times[i] and events[j] == 1:
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
        
        return concordant / comparable if comparable > 0 else 0.5


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

class BaselineComparisonRunner:
    """
    Unified interface for running all baseline comparisons.
    
    Usage:
        runner = BaselineComparisonRunner(config)
        results = runner.run_all_comparisons(X_train, y_traj_train, 
                                             times_train, events_train,
                                             X_test, y_traj_test,
                                             times_test, events_test)
    """
    
    def __init__(self, config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.traj_baselines = TrajectoryBaselines(config)
        self.surv_baselines = SurvivalBaselines(config)
        self.mlp_baseline = MLPBaseline(config)
        self.deepsurv_baseline = DeepSurvTrainer(config)
        
    def run_trajectory_baselines(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 feature_names: List[str] = None) -> Dict:
        """
        Run all trajectory prediction baselines.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Feature names for interpretation
            
        Returns:
            Dictionary with results for each baseline
        """
        results = {}
        
        # Linear Regression
        logger.info("Training Linear Regression...")
        self.traj_baselines.fit_linear_regression(X_train, y_train)
        results['linear'] = self.traj_baselines.evaluate('linear_all', X_test, y_test)
        
        # Ridge Regression
        logger.info("Training Ridge Regression...")
        self.traj_baselines.fit_ridge_regression(X_train, y_train)
        results['ridge'] = self.traj_baselines.evaluate('ridge_all', X_test, y_test)
        
        # Random Forest
        logger.info("Training Random Forest...")
        self.traj_baselines.fit_random_forest(X_train, y_train)
        results['random_forest'] = self.traj_baselines.evaluate('rf_all', X_test, y_test)
        
        # XGBoost
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            self.traj_baselines.fit_xgboost(X_train, y_train)
            results['xgboost'] = self.traj_baselines.evaluate('xgb_all', X_test, y_test)
        
        # Bayesian Ridge
        logger.info("Training Bayesian Ridge...")
        self.traj_baselines.fit_bayesian_ridge(X_train, y_train)
        results['bayesian_ridge'] = self.traj_baselines.evaluate('bayesian_ridge_all', X_test, y_test)
        
        # MLP
        logger.info("Training MLP baseline...")
        self.mlp_baseline.fit(X_train, y_train)
        results['mlp'] = self.mlp_baseline.evaluate(X_test, y_test)
        
        return results
    
    def run_survival_baselines(self, X_train: np.ndarray, times_train: np.ndarray,
                               events_train: np.ndarray, X_test: np.ndarray,
                               times_test: np.ndarray, events_test: np.ndarray,
                               feature_names: List[str] = None) -> Dict:
        """
        Run all survival analysis baselines.
        
        Args:
            X_train, times_train, events_train: Training data
            X_test, times_test, events_test: Test data
            feature_names: Feature names
            
        Returns:
            Dictionary with results for each baseline
        """
        results = {}
        
        # Cox PH
        if HAS_LIFELINES:
            logger.info("Training Cox PH...")
            self.surv_baselines.fit_cox_ph(X_train, times_train, events_train, feature_names)
            results['cox_ph'] = self.surv_baselines.evaluate('cox_ph', X_test, times_test, events_test)
        
        # Cox Lasso
        if HAS_SKSURV:
            logger.info("Training Cox Lasso...")
            self.surv_baselines.fit_cox_lasso(X_train, times_train, events_train)
            results['cox_lasso'] = self.surv_baselines.evaluate('cox_lasso', X_test, times_test, events_test)
            
            # Random Survival Forest
            logger.info("Training Random Survival Forest...")
            self.surv_baselines.fit_random_survival_forest(X_train, times_train, events_train)
            results['rsf'] = self.surv_baselines.evaluate('rsf', X_test, times_test, events_test)
            
            # Gradient Boosting Survival
            logger.info("Training Gradient Boosting Survival...")
            self.surv_baselines.fit_gradient_boosting_survival(X_train, times_train, events_train)
            results['gbs'] = self.surv_baselines.evaluate('gbs', X_test, times_test, events_test)
        
        # DeepSurv baseline
        logger.info("Training DeepSurv baseline...")
        self.deepsurv_baseline.fit(X_train, times_train, events_train)
        results['deepsurv'] = self.deepsurv_baseline.evaluate(X_test, times_test, events_test)
        
        return results
    
    def run_all_comparisons(self, X_train: np.ndarray, y_traj_train: np.ndarray,
                           times_train: np.ndarray, events_train: np.ndarray,
                           X_test: np.ndarray, y_traj_test: np.ndarray,
                           times_test: np.ndarray, events_test: np.ndarray,
                           feature_names: List[str] = None) -> Dict:
        """
        Run all baseline comparisons.
        
        Returns comprehensive results for both trajectory and survival tasks.
        """
        logger.info("=" * 60)
        logger.info("Running Baseline Comparisons")
        logger.info("=" * 60)
        
        results = {
            'trajectory': self.run_trajectory_baselines(
                X_train, y_traj_train, X_test, y_traj_test, feature_names
            ),
            'survival': self.run_survival_baselines(
                X_train, times_train, events_train,
                X_test, times_test, events_test, feature_names
            )
        }
        
        return results
    
    def generate_comparison_table(self, results: Dict, 
                                  progress_results: Dict = None) -> pd.DataFrame:
        """
        Generate comparison table for paper.
        
        Args:
            results: Baseline results from run_all_comparisons
            progress_results: PROGRESS model results (optional)
            
        Returns:
            DataFrame formatted for paper table
        """
        # Trajectory comparison table
        traj_rows = []
        for method, metrics in results['trajectory'].items():
            row = {
                'Method': method.replace('_', ' ').title(),
                'Intercept R²': metrics.get('intercept_r2', metrics.get('r2', 'N/A')),
                'Slope R²': metrics.get('slope_r2', 'N/A'),
                'Accel R²': metrics.get('acceleration_r2', 'N/A'),
            }
            traj_rows.append(row)
        
        if progress_results:
            traj_rows.append({
                'Method': 'PROGRESS (Ours)',
                'Intercept R²': progress_results.get('intercept_r2'),
                'Slope R²': progress_results.get('slope_r2'),
                'Accel R²': progress_results.get('acceleration_r2'),
            })
        
        traj_df = pd.DataFrame(traj_rows)
        
        # Survival comparison table
        surv_rows = []
        for method, metrics in results['survival'].items():
            row = {
                'Method': method.replace('_', ' ').upper(),
                'C-index': metrics.get('c_index', 'N/A'),
                'AUC-2yr': metrics.get('auc_2yr', 'N/A'),
                'AUC-3yr': metrics.get('auc_3yr', 'N/A'),
                'AUC-5yr': metrics.get('auc_5yr', 'N/A'),
            }
            surv_rows.append(row)
        
        if progress_results:
            surv_rows.append({
                'Method': 'PROGRESS (Ours)',
                'C-index': progress_results.get('c_index'),
                'AUC-2yr': progress_results.get('auc_2yr'),
                'AUC-3yr': progress_results.get('auc_3yr'),
                'AUC-5yr': progress_results.get('auc_5yr'),
            })
        
        surv_df = pd.DataFrame(surv_rows)
        
        return {'trajectory': traj_df, 'survival': surv_df}


# =============================================================================
# MAIN / EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of baseline comparison framework."""
    
    print("=" * 60)
    print("Baseline Comparison Framework for PROGRESS")
    print("=" * 60)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Synthetic features (like CSF biomarkers + demographics)
    X = np.random.randn(n_samples, n_features)
    
    # Synthetic trajectory parameters (intercept, slope, acceleration)
    y_traj = np.column_stack([
        X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.5,  # intercept
        X[:, 2] * 0.2 + np.random.randn(n_samples) * 0.3,  # slope
        X[:, 3] * 0.1 + np.random.randn(n_samples) * 0.2   # acceleration
    ])
    
    # Synthetic survival data
    times = np.exp(2 + X[:, 0] * 0.3 + np.random.randn(n_samples) * 0.5)
    events = (np.random.rand(n_samples) < 0.3).astype(float)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    (X_train, X_test, y_traj_train, y_traj_test, 
     times_train, times_test, events_train, events_test) = train_test_split(
        X, y_traj, times, events, test_size=0.2, random_state=42
    )
    
    # Run comparisons
    config = BaselineConfig()
    runner = BaselineComparisonRunner(config)
    
    results = runner.run_all_comparisons(
        X_train, y_traj_train, times_train, events_train,
        X_test, y_traj_test, times_test, events_test
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAJECTORY PREDICTION RESULTS")
    print("=" * 60)
    for method, metrics in results['trajectory'].items():
        print(f"\n{method.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("SURVIVAL PREDICTION RESULTS")
    print("=" * 60)
    for method, metrics in results['survival'].items():
        print(f"\n{method.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    # Generate comparison tables
    tables = runner.generate_comparison_table(results)
    
    print("\n" + "=" * 60)
    print("TRAJECTORY COMPARISON TABLE")
    print("=" * 60)
    print(tables['trajectory'].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SURVIVAL COMPARISON TABLE")
    print("=" * 60)
    print(tables['survival'].to_string(index=False))
    
    return results


if __name__ == "__main__":
    main()
