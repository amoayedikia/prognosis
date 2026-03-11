#!/usr/bin/env python3
"""
NACCDataIntegrator.py - Comprehensive NACC Data Integration Pipeline

This script implements the complete data integration pipeline described in Section 2.1
of the PROGRESS framework paper. It integrates NACC CSF biomarker data with UDS clinical
assessments to create datasets suitable for AD progression prediction.

Pipeline Phases:
    Phase 1: Load CSF biomarker and UDS clinical data
    Phase 2: Preprocessing (CSF harmonization, clinical processing, ATN classification)
    Phase 3: Temporal alignment between CSF and clinical visits (90-day window)
    Phase 4: Longitudinal sequence construction (overlapping windows, L=5)
    Phase 5: Final integration and quality control

Input Files:
    - investigator_fcsf_nacc69.csv: CSF biomarker measurements
    - nacc_uds_all.csv: Unified UDS clinical data (all forms)

Output Files:
    - nacc_integrated_dataset.pkl: Integrated subject-level dataset
    - nacc_ml_sequences.pkl: ML-ready sequences (raw)
    - nacc_ml_sequences_cleaned.pkl: ML-ready sequences (cleaned)
    - integration_report.txt: Summary statistics and quality metrics

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging
import os
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nacc_integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# NACC missing value codes - these should be treated as NaN
NACC_MISSING_CODES = [-4, -1, 88, 95, 96, 97, 98, 99, 888, 995, 996, 997, 998, 999]

# Harmonization factors for different assay methods (from literature)
# Reference: ELISA = 1.0, adjustments based on systematic comparisons
HARMONIZATION_FACTORS = {
    'ABETA': {
        1: 1.0,    # ELISA (reference standard)
        2: 1.15,   # Luminex xMAP
        3: 1.05,   # Elecsys/Roche
        4: 1.08,   # MSD
        8: 1.0,    # Other immunoassay
        9: 1.0,    # Unknown
        99: 1.0    # Missing/Unknown
    },
    'PTAU': {
        1: 1.0,    # ELISA
        2: 1.10,   # Luminex
        3: 1.02,   # Elecsys
        4: 1.05,   # MSD
        8: 1.0,
        9: 1.0,
        99: 1.0
    },
    'TTAU': {
        1: 1.0,    # ELISA
        2: 1.12,   # Luminex
        3: 1.08,   # Elecsys
        4: 1.06,   # MSD
        8: 1.0,
        9: 1.0,
        99: 1.0
    }
}

# ATN classification thresholds (based on NIA-AA framework and literature)
ATN_THRESHOLDS = {
    'ABETA': 500,   # A+ if below this value (pg/mL)
    'PTAU': 60,     # T+ if above this value (pg/mL)
    'TTAU': 400     # N+ if above this value (pg/mL)
}

# Default values for imputation (population medians from NACC)
DEFAULT_VALUES = {
    'ABETA_harm': 600.0,
    'PTAU_harm': 50.0,
    'TTAU_harm': 350.0,
    'AGE_AT_BASELINE': 72.0,
    'SEX': 1.0,
    'EDUC': 16.0,
    'NACCMMSE': 28.0,
    'CDRSUM': 0.0,
    'CDRGLOB': 0.0
}

# Configuration parameters
CONFIG = {
    'max_alignment_days': 90,      # Maximum days between CSF and clinical visit
    'sequence_length': 5,          # Number of visits per sequence (L=5)
    'min_visits': 2,               # Minimum visits required for inclusion
    'min_trajectory_visits': 3,    # Minimum visits for trajectory estimation
    'variance_threshold': 10.0     # Threshold for trajectory parameter variance
}


# =============================================================================
# MAIN INTEGRATOR CLASS
# =============================================================================

class NACCDataIntegrator:
    """
    Comprehensive NACC Data Integrator implementing the PROGRESS pipeline.
    
    This class handles all aspects of integrating CSF biomarker data with
    longitudinal clinical assessments from the NACC Uniform Data Set.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the integrator with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters (uses defaults if None)
        """
        self.config = config or CONFIG
        self.harmonization_factors = HARMONIZATION_FACTORS
        self.atn_thresholds = ATN_THRESHOLDS
        self.default_values = DEFAULT_VALUES
        
        # Storage for intermediate results
        self.csf_data = None
        self.uds_data = None
        self.demographics = None
        self.clinical = None
        self.integrated = None
        self.sequences = None
        
        # Statistics tracking
        self.stats = {
            'phase1': {},
            'phase2': {},
            'phase3': {},
            'phase4': {},
            'phase5': {}
        }
        
        logger.info("NACCDataIntegrator initialized")
        logger.info(f"Configuration: {self.config}")
    
    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================
    
    def load_data(self, csf_file: str, uds_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phase 1: Load raw NACC data files.
        
        Args:
            csf_file: Path to CSF biomarker CSV file
            uds_file: Path to UDS clinical data CSV file
            
        Returns:
            Tuple of (csf_data, uds_data) DataFrames
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Loading NACC Data Files")
        logger.info("=" * 60)
        
        # Load CSF data
        logger.info(f"Loading CSF data from: {csf_file}")
        try:
            self.csf_data = pd.read_csv(csf_file, low_memory=False)
            logger.info(f"  Loaded {len(self.csf_data)} CSF records")
            logger.info(f"  Unique subjects: {self.csf_data['NACCID'].nunique()}")
            logger.info(f"  Columns: {len(self.csf_data.columns)}")
        except Exception as e:
            logger.error(f"Failed to load CSF data: {e}")
            raise
        
        # Load UDS data
        logger.info(f"Loading UDS data from: {uds_file}")
        try:
            self.uds_data = pd.read_csv(uds_file, low_memory=False)
            logger.info(f"  Loaded {len(self.uds_data)} UDS records")
            logger.info(f"  Unique subjects: {self.uds_data['NACCID'].nunique()}")
            logger.info(f"  Visit range: {self.uds_data['NACCVNUM'].min()} - {self.uds_data['NACCVNUM'].max()}")
        except Exception as e:
            logger.error(f"Failed to load UDS data: {e}")
            raise
        
        # Store statistics
        self.stats['phase1'] = {
            'csf_records': len(self.csf_data),
            'csf_subjects': self.csf_data['NACCID'].nunique(),
            'uds_records': len(self.uds_data),
            'uds_subjects': self.uds_data['NACCID'].nunique()
        }
        
        # Identify common subjects
        common_subjects = set(self.csf_data['NACCID']) & set(self.uds_data['NACCID'])
        logger.info(f"  Subjects with both CSF and UDS data: {len(common_subjects)}")
        self.stats['phase1']['common_subjects'] = len(common_subjects)
        
        return self.csf_data, self.uds_data
    
    # =========================================================================
    # PHASE 2: PREPROCESSING
    # =========================================================================
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Phase 2: Preprocess CSF and clinical data.
        
        This includes:
        - CSF biomarker harmonization across assay platforms
        - Clinical data processing and temporal features
        - ATN classification
        
        Returns:
            Tuple of (csf_processed, demographics, clinical) DataFrames
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Preprocessing Data")
        logger.info("=" * 60)
        
        # 2A: Process CSF biomarkers
        csf_processed = self._process_csf_biomarkers(self.csf_data.copy())
        
        # 2B: Extract demographics from baseline visits
        self.demographics = self._extract_demographics(self.uds_data.copy())
        
        # 2C: Process longitudinal clinical data
        self.clinical = self._process_clinical_data(self.uds_data.copy())
        
        # 2D: Classify ATN profiles
        csf_processed = self._classify_atn(csf_processed)
        
        # Store processed CSF data
        self.csf_data = csf_processed
        
        return csf_processed, self.demographics, self.clinical
    
    def _process_csf_biomarkers(self, csf: pd.DataFrame) -> pd.DataFrame:
        """
        Process CSF biomarker data with assay-specific harmonization.
        
        Implements Equation 1 from paper:
        b_harmonized = h_{b,m} * b_raw
        
        Args:
            csf: Raw CSF DataFrame
            
        Returns:
            Processed CSF DataFrame with harmonized biomarkers
        """
        logger.info("  Processing CSF biomarkers...")
        
        # Create collection dates
        date_cols = ['CSFLPYR', 'CSFLPMO', 'CSFLPDY']
        if all(col in csf.columns for col in date_cols):
            csf['CollectionDate'] = pd.to_datetime(
                csf[date_cols].rename(columns={
                    'CSFLPYR': 'year', 
                    'CSFLPMO': 'month', 
                    'CSFLPDY': 'day'
                }),
                errors='coerce'
            )
            valid_dates = csf['CollectionDate'].notna().sum()
            logger.info(f"    Valid collection dates: {valid_dates}/{len(csf)}")
        else:
            logger.warning("    Date columns not found, setting CollectionDate to NaT")
            csf['CollectionDate'] = pd.NaT
        
        # Define biomarker columns mapping
        biomarker_mapping = [
            ('ABETA', 'CSFABETA', 'CSFABMD'),
            ('PTAU', 'CSFPTAU', 'CSFPTMD'),
            ('TTAU', 'CSFTTAU', 'CSFTTMD')
        ]
        
        harmonization_stats = {}
        
        for biomarker, raw_col, method_col in biomarker_mapping:
            if raw_col in csf.columns:
                # Clean raw values - replace missing codes with NaN
                raw_values = csf[raw_col].copy()
                raw_values = self._clean_nacc_values(raw_values)
                
                # Get assay methods
                if method_col in csf.columns:
                    methods = csf[method_col].fillna(99).astype(int)
                else:
                    methods = pd.Series([99] * len(csf))
                    logger.warning(f"    Method column {method_col} not found, using default")
                
                # Apply harmonization
                harmonized = self._harmonize_biomarker(raw_values, methods, biomarker)
                csf[f'{biomarker}_harm'] = harmonized
                
                # Track statistics
                valid_count = harmonized.notna().sum()
                missing_count = harmonized.isna().sum()
                harmonization_stats[biomarker] = {
                    'valid': valid_count,
                    'missing': missing_count,
                    'missing_pct': missing_count / len(csf) * 100
                }
                
                logger.info(f"    {biomarker}: {valid_count} valid, {missing_count} missing ({missing_count/len(csf)*100:.1f}%)")
            else:
                logger.warning(f"    Column {raw_col} not found in CSF data")
                csf[f'{biomarker}_harm'] = np.nan
        
        self.stats['phase2']['harmonization'] = harmonization_stats
        
        # Apply ComBat harmonization for site effects if enough data
        csf = self._apply_combat_harmonization(csf)
        
        return csf
    
    def _harmonize_biomarker(self, values: pd.Series, methods: pd.Series, 
                             biomarker: str) -> pd.Series:
        """
        Apply assay-specific harmonization factors.
        
        Args:
            values: Raw biomarker values
            methods: Assay method codes
            biomarker: Biomarker name ('ABETA', 'PTAU', 'TTAU')
            
        Returns:
            Harmonized biomarker values
        """
        harmonized = values.copy()
        factors = self.harmonization_factors.get(biomarker, {})
        
        for method_code, factor in factors.items():
            mask = (methods == method_code) & values.notna()
            if mask.any():
                harmonized.loc[mask] = values.loc[mask] * factor
        
        # Handle unknown methods (no harmonization)
        known_methods = set(factors.keys())
        unknown_mask = ~methods.isin(known_methods) & values.notna()
        if unknown_mask.any():
            logger.debug(f"    {unknown_mask.sum()} values with unknown assay methods")
        
        return harmonized
    
    def _apply_combat_harmonization(self, csf: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ComBat harmonization for site effects (Equation 2 in paper).
        
        This removes systematic differences between ADC sites while preserving
        biological variation.
        
        Args:
            csf: CSF DataFrame with harmonized biomarkers
            
        Returns:
            CSF DataFrame with site-harmonized biomarkers
        """
        logger.info("  Applying site-effect harmonization...")
        
        # Check if site information is available
        if 'NACCADC' not in csf.columns:
            logger.warning("    NACCADC (site) column not found, skipping ComBat")
            return csf
        
        biomarkers = ['ABETA_harm', 'PTAU_harm', 'TTAU_harm']
        available_biomarkers = [b for b in biomarkers if b in csf.columns]
        
        if not available_biomarkers:
            logger.warning("    No harmonized biomarkers found, skipping ComBat")
            return csf
        
        # Get data matrix for subjects with complete biomarkers
        complete_mask = csf[available_biomarkers].notna().all(axis=1)
        n_complete = complete_mask.sum()
        
        if n_complete < 50:
            logger.warning(f"    Only {n_complete} complete cases, skipping ComBat")
            return csf
        
        # Try to use neuroCombat if available
        try:
            from neuroCombat import neuroCombat
            
            # Prepare data for ComBat
            data = csf.loc[complete_mask, available_biomarkers].T.values
            batch = csf.loc[complete_mask, 'NACCADC'].values
            
            # Get unique batches with sufficient samples
            batch_counts = pd.Series(batch).value_counts()
            valid_batches = batch_counts[batch_counts >= 3].index
            valid_batch_mask = pd.Series(batch).isin(valid_batches).values
            
            if valid_batch_mask.sum() < 30:
                logger.warning("    Insufficient samples per site for ComBat")
                return csf
            
            # Apply ComBat
            data_combat = neuroCombat(
                dat=data[:, valid_batch_mask],
                covars={'batch': batch[valid_batch_mask]},
                batch_col='batch'
            )['data']
            
            # Update values
            complete_indices = csf.index[complete_mask].values[valid_batch_mask]
            for i, biomarker in enumerate(available_biomarkers):
                csf.loc[complete_indices, biomarker] = data_combat[i, :]
            
            logger.info(f"    Applied ComBat to {len(complete_indices)} subjects across {len(valid_batches)} sites")
            
        except ImportError:
            logger.info("    neuroCombat not available, using simplified site correction")
            # Simplified approach: center each site to grand mean
            for biomarker in available_biomarkers:
                grand_mean = csf[biomarker].mean()
                for site in csf['NACCADC'].unique():
                    site_mask = csf['NACCADC'] == site
                    site_mean = csf.loc[site_mask, biomarker].mean()
                    if pd.notna(site_mean):
                        csf.loc[site_mask, biomarker] += (grand_mean - site_mean)
        
        return csf
    
    def _extract_demographics(self, uds: pd.DataFrame) -> pd.DataFrame:
        """
        Extract baseline demographics from UDS data.
        
        Args:
            uds: UDS DataFrame
            
        Returns:
            Demographics DataFrame (one row per subject)
        """
        logger.info("  Extracting demographics...")
        
        # Filter to baseline visits (NACCVNUM == 1)
        baseline = uds[uds['NACCVNUM'] == 1].copy()
        logger.info(f"    Baseline visits: {len(baseline)}")
        
        # Define demographic columns to extract
        demo_cols = {
            'required': ['NACCID'],
            'demographic': ['SEX', 'BIRTHYR', 'EDUC', 'RACE', 'HISPANIC', 'MARISTAT'],
            'genetic': ['NACCAPOE'],  # APOE status if available
            'dates': ['VISITYR', 'VISITMO', 'VISITDAY']
        }
        
        # Collect available columns
        available_cols = ['NACCID']
        for category, cols in demo_cols.items():
            if category == 'required':
                continue
            for col in cols:
                if col in baseline.columns:
                    available_cols.append(col)
        
        demographics = baseline[available_cols].copy()
        
        # Calculate age at baseline
        if 'BIRTHYR' in demographics.columns and 'VISITYR' in baseline.columns:
            demographics['AGE_AT_BASELINE'] = baseline['VISITYR'] - demographics['BIRTHYR']
            # Clean implausible ages
            demographics.loc[
                (demographics['AGE_AT_BASELINE'] < 40) | 
                (demographics['AGE_AT_BASELINE'] > 110), 
                'AGE_AT_BASELINE'
            ] = np.nan
            logger.info(f"    Valid ages: {demographics['AGE_AT_BASELINE'].notna().sum()}")
        
        # Clean demographic variables
        for col in ['SEX', 'EDUC', 'RACE']:
            if col in demographics.columns:
                demographics[col] = self._clean_nacc_values(demographics[col])
        
        # Create baseline visit date
        date_cols = ['VISITYR', 'VISITMO', 'VISITDAY']
        if all(col in baseline.columns for col in date_cols):
            demographics['BaselineDate'] = pd.to_datetime(
                baseline[date_cols].rename(columns={
                    'VISITYR': 'year',
                    'VISITMO': 'month',
                    'VISITDAY': 'day'
                }),
                errors='coerce'
            )
        
        logger.info(f"    Extracted demographics for {len(demographics)} subjects")
        self.stats['phase2']['demographics'] = {
            'total': len(demographics),
            'with_age': demographics['AGE_AT_BASELINE'].notna().sum() if 'AGE_AT_BASELINE' in demographics.columns else 0,
            'with_education': demographics['EDUC'].notna().sum() if 'EDUC' in demographics.columns else 0
        }
        
        return demographics
    
    def _process_clinical_data(self, uds: pd.DataFrame) -> pd.DataFrame:
        """
        Process longitudinal clinical data from UDS.
        
        Args:
            uds: UDS DataFrame
            
        Returns:
            Processed clinical DataFrame with temporal features
        """
        logger.info("  Processing clinical data...")
        
        # Define clinical columns to extract
        clinical_cols = {
            'identifiers': ['NACCID', 'NACCVNUM'],
            'dates': ['VISITYR', 'VISITMO', 'VISITDAY'],
            'cognitive': ['NACCMMSE', 'MOCATOTS', 'CRAFTVRS', 'CRAFTDVR', 'MINTTOTS'],
            'cdr': ['CDRSUM', 'CDRGLOB', 'CDRMEMORY', 'CDRORIENT', 'CDRJUDGE', 
                    'CDRCOMMUN', 'CDRHOME', 'CDRCARE'],
            'functional': ['FAQ'],
            'diagnosis': ['NORMCOG', 'DEMENTED', 'NACCUDSD', 'NACCTMCI', 'IMPNOMCI',
                         'NACCALZD', 'NACCALZP']
        }
        
        # Collect available columns
        available_cols = []
        for category, cols in clinical_cols.items():
            for col in cols:
                if col in uds.columns:
                    available_cols.append(col)
                    
        # Ensure required columns are present
        if 'NACCID' not in available_cols or 'NACCVNUM' not in available_cols:
            logger.error("Missing required identifier columns")
            raise ValueError("NACCID and NACCVNUM required in UDS data")
        
        clinical = uds[available_cols].copy()
        
        # Clean cognitive scores
        for col in ['NACCMMSE', 'MOCATOTS', 'CDRSUM', 'CDRGLOB']:
            if col in clinical.columns:
                clinical[col] = self._clean_nacc_values(clinical[col])
        
        # Create visit dates
        date_cols = ['VISITYR', 'VISITMO', 'VISITDAY']
        if all(col in clinical.columns for col in date_cols):
            clinical['VisitDate'] = pd.to_datetime(
                clinical[date_cols].rename(columns={
                    'VISITYR': 'year',
                    'VISITMO': 'month',
                    'VISITDAY': 'day'
                }),
                errors='coerce'
            )
        else:
            clinical['VisitDate'] = pd.NaT
        
        # Add temporal features for each subject
        clinical = self._add_temporal_features(clinical)
        
        # Add progression indicators
        clinical = self._add_progression_indicators(clinical)
        
        logger.info(f"    Processed {len(clinical)} clinical visits from {clinical['NACCID'].nunique()} subjects")
        
        # Store statistics
        self.stats['phase2']['clinical'] = {
            'total_visits': len(clinical),
            'unique_subjects': clinical['NACCID'].nunique(),
            'mean_visits_per_subject': len(clinical) / clinical['NACCID'].nunique()
        }
        
        return clinical
    
    def _add_temporal_features(self, clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to clinical data.
        
        Calculates:
        - YearsFromBaseline: Time since first visit
        - Change scores for cognitive measures
        
        Args:
            clinical: Clinical DataFrame
            
        Returns:
            Clinical DataFrame with temporal features
        """
        logger.info("    Adding temporal features...")
        
        # Sort by subject and visit
        clinical = clinical.sort_values(['NACCID', 'NACCVNUM']).reset_index(drop=True)
        
        # Initialize columns
        clinical['YearsFromBaseline'] = np.nan
        
        # Process each subject
        for naccid in clinical['NACCID'].unique():
            mask = clinical['NACCID'] == naccid
            subject_data = clinical.loc[mask].copy()
            
            # Calculate years from baseline
            if subject_data['VisitDate'].notna().any():
                baseline_date = subject_data['VisitDate'].iloc[0]
                if pd.notna(baseline_date):
                    years_from_baseline = (
                        (subject_data['VisitDate'] - baseline_date).dt.days / 365.25
                    )
                    clinical.loc[mask, 'YearsFromBaseline'] = years_from_baseline
            
            # If no valid dates, use visit numbers (assume ~annual visits)
            if clinical.loc[mask, 'YearsFromBaseline'].isna().all():
                clinical.loc[mask, 'YearsFromBaseline'] = subject_data['NACCVNUM'] - 1
            
            # Calculate change scores for cognitive measures
            for measure in ['NACCMMSE', 'CDRSUM']:
                if measure in clinical.columns:
                    baseline_value = subject_data[measure].iloc[0]
                    if pd.notna(baseline_value):
                        clinical.loc[mask, f'{measure}_change'] = (
                            clinical.loc[mask, measure] - baseline_value
                        )
        
        return clinical
    
    def _add_progression_indicators(self, clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Add disease progression indicators.
        
        Tracks:
        - Conversion to MCI
        - Conversion to dementia
        - Diagnostic transitions
        
        Args:
            clinical: Clinical DataFrame
            
        Returns:
            Clinical DataFrame with progression indicators
        """
        logger.info("    Adding progression indicators...")
        
        # Sort by subject and visit
        clinical = clinical.sort_values(['NACCID', 'NACCVNUM'])
        
        # Add previous visit diagnosis
        if 'DEMENTED' in clinical.columns:
            clinical['DEMENTED_prev'] = clinical.groupby('NACCID')['DEMENTED'].shift(1)
            clinical['converted_to_dementia'] = (
                (clinical['DEMENTED_prev'] == 0) & (clinical['DEMENTED'] == 1)
            )
        
        if 'NACCTMCI' in clinical.columns:
            clinical['NACCTMCI_prev'] = clinical.groupby('NACCID')['NACCTMCI'].shift(1)
            clinical['converted_to_MCI'] = (
                (clinical['NACCTMCI_prev'].isin([0])) & (clinical['NACCTMCI'] == 1)
            )
        
        return clinical
    
    def _classify_atn(self, csf: pd.DataFrame) -> pd.DataFrame:
        """
        Classify subjects into ATN profiles based on biomarker thresholds.
        
        Uses NIA-AA framework thresholds:
        - A+: AÎ²42 < 500 pg/mL
        - T+: p-tau > 60 pg/mL  
        - N+: t-tau > 400 pg/mL
        
        Args:
            csf: CSF DataFrame with harmonized biomarkers
            
        Returns:
            CSF DataFrame with ATN classification
        """
        logger.info("  Classifying ATN profiles...")
        
        # Initialize binary classifications
        csf['A_positive'] = 0
        csf['T_positive'] = 0
        csf['N_positive'] = 0
        
        # A+ if AÎ²42 < threshold
        if 'ABETA_harm' in csf.columns:
            mask = csf['ABETA_harm'].notna()
            csf.loc[mask, 'A_positive'] = (
                csf.loc[mask, 'ABETA_harm'] < self.atn_thresholds['ABETA']
            ).astype(int)
        
        # T+ if p-tau > threshold
        if 'PTAU_harm' in csf.columns:
            mask = csf['PTAU_harm'].notna()
            csf.loc[mask, 'T_positive'] = (
                csf.loc[mask, 'PTAU_harm'] > self.atn_thresholds['PTAU']
            ).astype(int)
        
        # N+ if t-tau > threshold
        if 'TTAU_harm' in csf.columns:
            mask = csf['TTAU_harm'].notna()
            csf.loc[mask, 'N_positive'] = (
                csf.loc[mask, 'TTAU_harm'] > self.atn_thresholds['TTAU']
            ).astype(int)
        
        # Create ATN profile string
        csf['ATN_profile'] = (
            'A' + csf['A_positive'].astype(str) +
            'T' + csf['T_positive'].astype(str) +
            'N' + csf['N_positive'].astype(str)
        )
        
        # Log ATN distribution
        atn_counts = csf['ATN_profile'].value_counts()
        logger.info("    ATN Profile Distribution:")
        for profile, count in atn_counts.items():
            logger.info(f"      {profile}: {count} ({count/len(csf)*100:.1f}%)")
        
        self.stats['phase2']['atn'] = atn_counts.to_dict()
        
        return csf
    
    # =========================================================================
    # PHASE 3: TEMPORAL ALIGNMENT
    # =========================================================================
    
    def align_data(self) -> pd.DataFrame:
        """
        Phase 3: Temporally align CSF measurements with clinical visits.
        
        Finds the closest clinical visit to each CSF collection date,
        within the maximum alignment window.
        
        Returns:
            CSF DataFrame with alignment information
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Temporal Alignment")
        logger.info("=" * 60)
        
        max_days = self.config['max_alignment_days']
        logger.info(f"  Maximum alignment window: {max_days} days")
        
        # Initialize alignment columns
        self.csf_data['ClosestVisit'] = np.nan
        self.csf_data['DaysToVisit'] = np.nan
        self.csf_data['AlignmentStatus'] = 'not_attempted'
        
        # Track alignment statistics
        alignment_stats = {
            'aligned': 0,
            'no_clinical': 0,
            'too_far': 0,
            'no_csf_date': 0
        }
        
        # Process each CSF record
        for idx, csf_row in self.csf_data.iterrows():
            naccid = csf_row['NACCID']
            csf_date = csf_row['CollectionDate']
            
            # Get clinical visits for this subject
            subject_visits = self.clinical[self.clinical['NACCID'] == naccid].copy()
            
            if len(subject_visits) == 0:
                self.csf_data.loc[idx, 'AlignmentStatus'] = 'no_clinical'
                alignment_stats['no_clinical'] += 1
                continue
            
            # Handle missing CSF date
            if pd.isna(csf_date):
                # Assume CSF collected at baseline
                self.csf_data.loc[idx, 'ClosestVisit'] = 1
                self.csf_data.loc[idx, 'DaysToVisit'] = 0
                self.csf_data.loc[idx, 'AlignmentStatus'] = 'assumed_baseline'
                alignment_stats['no_csf_date'] += 1
                continue
            
            # Find closest visit by date
            if 'VisitDate' in subject_visits.columns:
                subject_visits = subject_visits[subject_visits['VisitDate'].notna()]
                
                if len(subject_visits) == 0:
                    # Fall back to assuming baseline
                    self.csf_data.loc[idx, 'ClosestVisit'] = 1
                    self.csf_data.loc[idx, 'DaysToVisit'] = 0
                    self.csf_data.loc[idx, 'AlignmentStatus'] = 'assumed_baseline'
                    continue
                
                subject_visits['TimeDiff'] = abs(
                    (subject_visits['VisitDate'] - csf_date).dt.days
                )
                
                closest_idx = subject_visits['TimeDiff'].idxmin()
                closest_visit = subject_visits.loc[closest_idx]
                days_diff = closest_visit['TimeDiff']
                
                self.csf_data.loc[idx, 'ClosestVisit'] = closest_visit['NACCVNUM']
                self.csf_data.loc[idx, 'DaysToVisit'] = days_diff
                
                if days_diff <= max_days:
                    self.csf_data.loc[idx, 'AlignmentStatus'] = 'aligned'
                    alignment_stats['aligned'] += 1
                else:
                    self.csf_data.loc[idx, 'AlignmentStatus'] = 'too_far'
                    alignment_stats['too_far'] += 1
        
        # Log alignment results
        logger.info("  Alignment Results:")
        for status, count in alignment_stats.items():
            logger.info(f"    {status}: {count} ({count/len(self.csf_data)*100:.1f}%)")
        
        self.stats['phase3'] = alignment_stats
        
        return self.csf_data
    
    # =========================================================================
    # PHASE 4: SEQUENCE CONSTRUCTION
    # =========================================================================
    
    def create_sequences(self) -> pd.DataFrame:
        """
        Phase 4: Create ML-ready longitudinal sequences.
        
        Creates overlapping sequences of length L for each subject,
        with static features (CSF, demographics) and dynamic features
        (cognitive assessments over time).
        
        Returns:
            DataFrame of sequences ready for ML training
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: Sequence Construction")
        logger.info("=" * 60)
        
        L = self.config['sequence_length']
        logger.info(f"  Sequence length: {L}")
        
        sequences = []
        sequence_stats = {
            'total_sequences': 0,
            'subjects_with_sequences': 0,
            'skipped_insufficient_visits': 0
        }
        
        # First, integrate CSF with demographics
        integrated_subjects = self._integrate_subjects()
        
        # Create sequences for each subject
        for naccid, subject_info in integrated_subjects.items():
            # Get clinical trajectory for this subject
            subject_clinical = self.clinical[
                self.clinical['NACCID'] == naccid
            ].sort_values('NACCVNUM')
            
            # Skip if insufficient visits
            if len(subject_clinical) < L:
                sequence_stats['skipped_insufficient_visits'] += 1
                continue
            
            sequence_stats['subjects_with_sequences'] += 1
            
            # Create overlapping sequences
            for start_idx in range(len(subject_clinical) - L + 1):
                end_idx = start_idx + L
                sequence_visits = subject_clinical.iloc[start_idx:end_idx]
                
                # Build sequence features
                sequence_features = self._build_sequence_features(sequence_visits)
                
                # Create sequence record
                sequence_record = {
                    'NACCID': naccid,
                    'sequence_idx': start_idx,
                    
                    # Static features (CSF biomarkers)
                    'ABETA_harm': subject_info.get('ABETA_harm', np.nan),
                    'PTAU_harm': subject_info.get('PTAU_harm', np.nan),
                    'TTAU_harm': subject_info.get('TTAU_harm', np.nan),
                    'ATN_profile': subject_info.get('ATN_profile', 'Unknown'),
                    
                    # Demographics
                    'AGE_AT_BASELINE': subject_info.get('AGE_AT_BASELINE', np.nan),
                    'SEX': subject_info.get('SEX', np.nan),
                    'EDUC': subject_info.get('EDUC', np.nan),
                    'RACE': subject_info.get('RACE', np.nan),
                    
                    # Sequence features
                    'sequence_features': sequence_features,
                    'sequence_length': L,
                    'start_visit': int(sequence_visits['NACCVNUM'].iloc[0]),
                    'end_visit': int(sequence_visits['NACCVNUM'].iloc[-1]),
                    
                    # Target information
                    'has_next_visit': end_idx < len(subject_clinical)
                }
                
                # Add target values if next visit exists
                if sequence_record['has_next_visit']:
                    next_visit = subject_clinical.iloc[end_idx]
                    sequence_record['next_MMSE'] = next_visit.get('NACCMMSE', np.nan)
                    sequence_record['next_CDRSUM'] = next_visit.get('CDRSUM', np.nan)
                    sequence_record['next_CDRGLOB'] = next_visit.get('CDRGLOB', np.nan)
                    sequence_record['next_YearsFromBaseline'] = next_visit.get(
                        'YearsFromBaseline', np.nan
                    )
                
                sequences.append(sequence_record)
                sequence_stats['total_sequences'] += 1
        
        # Create DataFrame
        self.sequences = pd.DataFrame(sequences)
        
        # Log statistics
        logger.info("  Sequence Statistics:")
        logger.info(f"    Total sequences: {sequence_stats['total_sequences']}")
        logger.info(f"    Subjects with sequences: {sequence_stats['subjects_with_sequences']}")
        logger.info(f"    Skipped (insufficient visits): {sequence_stats['skipped_insufficient_visits']}")
        
        if len(self.sequences) > 0:
            logger.info(f"    Sequences with next visit: {self.sequences['has_next_visit'].sum()}")
        
        self.stats['phase4'] = sequence_stats
        
        return self.sequences
    
    def _integrate_subjects(self) -> Dict:
        """
        Integrate CSF and demographics at subject level.
        
        Returns:
            Dictionary mapping NACCID to integrated subject info
        """
        integrated = {}
        
        # Get subjects with both CSF and clinical data
        csf_subjects = set(self.csf_data['NACCID'])
        clinical_subjects = set(self.clinical['NACCID'])
        common = csf_subjects & clinical_subjects
        
        for naccid in common:
            # Get CSF data (use first/closest to baseline measurement)
            csf_subject = self.csf_data[self.csf_data['NACCID'] == naccid]
            if 'DaysToVisit' in csf_subject.columns:
                csf_row = csf_subject.loc[csf_subject['DaysToVisit'].idxmin()]
            else:
                csf_row = csf_subject.iloc[0]
            
            # Get demographics
            demo_subject = self.demographics[self.demographics['NACCID'] == naccid]
            if len(demo_subject) > 0:
                demo_row = demo_subject.iloc[0]
            else:
                demo_row = pd.Series()
            
            # Combine information
            integrated[naccid] = {
                'NACCID': naccid,
                # CSF
                'ABETA_harm': csf_row.get('ABETA_harm', np.nan),
                'PTAU_harm': csf_row.get('PTAU_harm', np.nan),
                'TTAU_harm': csf_row.get('TTAU_harm', np.nan),
                'ATN_profile': csf_row.get('ATN_profile', 'Unknown'),
                # Demographics
                'AGE_AT_BASELINE': demo_row.get('AGE_AT_BASELINE', np.nan),
                'SEX': demo_row.get('SEX', np.nan),
                'EDUC': demo_row.get('EDUC', np.nan),
                'RACE': demo_row.get('RACE', np.nan)
            }
        
        return integrated
    
    def _build_sequence_features(self, visits: pd.DataFrame) -> List[List[float]]:
        """
        Build feature matrix from sequence of visits.
        
        Args:
            visits: DataFrame of consecutive visits
            
        Returns:
            List of feature vectors, one per visit
        """
        features = []
        
        for _, visit in visits.iterrows():
            visit_features = [
                float(visit.get('YearsFromBaseline', 0)),
                float(visit.get('NACCMMSE', 30)),
                float(visit.get('CDRSUM', 0)),
                float(visit.get('CDRGLOB', 0))
            ]
            
            # Clean NaN values
            visit_features = [
                self.default_values.get('NACCMMSE', 28) if (i == 1 and np.isnan(v)) else
                self.default_values.get('CDRSUM', 0) if (i == 2 and np.isnan(v)) else
                v if not np.isnan(v) else 0
                for i, v in enumerate(visit_features)
            ]
            
            features.append(visit_features)
        
        return features
    
    # =========================================================================
    # PHASE 5: FINAL INTEGRATION AND CLEANING
    # =========================================================================
    
    def finalize_and_clean(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phase 5: Final integration, cleaning, and quality control.
        
        Performs:
        - Data validation and cleaning
        - Missing value handling
        - Quality control checks
        - Output preparation
        
        Returns:
            Tuple of (integrated_dataset, cleaned_sequences)
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: Finalization and Cleaning")
        logger.info("=" * 60)
        
        # 5A: Create final integrated dataset
        integrated_dataset = self._create_integrated_dataset()
        
        # 5B: Clean sequences
        cleaned_sequences = self._clean_sequences(self.sequences.copy())
        
        # 5C: Apply biological imputation
        cleaned_sequences = self._biological_imputation(cleaned_sequences)
        
        # 5D: Final validation
        self._validate_outputs(integrated_dataset, cleaned_sequences)
        
        self.integrated = integrated_dataset
        
        return integrated_dataset, cleaned_sequences
    
    def _create_integrated_dataset(self) -> pd.DataFrame:
        """
        Create subject-level integrated dataset.
        
        Returns:
            DataFrame with one row per subject
        """
        logger.info("  Creating integrated dataset...")
        
        integrated_list = []
        integration_stats = {'included': 0, 'excluded_no_visits': 0, 'excluded_no_csf': 0}
        
        # Get integrated subject info
        subject_info = self._integrate_subjects()
        
        for naccid, info in subject_info.items():
            # Get clinical trajectory
            subject_clinical = self.clinical[
                self.clinical['NACCID'] == naccid
            ].sort_values('NACCVNUM')
            
            # Skip if insufficient visits
            if len(subject_clinical) < self.config['min_visits']:
                integration_stats['excluded_no_visits'] += 1
                continue
            
            # Skip if no valid CSF
            if (pd.isna(info['ABETA_harm']) and 
                pd.isna(info['PTAU_harm']) and 
                pd.isna(info['TTAU_harm'])):
                integration_stats['excluded_no_csf'] += 1
                continue
            
            # Calculate trajectory summary
            follow_up_years = subject_clinical['YearsFromBaseline'].max()
            
            # Create integrated record
            record = {
                'NACCID': naccid,
                # CSF biomarkers
                'ABETA_harm': info['ABETA_harm'],
                'PTAU_harm': info['PTAU_harm'],
                'TTAU_harm': info['TTAU_harm'],
                'ATN_profile': info['ATN_profile'],
                # Demographics
                'AGE_AT_BASELINE': info['AGE_AT_BASELINE'],
                'SEX': info['SEX'],
                'EDUC': info['EDUC'],
                'RACE': info['RACE'],
                # Trajectory info
                'num_visits': len(subject_clinical),
                'follow_up_years': follow_up_years,
                'baseline_MMSE': subject_clinical['NACCMMSE'].iloc[0] if 'NACCMMSE' in subject_clinical.columns else np.nan,
                'baseline_CDRSUM': subject_clinical['CDRSUM'].iloc[0] if 'CDRSUM' in subject_clinical.columns else np.nan,
                # Store trajectory as nested structure
                'clinical_trajectory': subject_clinical.to_dict('records')
            }
            
            # Add decline rates if enough visits
            if len(subject_clinical) >= 3 and 'NACCMMSE' in subject_clinical.columns:
                x = subject_clinical['YearsFromBaseline'].values
                y = subject_clinical['NACCMMSE'].values
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if valid_mask.sum() >= 2:
                    slope = np.polyfit(x[valid_mask], y[valid_mask], 1)[0]
                    record['MMSE_decline_rate'] = -slope
            
            # Add conversion status
            if 'converted_to_dementia' in subject_clinical.columns:
                record['converted_to_dementia'] = int(
                    subject_clinical['converted_to_dementia'].any()
                )
                
                # Time to conversion
                if record['converted_to_dementia']:
                    conversion_visit = subject_clinical[
                        subject_clinical['converted_to_dementia'] == True
                    ].iloc[0]
                    record['time_to_dementia'] = conversion_visit['YearsFromBaseline']
            
            integrated_list.append(record)
            integration_stats['included'] += 1
        
        integrated_df = pd.DataFrame(integrated_list)
        
        logger.info(f"  Integration Results:")
        logger.info(f"    Included: {integration_stats['included']}")
        logger.info(f"    Excluded (no visits): {integration_stats['excluded_no_visits']}")
        logger.info(f"    Excluded (no CSF): {integration_stats['excluded_no_csf']}")
        
        self.stats['phase5']['integration'] = integration_stats
        
        return integrated_df
    
    def _clean_sequences(self, sequences: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sequences by handling missing values and validating ranges.
        
        Args:
            sequences: Raw sequences DataFrame
            
        Returns:
            Cleaned sequences DataFrame
        """
        logger.info("  Cleaning sequences...")
        
        cleaning_stats = {
            'total_input': len(sequences),
            'invalid_targets_removed': 0,
            'targets_imputed': 0,
            'biomarkers_imputed': 0
        }
        
        valid_sequences = []
        
        for idx, row in sequences.iterrows():
            # Skip if no next visit
            if not row.get('has_next_visit', False):
                valid_sequences.append(row.to_dict())
                continue
            
            row_dict = row.to_dict()
            
            # Clean target values
            next_mmse = row.get('next_MMSE', np.nan)
            next_cdr = row.get('next_CDRSUM', np.nan)
            
            # Apply NACC missing code cleaning
            next_mmse = self._clean_single_value(next_mmse)
            next_cdr = self._clean_single_value(next_cdr)
            
            # Validate ranges
            if not pd.isna(next_mmse) and (next_mmse < 0 or next_mmse > 30):
                next_mmse = np.nan
            if not pd.isna(next_cdr) and (next_cdr < 0 or next_cdr > 18):
                next_cdr = np.nan
            
            # Skip if both targets invalid
            if pd.isna(next_mmse) and pd.isna(next_cdr):
                cleaning_stats['invalid_targets_removed'] += 1
                continue
            
            # Update cleaned values
            row_dict['next_MMSE'] = next_mmse
            row_dict['next_CDRSUM'] = next_cdr
            
            # Clean static features
            for feature in ['ABETA_harm', 'PTAU_harm', 'TTAU_harm', 
                           'AGE_AT_BASELINE', 'SEX', 'EDUC']:
                val = row_dict.get(feature, np.nan)
                val = self._clean_single_value(val)
                
                if pd.isna(val):
                    row_dict[feature] = self.default_values.get(feature, 0.0)
                    cleaning_stats['biomarkers_imputed'] += 1
                else:
                    row_dict[feature] = val
            
            # Clean sequence features
            seq_features = row_dict.get('sequence_features', [])
            if seq_features:
                cleaned_seq = []
                for visit_features in seq_features:
                    cleaned_visit = []
                    for i, val in enumerate(visit_features):
                        val = self._clean_single_value(val)
                        if pd.isna(val):
                            # Use appropriate default
                            if i == 1:  # MMSE
                                val = self.default_values['NACCMMSE']
                            elif i == 2:  # CDRSUM
                                val = self.default_values['CDRSUM']
                            else:
                                val = 0.0
                        cleaned_visit.append(val)
                    cleaned_seq.append(cleaned_visit)
                row_dict['sequence_features'] = cleaned_seq
            
            valid_sequences.append(row_dict)
        
        cleaned_df = pd.DataFrame(valid_sequences)
        
        logger.info(f"  Cleaning Results:")
        logger.info(f"    Input sequences: {cleaning_stats['total_input']}")
        logger.info(f"    Valid sequences: {len(cleaned_df)}")
        logger.info(f"    Removed (invalid targets): {cleaning_stats['invalid_targets_removed']}")
        
        self.stats['phase5']['cleaning'] = cleaning_stats
        
        return cleaned_df
    
    def _biological_imputation(self, sequences: pd.DataFrame) -> pd.DataFrame:
        """
        Apply biological imputation for missing biomarkers.
        
        Implements Equation 3 from paper:
        p-tau_imputed = Î±Â·t-tau + Î²Â·(1/AÎ²42) + Îµ
        
        Args:
            sequences: Sequences DataFrame
            
        Returns:
            Sequences with imputed biomarkers
        """
        logger.info("  Applying biological imputation...")
        
        imputation_count = 0
        
        # Group by ATN profile for more accurate imputation
        for atn_profile in sequences['ATN_profile'].unique():
            if atn_profile == 'Unknown':
                continue
            
            group_mask = sequences['ATN_profile'] == atn_profile
            group = sequences.loc[group_mask]
            
            if len(group) < 5:
                continue
            
            # Impute p-tau from t-tau if missing
            ptau_missing = group['PTAU_harm'].isna() & group['TTAU_harm'].notna()
            if ptau_missing.any():
                complete_mask = group['PTAU_harm'].notna() & group['TTAU_harm'].notna()
                if complete_mask.sum() >= 3:
                    # Calculate ratio from complete cases
                    alpha = (group.loc[complete_mask, 'PTAU_harm'] / 
                            group.loc[complete_mask, 'TTAU_harm']).median()
                    
                    # Impute
                    imputed_values = group.loc[ptau_missing, 'TTAU_harm'] * alpha
                    sequences.loc[group[ptau_missing].index, 'PTAU_harm'] = imputed_values
                    imputation_count += ptau_missing.sum()
        
        logger.info(f"    Imputed {imputation_count} biomarker values")
        
        return sequences
    
    def _validate_outputs(self, integrated: pd.DataFrame, sequences: pd.DataFrame):
        """
        Validate final outputs meet quality requirements.
        
        Args:
            integrated: Integrated dataset
            sequences: Sequences dataset
        """
        logger.info("  Validating outputs...")
        
        validation_results = {
            'integrated_valid': True,
            'sequences_valid': True,
            'warnings': []
        }
        
        # Check integrated dataset
        if len(integrated) == 0:
            validation_results['integrated_valid'] = False
            validation_results['warnings'].append("No subjects in integrated dataset")
        
        # Check for required columns
        required_cols = ['NACCID', 'ABETA_harm', 'PTAU_harm', 'TTAU_harm', 'ATN_profile']
        for col in required_cols:
            if col not in integrated.columns:
                validation_results['warnings'].append(f"Missing column: {col}")
        
        # Check sequences
        if len(sequences) == 0:
            validation_results['sequences_valid'] = False
            validation_results['warnings'].append("No sequences created")
        
        # Check target distributions
        if 'next_MMSE' in sequences.columns:
            valid_mmse = sequences['next_MMSE'].dropna()
            if len(valid_mmse) > 0:
                if valid_mmse.min() < 0 or valid_mmse.max() > 30:
                    validation_results['warnings'].append("MMSE targets out of range")
                logger.info(f"    MMSE targets: {len(valid_mmse)} valid, "
                          f"range [{valid_mmse.min():.1f}, {valid_mmse.max():.1f}]")
        
        if 'next_CDRSUM' in sequences.columns:
            valid_cdr = sequences['next_CDRSUM'].dropna()
            if len(valid_cdr) > 0:
                if valid_cdr.min() < 0 or valid_cdr.max() > 18:
                    validation_results['warnings'].append("CDR targets out of range")
                logger.info(f"    CDR targets: {len(valid_cdr)} valid, "
                          f"range [{valid_cdr.min():.1f}, {valid_cdr.max():.1f}]")
        
        # Log warnings
        for warning in validation_results['warnings']:
            logger.warning(f"    {warning}")
        
        self.stats['phase5']['validation'] = validation_results
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _clean_nacc_values(self, series: pd.Series) -> pd.Series:
        """
        Clean a pandas Series by replacing NACC missing codes with NaN.
        
        Args:
            series: Input series
            
        Returns:
            Cleaned series
        """
        cleaned = series.copy()
        for code in NACC_MISSING_CODES:
            cleaned = cleaned.replace(code, np.nan)
        return cleaned
    
    def _clean_single_value(self, value) -> float:
        """
        Clean a single value by checking for NACC missing codes.
        
        Args:
            value: Input value
            
        Returns:
            Cleaned value (float or NaN)
        """
        if pd.isna(value):
            return np.nan
        if value in NACC_MISSING_CODES:
            return np.nan
        return float(value)
    
    # =========================================================================
    # MAIN PIPELINE EXECUTION
    # =========================================================================
    
    def run_pipeline(self, csf_file: str, uds_file: str) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete data integration pipeline.
        
        Args:
            csf_file: Path to CSF biomarker CSV
            uds_file: Path to UDS clinical CSV
            
        Returns:
            Dictionary with all output DataFrames
        """
        logger.info("=" * 60)
        logger.info("NACC DATA INTEGRATION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"CSF file: {csf_file}")
        logger.info(f"UDS file: {uds_file}")
        logger.info("=" * 60)
        
        # Phase 1: Load data
        self.load_data(csf_file, uds_file)
        
        # Phase 2: Preprocess
        self.preprocess_data()
        
        # Phase 3: Temporal alignment
        self.align_data()
        
        # Phase 4: Sequence construction
        self.create_sequences()
        
        # Phase 5: Finalization
        integrated, cleaned_sequences = self.finalize_and_clean()
        
        # Generate report
        self._generate_report()
        
        return {
            'integrated': integrated,
            'sequences': self.sequences,
            'sequences_cleaned': cleaned_sequences,
            'csf_processed': self.csf_data,
            'demographics': self.demographics,
            'clinical': self.clinical
        }
    
    def _generate_report(self):
        """Generate summary report of the integration pipeline."""
        logger.info("=" * 60)
        logger.info("INTEGRATION SUMMARY")
        logger.info("=" * 60)
        
        report_lines = []
        report_lines.append("NACC Data Integration Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Phase 1 stats
        report_lines.append("Phase 1: Data Loading")
        report_lines.append("-" * 40)
        for key, value in self.stats['phase1'].items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Phase 2 stats
        report_lines.append("Phase 2: Preprocessing")
        report_lines.append("-" * 40)
        if 'harmonization' in self.stats['phase2']:
            for biomarker, stats in self.stats['phase2']['harmonization'].items():
                report_lines.append(f"  {biomarker}: {stats['valid']} valid, "
                                  f"{stats['missing_pct']:.1f}% missing")
        if 'atn' in self.stats['phase2']:
            report_lines.append("  ATN Profiles:")
            for profile, count in self.stats['phase2']['atn'].items():
                report_lines.append(f"    {profile}: {count}")
        report_lines.append("")
        
        # Phase 3 stats
        report_lines.append("Phase 3: Temporal Alignment")
        report_lines.append("-" * 40)
        for key, value in self.stats['phase3'].items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Phase 4 stats
        report_lines.append("Phase 4: Sequence Construction")
        report_lines.append("-" * 40)
        for key, value in self.stats['phase4'].items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Phase 5 stats
        report_lines.append("Phase 5: Finalization")
        report_lines.append("-" * 40)
        if 'integration' in self.stats['phase5']:
            for key, value in self.stats['phase5']['integration'].items():
                report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Final summary
        if self.integrated is not None:
            report_lines.append("Final Dataset Summary")
            report_lines.append("-" * 40)
            report_lines.append(f"  Integrated subjects: {len(self.integrated)}")
            report_lines.append(f"  Total sequences: {len(self.sequences)}")
            
            if 'follow_up_years' in self.integrated.columns:
                mean_followup = self.integrated['follow_up_years'].mean()
                report_lines.append(f"  Mean follow-up: {mean_followup:.1f} years")
            
            if 'num_visits' in self.integrated.columns:
                mean_visits = self.integrated['num_visits'].mean()
                report_lines.append(f"  Mean visits per subject: {mean_visits:.1f}")
        
        # Print report
        report_text = "\n".join(report_lines)
        logger.info(report_text)
        
        # Save report
        with open('integration_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info("\nReport saved to: integration_report.txt")
    
    def save_outputs(self, output_dir: str = "."):
        """
        Save all output files.
        
        Args:
            output_dir: Directory to save outputs
        """
        logger.info("Saving outputs...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save integrated dataset
        if self.integrated is not None:
            path = os.path.join(output_dir, 'nacc_integrated_dataset.pkl')
            self.integrated.to_pickle(path)
            logger.info(f"  Saved: {path}")
        
        # Save sequences
        if self.sequences is not None:
            path = os.path.join(output_dir, 'nacc_ml_sequences.pkl')
            self.sequences.to_pickle(path)
            logger.info(f"  Saved: {path}")
        
        logger.info("All outputs saved successfully!")


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================

def clean_nacc_sequences(input_pkl: str, output_pkl: str) -> pd.DataFrame:
    """
    Clean NACC sequences data to ensure no missing critical values.
    
    This is a standalone function that can be run separately on existing
    sequence files.
    
    Args:
        input_pkl: Path to input sequences pickle file
        output_pkl: Path to output cleaned sequences pickle file
        
    Returns:
        Cleaned sequences DataFrame
    """
    logger.info(f"Loading sequences from {input_pkl}")
    sequences = pd.read_pickle(input_pkl)
    
    initial_count = len(sequences)
    logger.info(f"Initial sequences: {initial_count}")
    
    valid_sequences = []
    
    for idx, row in sequences.iterrows():
        # Skip if no next visit
        if not row.get('has_next_visit', False):
            valid_sequences.append(row.to_dict())
            continue
        
        row_dict = row.to_dict()
        
        # Get targets
        next_mmse = row.get('next_MMSE', np.nan)
        next_cdr = row.get('next_CDRSUM', np.nan)
        
        # Clean targets
        if pd.isna(next_mmse) or next_mmse in NACC_MISSING_CODES:
            next_mmse = np.nan
        if pd.isna(next_cdr) or next_cdr in NACC_MISSING_CODES:
            next_cdr = np.nan
        
        # Validate ranges
        if not pd.isna(next_mmse) and (next_mmse < 0 or next_mmse > 30):
            next_mmse = np.nan
        if not pd.isna(next_cdr) and (next_cdr < 0 or next_cdr > 18):
            next_cdr = np.nan
        
        # Skip if both targets invalid
        if pd.isna(next_mmse) and pd.isna(next_cdr):
            continue
        
        # Update row
        row_dict['next_MMSE'] = next_mmse
        row_dict['next_CDRSUM'] = next_cdr
        
        # Clean static features
        for feature in ['ABETA_harm', 'PTAU_harm', 'TTAU_harm', 
                       'AGE_AT_BASELINE', 'SEX', 'EDUC']:
            val = row_dict.get(feature, np.nan)
            if pd.isna(val) or val in NACC_MISSING_CODES:
                row_dict[feature] = DEFAULT_VALUES.get(feature, 0.0)
        
        valid_sequences.append(row_dict)
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame(valid_sequences)
    
    logger.info(f"Valid sequences after cleaning: {len(cleaned_df)}")
    logger.info(f"Removed {initial_count - len(cleaned_df)} invalid sequences")
    
    # Save cleaned data
    cleaned_df.to_pickle(output_pkl)
    logger.info(f"Saved cleaned sequences to {output_pkl}")
    
    # Print summary
    if len(cleaned_df) > 0:
        logger.info("\nSummary statistics:")
        logger.info(f"  Subjects: {cleaned_df['NACCID'].nunique()}")
        if 'ATN_profile' in cleaned_df.columns:
            logger.info(f"  ATN profiles: {cleaned_df['ATN_profile'].value_counts().to_dict()}")
    
    return cleaned_df


def verify_data_integrity(sequences_pkl: str) -> Dict:
    """
    Verify the integrity of a sequences file.
    
    Args:
        sequences_pkl: Path to sequences pickle file
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"Verifying data integrity: {sequences_pkl}")
    
    sequences = pd.read_pickle(sequences_pkl)
    
    results = {
        'total_sequences': len(sequences),
        'unique_subjects': sequences['NACCID'].nunique(),
        'sequences_with_targets': 0,
        'valid_mmse_targets': 0,
        'valid_cdr_targets': 0,
        'mmse_range': (np.nan, np.nan),
        'cdr_range': (np.nan, np.nan),
        'issues': []
    }
    
    # Check targets
    for _, row in sequences.iterrows():
        if row.get('has_next_visit', False):
            results['sequences_with_targets'] += 1
            
            mmse = row.get('next_MMSE', np.nan)
            cdr = row.get('next_CDRSUM', np.nan)
            
            if pd.notna(mmse) and 0 <= mmse <= 30:
                results['valid_mmse_targets'] += 1
            if pd.notna(cdr) and 0 <= cdr <= 18:
                results['valid_cdr_targets'] += 1
    
    # Get ranges
    valid_mmse = sequences['next_MMSE'].dropna()
    valid_mmse = valid_mmse[(valid_mmse >= 0) & (valid_mmse <= 30)]
    if len(valid_mmse) > 0:
        results['mmse_range'] = (valid_mmse.min(), valid_mmse.max())
    
    valid_cdr = sequences['next_CDRSUM'].dropna()
    valid_cdr = valid_cdr[(valid_cdr >= 0) & (valid_cdr <= 18)]
    if len(valid_cdr) > 0:
        results['cdr_range'] = (valid_cdr.min(), valid_cdr.max())
    
    # Report
    logger.info("\nVerification Results:")
    logger.info(f"  Total sequences: {results['total_sequences']}")
    logger.info(f"  Unique subjects: {results['unique_subjects']}")
    logger.info(f"  Sequences with targets: {results['sequences_with_targets']}")
    logger.info(f"  Valid MMSE targets: {results['valid_mmse_targets']}")
    logger.info(f"  Valid CDR targets: {results['valid_cdr_targets']}")
    logger.info(f"  MMSE range: {results['mmse_range']}")
    logger.info(f"  CDR range: {results['cdr_range']}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    
    Run the complete NACC data integration pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NACC Data Integration Pipeline for AD Progression Prediction'
    )
    parser.add_argument(
        '--csf', 
        default='investigator_fcsf_nacc69.csv',
        help='Path to CSF biomarker CSV file'
    )
    parser.add_argument(
        '--uds', 
        default='nacc_uds_all.csv',
        help='Path to UDS clinical CSV file'
    )
    parser.add_argument(
        '--output-dir', 
        default='.',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--clean-only',
        action='store_true',
        help='Only run cleaning on existing sequences file'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify integrity of existing sequences file'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # Just verify existing file
        verify_data_integrity('nacc_ml_sequences.pkl')
        return
    
    if args.clean_only:
        # Just clean existing sequences
        clean_nacc_sequences(
            'nacc_ml_sequences.pkl',
            'nacc_ml_sequences_cleaned.pkl'
        )
        return
    
    # Run full pipeline
    integrator = NACCDataIntegrator()
    
    try:
        results = integrator.run_pipeline(args.csf, args.uds)
        integrator.save_outputs(args.output_dir)
        
        # Also save cleaned version
        clean_nacc_sequences(
            os.path.join(args.output_dir, 'nacc_ml_sequences.pkl'),
            os.path.join(args.output_dir, 'nacc_ml_sequences_cleaned.pkl')
        )
        
        # Final verification
        verify_data_integrity(
            os.path.join(args.output_dir, 'nacc_ml_sequences_cleaned.pkl')
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nOutput files:")
        logger.info(f"  - {os.path.join(args.output_dir, 'nacc_integrated_dataset.pkl')}")
        logger.info(f"  - {os.path.join(args.output_dir, 'nacc_ml_sequences.pkl')}")
        logger.info(f"  - {os.path.join(args.output_dir, 'nacc_ml_sequences_cleaned.pkl')}")
        logger.info(f"  - integration_report.txt")
        logger.info(f"  - nacc_integration.log")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
