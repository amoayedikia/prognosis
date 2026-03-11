# PROGRESS: PRognostic Generalization from REsting Static Signatures

A dual-model deep learning framework for Alzheimer's disease progression prediction using baseline CSF biomarkers.

## Overview

PROGRESS transforms a single baseline cerebrospinal fluid (CSF) biomarker assessment into actionable prognostic estimates without requiring prior clinical history. The framework addresses two complementary clinical questions:

1. **Trajectory Prediction**: A probabilistic network predicts individualized cognitive decline parameters (intercept, slope, acceleration) with calibrated uncertainty bounds
2. **Survival Prediction**: A deep survival model estimates time-to-conversion from MCI to dementia

## Key Results

- **Survival Prediction**: C-index = 0.83 (held-out test), outperforming Cox PH, Random Survival Forests, and DeepSurv
- **Risk Stratification**: Identifies patient groups with 7-fold differences in conversion rates (6% vs 43%)
- **Cross-Center Validation**: C-index > 0.90 across all 43 ADRCs in leave-one-center-out validation
- **Uncertainty Calibration**: Prediction interval coverage probability 95-98% across trajectory parameters

## Repository Structure

```
├── PROGRESS.py                      # Main dual-model framework
├── NACCDataIntegrator.py            # NACC data integration pipeline
├── PROGRESS_experiments.py          # Main experimental pipeline
├── baseline_comparison.py           # Comparison with baseline methods
├── complete_baseline_comparison.py  # Extended baseline comparisons
├── cross_center_validation.py       # Leave-One-Center-Out validation
├── cross_center_lr_analysis.py      # Cross-center learning rate analysis
├── demographic_fairness_analysis.py # Fairness analysis (sex, age, education)
├── demographic_fairness_analysis_v2.py  # Updated fairness analysis
├── survival_significance_tests.py   # Statistical significance testing
├── unified_comparison.py            # Unified method comparison
├── proper_unified_comparison.py     # Proper unified comparison
├── fair_unified_comparison.py       # Fair unified comparison framework
├── progress_ablation_actual.py      # Ablation studies
└── run_baseline_comparison.py       # Runner script for baselines
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-survival
- lifelines
- pandas
- numpy
- scipy

## Data

This study uses data from the National Alzheimer's Coordinating Center (NACC):
- CSF biomarkers: Aβ42, p-tau, t-tau
- Longitudinal cognitive assessments from the Uniform Data Set (UDS)
- 3,051 participants across 43 Alzheimer's Disease Research Centers

Data access requires approval from NACC: https://naccdata.org/

## Usage

```python
from PROGRESS import PROGRESSFramework
from NACCDataIntegrator import NACCDataIntegrator

# Load and integrate data
integrator = NACCDataIntegrator()
data = integrator.integrate()

# Initialize and train PROGRESS
model = PROGRESSFramework()
model.fit(data)

# Predict
trajectory_params, survival_curves = model.predict(new_patient_features)
```

## Citation

If you use this code, please cite:

```bibtex
@article{moayedikia2025progress,
  title={Dual-Model Deep Learning for Alzheimer's Prognostication},
  author={Fin, Sara and Moayedikia, Alireza and Wiil, Uffe Kock},
  journal={Computers in Biology and Medicine},
  year={2025},
  note={Under Review}
}
```

## Related Work

This work builds upon our previous research on multi-objective optimization for AD clinical trial patient selection:

```bibtex
@article{moayedikia2025jbi,
  title={Multi-objective optimization formulation for Alzheimer's disease trial patient selection},
  author={Moayedikia, Alireza and Fin, Sara and Wiil, Uffe Kock},
  journal={Journal of Biomedical Informatics},
  volume={172},
  pages={104955},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

The NACC database is funded by NIA/NIH Grant U24 AG072122. NACC data are contributed by the NIA-funded ADRCs.

## Contact

Alireza Moayedikia (Corresponding Author)  
Department of Business Technology and Entrepreneurship  
Swinburne University of Technology  
Email: amoayedikia@swin.edu.au
