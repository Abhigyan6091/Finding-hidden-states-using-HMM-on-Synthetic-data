# Synthetic HMM Test & Model Selection

## Overview

This project implements and validates a **Hidden Markov Model (HMM) synthetic data generator** combined with an **IETE (Information-Entropy-based Temporal Estimation) model selection method** for determining the optimal number of hidden states in flight trajectory data.

The main script (`synthetic_hmm_test.py`) creates synthetic flight sequences using known HMM parameters and tests the effectiveness of the model selection approach without relying on external `.mat` files.

---

## Purpose

**Problem:** Determining the correct number of hidden states (K) in an HMM is difficult. Using too many states leads to overfitting; too few underfit the data.

**Solution:** This script implements a **penalized likelihood criterion** that combines:
- **Log-Likelihood** from the trained HMM
- **KL Divergence Stability** measured across subsampled data
- **Penalty Term** scaled by model complexity (BIC-style)

The method selects the K that minimizes: `Score(K) = -2 * LogLikelihood + λ * KL_Divergence`

---

## Features

### 1. **Synthetic Data Generation**
- Generates a 10,000-sample sequence using ground-truth HMM matrices (K=3)
- True transition matrix (A) and emission matrix (B) are predefined
- Provides realistic synthetic flight data without external dependencies

### 2. **Multi-Restart HMM Training**
- Trains categorical HMM models for K = 2, 3, 4, 5, 6 states
- Uses 10 random restarts for each K to find local optima
- Tests different initialization perturbations using `NOISE_SCALE`

### 3. **IETE Model Selection**
- Evaluates model stability by subsampling the data at different probabilities (p = 0.80, 0.85)
- Computes KL divergence between original and subsampled model parameters
- Applies bias correction formula to transition matrices
- Combines log-likelihood and KL divergence for final model selection

### 4. **Validation Metrics**
- **State Recovery Accuracy:** How well the learned model recovers the true hidden states
- **Matrix Recovery:** Compares learned A and B matrices to ground truth
- **Confusion Matrix Analysis:** Maps learned states to true states (accounting for label permutation)

### 5. **Visualization**
Generates a 4-panel plot:
1. **KL Heatmap:** Stability across subsampling probabilities and K values
2. **Selection Scores:** Penalized score for each K (highlights optimal choice)
3. **Observation Sequence:** Sample of the extracted observation symbols
4. **State Recovery:** True vs. learned hidden states with accuracy metric

---

## Installation

### Requirements
```bash
pip install numpy matplotlib hmmlearn scipy scikit-learn
```

### Dependencies
- **numpy:** Numerical computations and matrix operations
- **matplotlib:** Visualization and plotting
- **hmmlearn:** Hidden Markov Model training (`CategoricalHMM`)
- **scipy:** Optimization and signal processing utilities
- **scikit-learn:** (optional) Additional ML utilities

---

## Usage

### Basic Run
```bash
python synthetic_hmm_test.py
```

### Configuration (Edit in Script)
Key parameters can be modified at the top of `synthetic_hmm_test.py`:

```python
SEQ_LENGTH   = 10000              # Length of synthetic sequence
PROB_VEC     = [0.80, 0.85]       # Subsampling probabilities for stability test
STATE_RANGE  = [2, 3, 4, 5, 6]    # K values to test
MAX_ITER     = 1000               # Max EM iterations
TOL          = 1e-6               # Convergence tolerance
NUM_RESTARTS = 10                 # Number of random restarts per K
NOISE_SCALE  = 0.05               # Perturbation scale for initializations
```

### Expected Output
```
=== SYNTHETIC HMM DATA GENERATOR & TESTER ===
Generating synthetic flight of length 10000 from TRUE matrices (K=3)...
Generated 10000 symbols.
  Symbol 0: 3380 (33.80%)
  Symbol 1: 3295 (32.95%)
  Symbol 2: 3325 (33.25%)

States occupancy (ground truth):
  State 0: 3333 instances
  State 1: 3334 instances
  State 2: 3333 instances

--- Training on Synthetic Data (multi-restart) ---
  K= 2 | Best Log-Likelihood: -9650.95
  K= 3 | Best Log-Likelihood: -8950.42
  K= 4 | Best Log-Likelihood: -8945.78
  ...

=== IETE Model Selection Results on Synthetic Data ===
  K = 2 | Mean KL: 0.025430 | LogLikelihood: -9650.95 | Score: 19450.23
  K = 3 | Mean KL: 0.001250 | LogLikelihood: -8950.42 | Score: 17958.12  <-- OPTIMUM MODEL
  K = 4 | Mean KL: 0.003450 | LogLikelihood: -8945.78 | Score: 17982.34
  ...

=== Comparing Learned K=3 Matrices to Ground Truth ===
True A matrix:
[[0.8  0.15 0.05]
 [0.1  0.8  0.1 ]
 [0.05 0.15 0.8 ]]

Learned A matrix (from K=3 testing):
[[0.79  0.16 0.05]
 [0.11  0.79 0.1 ]
 [0.05  0.14 0.81]]

=== Viterbi Decoding against Ground Truth ===
State recovery accuracy: 98.45% (considering state permutations)

Plot saved to synthetic_iete_result.png
```

---

## Output Files

- **`synthetic_iete_result.png`** – 4-panel validation plot showing:
  - KL divergence heatmap
  - Model selection scores across K values
  - Sample observations from synthetic sequence
  - State recovery comparison (true vs. learned)

---

## Mathematical Foundation

### 1. Transition Matrix Bias Correction
When training on a subsample (probability p), the learned transition matrix A_bw has bias. We apply:

$$A_c = A_{bw} \cdot (p \cdot (I - (1-p) \cdot A_{bw})^{-1})$$

This corrected matrix is used to compute KL divergence.

### 2. KL Divergence
For comparing probability distributions P and Q:

$$KL(P \Vert Q) = \sum_{i} P_i \log_2 \left(\frac{P_i}{Q_i}\right)$$

### 3. Penalized Likelihood Criterion
Dynamic penalty scaling (BIC-style):

$$\text{Score}(K) = -2 \cdot \log L(K) + \lambda \cdot KL(K)$$

where $\lambda = 500 \cdot \log(N)$ accounts for dataset size.

---

## Interpretation of Results

### When Model Selection Succeeds
- **Optimal K matches ground truth (K=3)** – Low KL divergence indicates stable parameters
- **Log-likelihood improves** as K increases up to the true K, then plateaus/decreases
- **High state recovery accuracy** (>95%) – Learned states align with true states

### Common Issues
1. **All scores are very similar** – Increase penalty lambda or check data quality
2. **K=2 selected instead of K=3** – Reduce NUM_RESTARTS or NOISE_SCALE
3. **State recovery accuracy is low** – Check that generated data actually contains 3 distinct patterns

---

## Extensions & Future Work

1. **Real Flight Data:** Replace synthetic generation with actual flight trajectory data
2. **Continuous Observations:** Adapt to Gaussian HMM for continuous altitude/speed data
3. **Hierarchical Clustering:** Combine this with multi-flight clustering
4. **Streaming Updates:** Implement online/incremental HMM training for real-time detection
5. **Cross-Validation:** Replace subsampling with proper K-fold cross-validation

---

## References

- **hmmlearn Documentation:** https://hmmlearn.readthedocs.io/
- **Baum-Welch Algorithm:** Core EM algorithm for HMM training
- **KL Divergence:** Information-theoretic measure of distribution difference
- **BIC (Bayesian Information Criterion):** Model selection penalty framework

---

## License & Disclaimer

This code is provided as-is for research and educational purposes. Use at your own discretion for production systems.

---

## Author Notes

The IETE approach combines information-theoretic stability metrics with traditional likelihood-based model selection. It is particularly effective for:
- **Aviation anomaly detection:** Classifying normal vs. irregular flight patterns
- **Time-series segmentation:** Identifying regime changes in sequential data
- **Dimensionality in hidden processes:** Automatically determining model complexity

For questions or improvements, refer to the comments in `synthetic_hmm_test.py`.
