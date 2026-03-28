"""
synthetic_hmm_test.py
=====================
Generates a custom HMM sequence using the exact transition and emission 
matrices for K=3, pretending it's a single quantized flight. 
It then tests our hmmlearn CategoricalHMM training approach on this 
synthetic sequence without relying on .mat files.
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  USER SETTINGS
# ============================================================
SEQ_LENGTH   = 10000                   # 10,000 samples
PROB_VEC     = [0.80, 0.85]            # Lower probabilities to stress test
STATE_RANGE  = [2, 3, 4, 5, 6]         # Testing K=2 through 6
MAX_ITER     = 1000
TOL          = 1e-6
NUM_RESTARTS = 10
NOISE_SCALE  = 0.05
# ============================================================

# ============================================================
#  1. GENERATE CUSTOM HMM SERIES (K=3)
# ============================================================
# The exact starting conditions for K=3 we had earlier:
A_true = np.array([
    [0.80, 0.15, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.15, 0.80]
])

B_true = np.array([
    [0.70, 0.20, 0.10],
    [0.20, 0.60, 0.20],
    [0.10, 0.20, 0.70]
])

def generate_flight(length, A, B):
    """Generates an observation sequence based on the A and B matrices."""
    K, num_obs = B.shape
    
    # Set up the exact "true" model
    true_model = hmm.CategoricalHMM(n_components=K, init_params="", params="")
    true_model.startprob_ = np.ones(K) / K
    true_model.transmat_ = A
    true_model.emissionprob_ = B

    # Generate sequence
    X, Z = true_model.sample(length)
    return X.ravel(), Z.ravel(), num_obs

# ============================================================
#  HELPER FUNCTIONS (From our approach)
# ============================================================
def perturb_matrix(M: np.ndarray, scale: float) -> np.ndarray:
    noise = scale * (np.random.rand(*M.shape) - 0.5)
    M_noisy = M + noise
    M_noisy = np.maximum(M_noisy, 1e-6)
    return M_noisy / M_noisy.sum(axis=1, keepdims=True)

def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    P_flat, Q_flat = P.ravel(), Q.ravel()
    valid = (P_flat > 0) & (Q_flat > 0)
    kld = np.sum(P_flat[valid] * np.log2(P_flat[valid] / Q_flat[valid]))
    return float(kld) if not np.isnan(kld) else np.inf

def corrected_A(A_bw: np.ndarray, p: float) -> np.ndarray:
    K = A_bw.shape[0]
    I_K = np.eye(K)
    try:
        A_c = A_bw @ (p * np.linalg.inv(I_K - (1 - p) * A_bw)) # our formula
    except np.linalg.LinAlgError:
        A_c = A_bw.copy()
    A_c = np.maximum(A_c, 1e-10)
    return A_c / A_c.sum(axis=1, keepdims=True)

def train_hmm(obs, A_init, B_init, num_symbols):
    K = A_init.shape[0]
    model = hmm.CategoricalHMM(n_components=K, n_iter=MAX_ITER, tol=TOL,
                               init_params="", params="ste")
    model.startprob_ = np.ones(K) / K
    model.transmat_ = A_init
    model.emissionprob_ = B_init
    
    try:
        model.fit(obs.reshape(-1, 1).astype(int))
        ll = model.score(obs.reshape(-1, 1).astype(int))
        return model.transmat_, model.emissionprob_, ll, model
    except ValueError:
        return A_init, B_init, -np.inf, None 

# Extract base parameter builder logic from earlier script
def build_base_params(num_obs: int):
    A_base, B_base = {}, {}
    
    if num_obs == 3:
        # 3-symbol initializations
        A_base[2] = np.array([[0.85, 0.15], [0.20, 0.80]])
        B_base[2] = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
        
        A_base[3] = A_true.copy()
        B_base[3] = B_true.copy()
        
        A_base[4] = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.75, 0.1, 0.05],
            [0.05, 0.1, 0.8, 0.05],
            [0.05, 0.05, 0.1, 0.8]
        ])
        B_base[4] = np.array([
            [0.8, 0.15, 0.05],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6],
            [0.05, 0.15, 0.8]
        ])
        
        A_base[5] = np.array([
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.75, 0.1, 0.03, 0.02],
            [0.05, 0.1, 0.75, 0.08, 0.02],
            [0.03, 0.03, 0.08, 0.80, 0.06],
            [0.02, 0.02, 0.02, 0.06, 0.88]
        ])
        B_base[5] = np.array([
            [0.8, 0.15, 0.05],
            [0.5, 0.4, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.4, 0.5],
            [0.05, 0.2, 0.75]
        ])
        
        A_base[6] = np.eye(6) * 0.7 + 0.06
        A_base[6] /= A_base[6].sum(axis=1, keepdims=True)
        B_base[6] = np.ones((6, 3)) / 3.0
        
    else:  # num_obs == 5
        A_base[2] = np.array([[0.85, 0.15], [0.20, 0.80]])
        B_base[2] = np.array([[0.35, 0.25, 0.20, 0.12, 0.08], [0.08, 0.12, 0.20, 0.25, 0.35]])
        
        A_base[3] = np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])
        B_base[3] = np.array([[0.40, 0.30, 0.15, 0.10, 0.05], [0.05, 0.15, 0.60, 0.15, 0.05], [0.05, 0.10, 0.15, 0.30, 0.40]])
        
        A_base[4] = np.array([[0.8, 0.1, 0.05, 0.05],[0.1, 0.75, 0.1, 0.05],[0.05, 0.1, 0.8, 0.05],[0.05, 0.05, 0.1, 0.8]])
        B_base[4] = np.array([[0.45, 0.25, 0.15, 0.10, 0.05],[0.10, 0.35, 0.30, 0.15, 0.10],[0.10, 0.15, 0.30, 0.35, 0.10],[0.05, 0.10, 0.15, 0.25, 0.45]])
        
        A_base[5] = np.array([[0.80, 0.10, 0.05, 0.03, 0.02], [0.10, 0.75, 0.10, 0.03, 0.02], [0.05, 0.10, 0.75, 0.08, 0.02], [0.03, 0.03, 0.08, 0.80, 0.06], [0.02, 0.02, 0.02, 0.06, 0.88]])
        B_base[5] = np.array([[0.50, 0.25, 0.12, 0.08, 0.05], [0.15, 0.40, 0.25, 0.12, 0.08], [0.05, 0.12, 0.66, 0.12, 0.05], [0.08, 0.12, 0.25, 0.40, 0.15], [0.05, 0.08, 0.12, 0.25, 0.50]])
        
        A_base[6] = np.eye(6) * 0.7 + 0.06
        A_base[6] /= A_base[6].sum(axis=1, keepdims=True)
        B_base[6] = np.ones((6, 5)) / 5.0

    return A_base, B_base


def main():
    np.random.seed(42) # Ensuring reproducibility of synthetic data
    
    print("=== SYNTHETIC HMM DATA GENERATOR & TESTER ===")
    print(f"Generating synthetic flight of length {SEQ_LENGTH} from TRUE matrices (K=3)...")
    obs, true_states, num_obs = generate_flight(SEQ_LENGTH, A_true, B_true)
    
    print(f"Generated {len(obs)} symbols.")
    for s in range(num_obs):
        print(f"  Symbol {s}: {sum(obs==s)} ({100.*sum(obs==s)/len(obs):.2f}%)")
        
    print(f"\nStates occupancy (ground truth):")
    for s in range(3):
        print(f"  State {s}: {sum(true_states==s)} instances")
        
    # Get base initialisation matrices
    A_base_dict, B_base_dict = build_base_params(num_obs)
    
    # ==========================================================
    #  TESTING OUR APPROACH ON THIS DATA 
    # ==========================================================
    print("\n--- Training on Synthetic Data (multi-restart) ---")
    A_bw, B_bw = {}, {}
    best_ll_dict = {} # Initialize best_ll_dict here
    best_base_models = {} # Initialize best_base_models here
    for ki, K in enumerate(STATE_RANGE):
        current_best_ll = -np.inf
        current_best_model = None
        for r in range(NUM_RESTARTS):
            A0 = perturb_matrix(A_base_dict[K], NOISE_SCALE)
            B0 = perturb_matrix(B_base_dict[K], NOISE_SCALE)
            A_k, B_k, ll_k, model_k = train_hmm(obs, A0, B0, num_obs)
            if ll_k > current_best_ll:
                current_best_ll, A_bw[K], B_bw[K], current_best_model = ll_k, A_k, B_k, model_k
        best_ll_dict[K] = current_best_ll
        best_base_models[K] = current_best_model
        print(f"  K={K:>2} | Best Log-Likelihood: {current_best_ll:.2f}")

    print("\n--- Running IETE Subsampling KL Divergence with Penalty ---")
    
    # We will score models using a penalized likelihood criterion:
    # Score(K) = -2 * LogLikelihood + lambda * KL_Divergence
    # We want to MINIMIZE this score. The KL acts as a strong penalty for instability.
    # We make lambda dynamic so it scales with data volume (BIC-style)
    penalty_lambda = 500 * np.log(len(obs)) 
    
    kl_results = {}
    score_results = {}
    
    # Store individual KLD values for the heatmap
    delta_matrix = np.full((len(PROB_VEC), len(STATE_RANGE)), np.nan)
    
    for ki, K in enumerate(STATE_RANGE):
        # 1) Get the best full-data model's log-likelihood
        ll_best = best_ll_dict[K]
        
        # 2) Calculate KL divergence across all subsamplings
        kl_list = []
        for pi, p in enumerate(PROB_VEC):
            subsample_len = int(p * len(obs))
            sub_obs = obs[:subsample_len]
            
            # Train on subsample
            model_sub = hmm.CategoricalHMM(n_components=K, n_iter=MAX_ITER, tol=TOL, init_params="", params="ste")
            if best_base_models[K] is not None:
                model_sub.startprob_ = best_base_models[K].startprob_
                model_sub.transmat_ = best_base_models[K].transmat_
                model_sub.emissionprob_ = best_base_models[K].emissionprob_
            else:
                model_sub.startprob_ = np.ones(K) / K
                model_sub.transmat_ = perturb_matrix(A_base_dict[K], NOISE_SCALE)
                model_sub.emissionprob_ = perturb_matrix(B_base_dict[K], NOISE_SCALE)
            
            try:
                model_sub.fit(sub_obs.reshape(-1, 1).astype(int))
                A_sub = model_sub.transmat_
                A_c = corrected_A(A_sub, p)
                
                # Calculate raw KL Div (Sum over matrix)
                P, Q = A_sub.ravel(), A_c.ravel()
                valid = (P > 0) & (Q > 0)
                if np.any(valid):
                    kl_sum = np.sum(P[valid] * np.log10(P[valid] / Q[valid]))
                    delta_matrix[pi, ki] = kl_sum
                    kl_list.append(kl_sum)
                else:
                    delta_matrix[pi, ki] = np.inf
                    kl_list.append(np.inf)
            except Exception:
                continue
            
        mean_kl = np.mean(kl_list) if kl_list else np.inf # Handle empty kl_list
        kl_results[K] = mean_kl
        
        # Calculate final Information Criterion Score
        score = -2 * ll_best + penalty_lambda * mean_kl
        score_results[K] = score

    # Find the optimum K (lowest score)
    # Filter out K values that resulted in infinite scores (e.g., due to failed training or KL)
    valid_scores = {k: v for k, v in score_results.items() if not np.isinf(v)}
    if valid_scores:
        best_K_sel = min(valid_scores, key=valid_scores.get)
    else:
        best_K_sel = None # No valid model found

    print("\n=== IETE Model Selection Results on Synthetic Data ===")
    for K in STATE_RANGE:
        marker = "  <-- OPTIMUM MODEL" if K == best_K_sel else ""
        print(f"  K = {K} | Mean KL: {kl_results.get(K, np.nan):.6f} | LogLikelihood: {best_ll_dict.get(K, np.nan):.1f} | Score: {score_results.get(K, np.nan):.1f}{marker}")
    
    print(f"\nSince the true data WAS generated with K=3, the method optimally should choose K=3!")
        
    print(f"\n=== Comparing Learned K=3 Matrices to Ground Truth ===")
    print("True A matrix:")
    print(np.round(A_true, 4))
    print("Learned A matrix (from K=3 testing):")
    print(np.round(A_bw[3], 4))
    
    print("\nTrue B matrix:")
    print(np.round(B_true, 4))
    print("Learned B matrix (from K=3 testing):")
    print(np.round(B_bw[3], 4))
    
    # Decoding against ground truth
    print("\n=== Viterbi Decoding against Ground Truth ===")
    model_decode = hmm.CategoricalHMM(n_components=3, init_params="", params="")
    model_decode.startprob_ = np.ones(3)/3
    model_decode.transmat_ = A_bw[3]
    model_decode.emissionprob_ = B_bw[3]
    learned_states = model_decode.predict(obs.reshape(-1, 1).astype(int))
    
    # Permutation matching (states might have swapped identities, though less likely if initialized well)
    from scipy.optimize import linear_sum_assignment
    
    # Create confusion matrix to map learned states to true states optimally
    conf_mat = np.zeros((3, 3), dtype=int)
    for t_s, l_s in zip(true_states, learned_states):
        conf_mat[t_s, l_s] += 1
        
    row_ind, col_ind = linear_sum_assignment(conf_mat.max() - conf_mat)
    mapped_learned_states = np.zeros_like(learned_states)
    for r, c in zip(row_ind, col_ind):
        mapped_learned_states[learned_states == c] = r
        
    accuracy = np.mean(mapped_learned_states == true_states) * 100
    print(f"State recovery accuracy: {accuracy:.2f}% (considering state permutations)")
    
    # ============================================================
    #  VISUALIZATION (4-Panel Plot)
    # ============================================================
    print("\nGenerating Synthetic Validation Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"HMM Model-Order Selection — Synthetic Validation (K=3 Ground Truth)", fontsize=13)

    # Plot 1: KL Heatmap
    ax = axes[0, 0]
    im = ax.imshow(delta_matrix, aspect="auto",
                   extent=[STATE_RANGE[0] - 0.5, STATE_RANGE[-1] + 0.5,
                            PROB_VEC[-1] - 0.025, PROB_VEC[0] + 0.025],
                   origin="upper")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Subsampling Probability (p)")
    ax.set_title("Stability Matrix: KL Divergence δ(p, K)")
    ax.set_xticks(STATE_RANGE)

    # Plot 2: Selection Scores (Total Penalized Score)
    ax = axes[0, 1]
    total_scores = [score_results[k] for k in STATE_RANGE]
    colors_bar = ["#2196F3" if k != best_K_sel else "#FF5722" for k in STATE_RANGE]
    ax.bar(STATE_RANGE, total_scores, color=colors_bar)
    
    # Zoom in on the y-axis to make the differences visible
    y_min = min(total_scores)
    y_max = max(total_scores)
    margin = (y_max - y_min) * 0.2
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Penalized Score (-2LL + λ·KL)")
    ax.set_title(f"Minimum Score at K = {best_K_sel}")
    ax.set_xticks(STATE_RANGE)

    # Plot 3: Observation Sequence (Samples)
    ax = axes[1, 0]
    ax.scatter(range(len(obs[:2000])), obs[:2000], s=1, c="gray", alpha=0.5)
    ax.set_xlabel("Sample (First 2000 steps)")
    ax.set_ylabel("Observation Symbol")
    ax.set_title("Synthetic Sequence Symbols")
    ax.set_yticks(range(num_obs))
    ax.set_ylim(-0.5, num_obs - 0.5)

    # Plot 4: State Recovery
    ax = axes[1, 1]
    ax.scatter(range(len(true_states[:500])), true_states[:500], s=15, label="True", alpha=0.6, marker='o')
    ax.scatter(range(len(mapped_learned_states[:500])), mapped_learned_states[:500], s=15, label="Learned", alpha=0.6, marker='x')
    ax.set_xlabel("Sample (First 500 steps)")
    ax.set_ylabel("Hidden State")
    ax.set_title(f"State Recovery Accuracy: {accuracy:.1f}%")
    ax.set_yticks(range(3))
    ax.legend(loc='upper right')
    ax.set_ylim(-0.5, 2.5)

    plt.tight_layout()
    plt.savefig("synthetic_iete_result.png", dpi=150)
    print("Plot saved to synthetic_iete_result.png")
    
    print("\nSuccessfully tested custom synthetic generation and hmmlearn quantized approach!")

if __name__ == "__main__":
    main()
