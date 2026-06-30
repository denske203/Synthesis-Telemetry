import numpy as np
import pandas as pd

class NeuroKinematicEngine:
    def __init__(self):
        # Fixed baseline centers calibrated from the unpolluted Aalto floor (N=3,153)
        # Note: These are population baselines, not hard universal ceiling limits.
        self.MU_LBIR = 3.7928
        self.MU_LAMBDA = -0.9609

    def apply_somatic_filter(self, raw_intervals):
        """
        Applies the Global Time-Series Run Configuration:
        Individual Dynamic Gate = Mean + 1.5 * StdDev.
        Masks transient somatic anomalies (EMG/jaw-clench noise).
        """
        if len(raw_intervals) < 10:
            return raw_intervals
            
        mean_val = np.mean(raw_intervals)
        std_val = np.std(raw_intervals)
        dynamic_gate = mean_val + (1.5 * std_val)
        
        # Clean out edge-ringing and transient mechanical/somatic artifacts
        clean_intervals = np.where(raw_intervals > dynamic_gate, np.nan, raw_intervals)
        return pd.Series(clean_intervals).interpolate(method='linear').bfill().ffill().values

    def extract_lbir(self, intra_word_intervals, hard_boundary_intervals):
        """
        Calculates Dimension 1: Lexical Boundary Invariance Ratio (LBIR).
        Measures the absolute variance of out-of-sample standardized boundary deviations.
        """
        mu_intra = np.mean(intra_word_intervals)
        sigma_intra = np.std(intra_word_intervals)
        if sigma_intra == 0: 
            sigma_intra = 1.0
            
        # Standardize boundary interactions out-of-sample
        z_hard = (np.array(hard_boundary_intervals) - mu_intra) / sigma_intra
        
        # LBIR is the absolute statistical variance of the deviation vector
        return np.var(z_hard)

    def extract_traction_recovery(self, post_stall_horizon, baseline_rec_80th):
        """
        Calculates Dimension 2: Traction Recovery Coefficient (Lambda).
        Measures log-decay velocity across a 10-step horizon following a Complexity Stall.
        The stall event triggers a universal fractional latency surge (25.77%), 
        but the ceiling threshold where it triggers is highly variable across individuals.
        """
        if len(post_stall_horizon) < 6:
            return self.MU_LAMBDA
            
        # Normalize the horizon vector against the user's 80th-percentile recovery baseline
        vector_norm = post_stall_horizon / (baseline_rec_80th + 1e-6)
        
        epsilon_1 = vector_norm[0] - 1.0
        epsilon_k = np.mean(vector_norm[1:6]) - 1.0  # Composite residual friction bounds
        
        if epsilon_1 == 0 or epsilon_k == 0:
            return 0.0
            
        # Structural decay velocity formula
        lambda_coef = -np.log(np.abs(epsilon_k / epsilon_1))
        return lambda_coef

    def evaluate_topology_drift(self, lbir, lambda_coef, cohort_std_lbir=1.0, cohort_std_lambda=1.0):
        """
        Resolves the absolute topological drift vector distance (Z_matrix)
        relative to the fixed historical population baselines.
        """
        z_lbir = (lbir - self.MU_LBIR) / cohort_std_lbir
        z_lambda = (lambda_coef - self.MU_LAMBDA) / cohort_std_lambda
        
        z_matrix = np.sqrt(z_lbir**2 + z_lambda**2)
        return z_matrix, z_lbir, z_lambda
