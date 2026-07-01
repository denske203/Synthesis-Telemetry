import numpy as np
import pandas as pd
from collections import deque

class TrueSequenceEngine:
    def __init__(self, baseline_rec_80th=150.0):
        # Fixed baseline centers calibrated from the unpolluted Aalto floor (N=3,153)
        self.MU_LBIR = 3.7928
        self.MU_LAMBDA = -0.9609
        
        # User-specific normalization constraints
        self.baseline_rec_80th = baseline_rec_80th
        
        # Real-time streaming buffers for rolling time-series tracking
        self.raw_interval_history = deque(maxlen=100)
        self.intra_word_buffer = deque(maxlen=50)
        self.hard_boundary_buffer = deque(maxlen=20)
        
        # Stall monitoring state machine
        self.in_post_stall_horizon = False
        self.stall_horizon_buffer = []
        self.STALL_SURGE_THRESHOLD = 1.2577 # Explicit 25.77% global latency surge

    def apply_somatic_filter(self, raw_intervals):
        """
        Applies the Global Time-Series Run Configuration.
        Masks transient somatic anomalies (EMG artifacts, jaw-clenches, edge-ringing).
        """
        if len(raw_intervals) < 10:
            return raw_intervals
            
        mean_val = np.mean(raw_intervals)
        std_val = np.std(raw_intervals)
        dynamic_gate = mean_val + (1.5 * std_val)
        
        clean_intervals = np.where(raw_intervals > dynamic_gate, np.nan, raw_intervals)
        return pd.Series(clean_intervals).interpolate(method='linear').bfill().ffill().values

    def ingest_keystroke_stream(self, press_interval, is_space_or_punctuation=False):
        """
        Core Streaming Ingestion Engine. Processes raw intervals sequentially,
        dynamically routing features and capturing threshold-driven network events.
        """
        # 1. Sanitize incoming raw interval using rolling somatic historical gate
        self.raw_interval_history.append(press_interval)
        filtered_intervals = self.apply_somatic_filter(np.array(self.raw_interval_history))
        sanitized_interval = filtered_intervals[-1]

        # 2. Dynamic Routing Layer based on Lexical Boundaries
        if is_space_or_punctuation:
            self.hard_boundary_buffer.append(sanitized_interval)
        else:
            self.intra_word_buffer.append(sanitized_interval)

        # 3. Real-Time Complexity Stall Detection State Machine
        metrics_output = None
        
        if self.in_post_stall_horizon:
            self.stall_horizon_buffer.append(sanitized_interval)
            if len(self.stall_horizon_buffer) >= 10:
                # Horizon complete: Calculate dynamic coefficients
                metrics_output = self._process_completed_horizon()
        else:
            # Check for sudden 25.77% threshold-driven network surge against running intra-word median
            if len(self.intra_word_buffer) > 5:
                running_median = np.median(list(self.intra_word_buffer))
                if sanitized_interval >= (running_median * self.STALL_SURGE_THRESHOLD):
                    # Complexity Stall Event Triggered
                    self.in_post_stall_horizon = True
                    self.stall_horizon_buffer = [sanitized_interval]
                    
        return metrics_output

    def _process_completed_horizon(self):
        """
        Internal execution block triggered instantly upon completion of a 10-step post-stall horizon.
        """
        lbir = self.calculate_lbir()
        lambda_coef = self.calculate_traction_recovery(np.array(self.stall_horizon_buffer))
        z_matrix, z_lbir, z_lambda = self.evaluate_topology_drift(lbir, lambda_coef)
        
        # Reset state machine tracking
        self.in_post_stall_horizon = False
        self.stall_horizon_buffer = []
        
        return {
            "Z_Matrix_Drift": z_matrix,
            "Z_LBIR": z_lbir,
            "Z_Lambda": z_lambda,
            "LBIR": lbir,
            "Lambda": lambda_coef
        }

    def calculate_lbir(self):
        """
        Dimension 1: Lexical Boundary Invariance Ratio (LBIR).
        Measures the absolute variance of out-of-sample standardized boundary deviations.
        """
        if len(self.intra_word_buffer) < 5 or len(self.hard_boundary_buffer) < 3:
            return self.MU_LBIR
            
        mu_intra = np.mean(list(self.intra_word_buffer))
        sigma_intra = np.std(list(self.intra_word_buffer))
        if sigma_intra == 0: 
            sigma_intra = 1.0
            
        z_hard = (np.array(list(self.hard_boundary_buffer)) - mu_intra) / sigma_intra
        return np.var(z_hard)

    def calculate_traction_recovery(self, post_stall_horizon):
        """
        Dimension 2: Traction Recovery Coefficient (Lambda).
        Measures log-decay velocity across the 10-step horizon following a Complexity Stall event.
        """
        if len(post_stall_horizon) < 6:
            return self.MU_LAMBDA
            
        vector_norm = post_stall_horizon / (self.baseline_rec_80th + 1e-6)
        
        epsilon_1 = vector_norm[0] - 1.0
        epsilon_k = np.mean(vector_norm[1:6]) - 1.0 
        
        if epsilon_1 == 0 or epsilon_k == 0:
            return 0.0
            
        return -np.log(np.abs(epsilon_k / epsilon_1))

    def evaluate_topology_drift(self, lbir, lambda_coef, cohort_std_lbir=1.0, cohort_std_lambda=1.0):
        """
        Resolves the absolute topological drift vector distance (Z_matrix)
        relative to the fixed historical population baselines.
        """
        z_lbir = (lbir - self.MU_LBIR) / cohort_std_lbir
        z_lambda = (lambda_coef - self.MU_LAMBDA) / cohort_std_lambda
        
        z_matrix = np.sqrt(z_lbir**2 + z_lambda**2)
        return z_matrix, z_lbir, z_lambda
