import numpy as np

class PhenotypicGatingPipeline:
    def __init__(self, engine):
        """
        Initializes the gating pipeline with the updated TrueSequenceEngine.
        """
        self.engine = engine
        
        # Historical validation thresholds from the Master Key
        self.BPD_Z_THRESHOLD = 2.0
        self.NPD_Z_THRESHOLD = 1.5
        self.AUTISM_Z_THRESHOLD = 2.0
        self.DEPRESSION_Z_THRESHOLD = 1.5

    def process_live_stream_event(self, press_interval, is_space_or_punctuation=False):
        """
        Ingests sequential time-series events into the streaming engine.
        If a threshold-driven window (Complexity Stall horizon) completes,
        it evaluates the state-space coordinates against the phenotypic gates.
        """
        # Feed the raw interval into the updated TrueSequenceEngine state machine
        engine_output = self.engine.ingest_keystroke_stream(press_interval, is_space_or_punctuation)
        
        # If the engine hasn't completed a full 10-step post-stall horizon yet, it returns None
        if engine_output is None:
            return None
            
        # Extract the calculated Z-scores from the engine's real-time dictionary
        z_matrix = engine_output["Z_Matrix_Drift"]
        z_lbir = engine_output["Z_LBIR"]
        z_lambda = engine_output["Z_Lambda"]
        
        # Route state-space coordinates through successive filtration gates
        classification = self.apply_filtration_gates(z_matrix, z_lbir, z_lambda)
        
        return {
            "metrics": {
                "lbir": float(engine_output["LBIR"]),
                "lambda_coef": float(engine_output["Lambda"])
            },
            "coordinates": {
                "z_matrix": float(z_matrix),
                "z_lbir": float(z_lbir),
                "z_lambda": float(z_lambda)
            },
            "phenotype_classification": classification
        }

    def apply_filtration_gates(self, z_matrix, z_lbir, z_lambda):
        """
        Executes explicit parameter masks to isolate high-tension network profiles.
        """
        # Quadrant II Mask: Borderline Volatile Manifold
        if z_matrix >= self.BPD_Z_THRESHOLD and z_lbir < 0.0 and z_lambda < 0.0:
            return "Quadrant II: Borderline Volatile Manifold (Loose Seam Boundary)"
            
        # Quadrant IV Mask: Narcissistic Static Stance
        elif z_matrix >= self.NPD_Z_THRESHOLD and z_lbir < 0.0 and z_lambda >= 0.0:
            return "Quadrant IV: Narcissistic Static Stance (Ironclad Algorithmic Lockout)"
            
        # Autism Spectrum Mask: Hyper-Isolated Loop Rigidity
        elif z_matrix >= self.AUTISM_Z_THRESHOLD and z_lbir > 0.0 and z_lambda >= 0.0:
            return "Autism Spectrum Configuration (Hyper-Isolated Operational Rigidity)"
            
        # Depressive Vector Mask: Kinetic Velocity Deceleration
        elif z_matrix >= self.DEPRESSION_Z_THRESHOLD and z_lambda < -1.5:
            return "Depressive Vector (Kinetic Velocity Deceleration / Persistent Stagnation)"
            
        # Default State: Invariant Homeostatic Field Engagement
        else:
            return "Nominal Volley Architecture (Stable Processing Baseline)"
