import numpy as np

class PhenotypicGatingPipeline:
    def __init__(self, engine):
        """
        Initializes the gating pipeline with the core NeuroKinematicEngine.
        """
        self.engine = engine
        
        # Hard historical validation thresholds established in the Master Key
        self.BPD_Z_THRESHOLD = 2.0
        self.NPD_Z_THRESHOLD = 1.5

    def process_subject_profile(self, intra_word, hard_boundary, post_stall, baseline_80th, cohort_stds=None):
        """
        Runs raw interval vectors through the entire end-to-end pipeline:
        Somatic Filtering -> Coordinate Extraction -> Topological Drift Evaluation -> Gating
        """
        # 1. Apply global run configuration noise-filtering matrix
        clean_intra = self.engine.apply_somatic_filter(intra_word)
        clean_hard = self.engine.apply_somatic_filter(hard_boundary)
        clean_post_stall = self.engine.apply_somatic_filter(post_stall)
        
        # 2. Extract scale-invariant metrics
        lbir = self.engine.extract_lbir(clean_intra, clean_hard)
        lambda_coef = self.engine.extract_traction_recovery(clean_post_stall, baseline_80th)
        
        # 3. Standardize deviations relative to the N=3,153 population floor
        # If cohort standard deviations aren't provided, default to standard normal normalization (1.0)
        std_lbir = cohort_stds.get('lbir', 1.0) if cohort_stds else 1.0
        std_lambda = cohort_stds.get('lambda', 1.0) if cohort_stds else 1.0
        
        z_matrix, z_lbir, z_lambda = self.engine.evaluate_topology_drift(
            lbir, lambda_coef, cohort_std_lbir=std_lbir, cohort_std_lambda=std_lambda
        )
        
        # 4. Route coordinates through successive filtration gates
        classification = self.apply_filtration_gates(z_matrix, z_lbir, z_lambda)
        
        return {
            "metrics": {
                "lbir": float(lbir),
                "lambda_coef": float(lambda_coef)
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
        Executes the explicit parameter masks to isolate high-tension network profiles.
        """
        # Quadrant II Mask: Borderline Volatile Manifold
        if z_matrix >= self.BPD_Z_THRESHOLD and z_lbir < 0.0 and z_lambda < 0.0:
            return "Quadrant II: Borderline Volatile Manifold (Loose Seam Boundary)"
            
        # Quadrant IV Mask: Narcissistic Static Stance
        elif z_matrix >= self.NPD_Z_THRESHOLD and z_lbir < 0.0 and z_lambda >= 0.0:
            return "Quadrant IV: Narcissistic Static Stance (Ironclad Algorithmic Lockout)"
            
        # Default State: Invariant Homeostatic Field Engagement
        else:
            return "Nominal Volley Architecture (Stable Processing Baseline)"
