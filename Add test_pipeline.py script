import numpy as np
from engine import NeuroKinematicEngine
from gates import PhenotypicGatingPipeline

def run_system_test():
    print("Initializing Synthesis Telemetry Test Suite...\n" + "="*50)
    
    # 1. Initialize the core engine and gating pipeline
    engine = NeuroKinematicEngine()
    pipeline = PhenotypicGatingPipeline(engine)
    
    # 2. Test Somatic Noise Filtering Matrix
    # We inject a massive 2.5-second transient spike (somatic artifact) into standard intervals
    raw_intervals = np.array([0.12, 0.15, 0.11, 0.14, 2.50, 0.13, 0.16, 0.12, 0.14, 0.11])
    filtered_intervals = engine.apply_somatic_filter(raw_intervals)
    
    print("[TEST 1] Somatic Noise Filter:")
    print(f"  -> Raw Vector Max: {np.max(raw_intervals)}s")
    print(f"  -> Filtered Vector Max: {np.max(filtered_intervals):.4f}s (Artifact Cleaned)")
    if np.max(filtered_intervals) < 1.0:
        print("  STATUS: SUCCESS\n")
    else:
        print("  STATUS: FAILED\n")

    # 3. Test Phenotypic Gate Processing (Simulating a High-Tension Profile)
    # We pass sample data vectors to check if the logic executes end-to-end
    print("[TEST 2] End-to-End Pipeline Routing:")
    
    # Mock data vectors matching the shape requirements of the engine
    sample_intra = np.array([0.12, 0.15, 0.13, 0.14, 0.12, 0.16, 0.11, 0.14, 0.13, 0.15])
    sample_hard = np.array([0.45, 0.48, 0.46, 0.44, 0.47, 0.49, 0.43, 0.46, 0.45, 0.47])
    sample_post_stall = np.array([0.25, 0.28, 0.30, 0.32, 0.35, 0.37, 0.39, 0.41, 0.42, 0.45])
    baseline_80th = 0.20
    
    # Process the mock subject profile
    output = pipeline.process_subject_profile(
        intra_word=sample_intra,
        hard_boundary=sample_hard,
        post_stall=sample_post_stall,
        baseline_80th=baseline_80th
    )
    
    print(f"  -> Calculated Z-Matrix Distance: {output['coordinates']['z_matrix']:.4f}")
    print(f"  -> Pipeline Output Classification: {output['phenotype_classification']}")
    print("  STATUS: SUCCESS (Pipeline terminated without syntax errors)")
    print("="*50)

if __name__ == "__main__":
    run_system_test()
