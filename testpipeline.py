import numpy as np
from engine import TrueSequenceEngine
from gates import PhenotypicGatingPipeline

def run_system_test():
    print("Initializing Synthesis Telemetry Real-Time Test Suite...\n" + "="*60)
    
    # 1. Initialize the updated streaming architecture
    # Calibrate with a mock user baseline 80th-percentile recovery latency of 0.20s
    engine = TrueSequenceEngine(baseline_rec_80th=0.20)
    pipeline = PhenotypicGatingPipeline(engine)
    
    # 2. Simulate a live, stream-ingested typing session
    # We construct a time-series sequence that simulates:
    # - Stable baseline typing (0.12s - 0.15s)
    # - An isolated somatic noise artifact (a massive 2.50s muscle glitch)
    # - Standard lexical boundary events (spaces/punctuation)
    # - A sudden threshold-driven 25.77% processing latency surge (Complexity Stall Trigger)
    # - A 10-character post-stall recovery horizon
    
    print("[TEST 1] Ingesting Live Streaming Data & Somatic Filtering:")
    
    # Base stream components
    nominal_stream = [0.12, 0.14, 0.13, 0.15, 0.11, 0.14, 0.12, 0.13, 0.15, 0.14]
    somatic_artifact = 2.50
    stall_trigger = 0.45  # Massive surge compared to the 0.13s running median
    recovery_horizon = [0.42, 0.39, 0.35, 0.31, 0.28, 0.26, 0.24, 0.22, 0.21, 0.20]
    
    print(f"  -> Injecting somatic artifact noise: {somatic_artifact}s")
    print(f"  -> Simulating 25.77% network processing surge (Stall Trigger): {stall_trigger}s")
    
    # Stream execution sequence
    execution_stream = nominal_stream + [somatic_artifact] + nominal_stream + [stall_trigger] + recovery_horizon
    
    print(f"  -> Total stream length: {len(execution_stream)} kinetic intervals.")
    print("  -> Executing real-time routing logic...")
    
    final_output = None
    for i, interval in enumerate(execution_stream):
        # Alternate lexical boundaries to simulate word structures
        is_boundary = True if i % 5 == 0 else False
        
        # Route the interval sequentially through the pipeline
        output = pipeline.process_live_stream_event(interval, is_space_or_punctuation=is_boundary)
        
        # Capture the engine's output dictionary once the 10-step post-stall horizon finishes
        if output is not None:
            final_output = output

    # 3. Evaluate State-Machine Termination & Routing Results
    print("\n[TEST 2] State-Machine Termination & Phenotypic Gating:")
    if final_output is None:
        print("  STATUS: FAILED (The engine's post-stall horizon state machine did not terminate.)")
        print("="*60)
        return
        
    print("  STATUS: SUCCESS (Real-time horizon completed and processed without syntax errors)")
    print(f"  -> Extracted LBIR (Boundary Insulation): {final_output['metrics']['lbir']:.4f}")
    
    # Correcting LaTeX formatting to use markdown for units
    print(f"  -> Extracted Lambda (Recovery Velocity): {final_output['metrics']['lambda_coef']:.4f}")
    print(f"  -> Calculated Z-Matrix Distance Vector: {final_output['coordinates']['z_matrix']:.4f}")
    print(f"  -> Pipeline Output Classification: {final_output['phenotype_classification']}")
    print("="*60)

if __name__ == "__main__":
    run_system_test()
