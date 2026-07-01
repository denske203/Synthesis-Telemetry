# Neuro-Kinematic Extraction Engine (NKEE)

## 1. Overview
The Neuro-Kinematic Extraction Engine (NKEE) is an open-source, high-resolution computational framework engineered to track global neuronal workspace integrity and transcallosal information dynamics via passive, micro-temporal alphanumeric typing telemetry. Moving beyond legacy cybersecurity configurations that treat typing rhythms as static cryptographic identity profiles, this pipeline re-operationalizes manual text input as a high-frequency projection of lateralized generative models executing cross-midline motor handshakes.

The runtime architecture captures asynchronous key-action events at a strict 1-millisecond resolution under high-entropy, real-world conditions. By evaluating internal mathematical ratios rather than raw temporal values, the engine systematically neutralizes mechanical hardware and device-specific latency variations, isolating authentic cortical transcallosal coordination from somatic and mechanical noise.

## 2. Theoretical Foundations & Processing Mechanics

### 2.1 The Universal Complexity Stall
The global processing latency ($\tau$) of the coupled interhemispheric pipeline is a function of incoming task entropy ($H$) relative to the system's realized horizontal Coupling Coefficient ($C$):

$$\tau = \frac{H}{C}$$

The **Complexity Stall** is operationalized as a universal, systematic, time-dependent disruption of the serial processing loop occurring when an engine hits its physical parsing limit under load. While the ceiling height (the point of failure) varies dynamically across individuals based on genetic factors, environmental load, and baseline translation tax ($T_{\text{tax}}$), the resulting phase transition enforces a conserved biophysical signature: a distinct **25.77% baseline latency surge** relative to the system's steady state. This surge quantifies the absolute topological distance and translation cost between the executive canopy (the Manager network) and the underlying procedural substrate (the Base LDH) during an uncomputable processing crisis.

### 2.2 Core Kinematic Metrics
The engine converts primitive asynchronous keystroke timestamps—Key-Down ($D_i$), Key-Up ($U_i$), and Character Token ($K_i$)—into scale-invariant coordinate mechanics across two primary axes:

*   **Lexical Boundary Invariance Ratio (LBIR):** Quantifies the variance of standardized boundary deviations across sentence or word limits. It isolates the structural permeability of processing boundaries, tracking whether a system executes state-dependent temporal shifts or enforces flatline, rigid serialization to insulate a vulnerable active register from external variance.
*   **Traction Recovery Coefficient ($\Lambda$ / $\gamma$):** Measures the log-decay velocity curve of the top-down active inference loop across a 10-step motor horizon following a Complexity Stall, tracking the efficiency with which the system clears its active registers and recovers baseline velocity.

## 3. Data Ingestion Standards & Artifact Filtering
To preserve mathematical validity and eliminate phase-orthogonality noise transients, the data ingestion pipeline is explicitly constrained from processing raw, un-enveloped continuous sinusoidal voltage time-series. The engine is strictly optimized for:
1. **Discrete Tokenization Sequences:** Alphanumeric timing intervals and keystroke dynamics.
2. **Macroscopic Amplitude Envelopes:** Long-range differential telemetry vectors pre-transformed via Hilbert amplitude extraction.

### Global Run Configuration
The signal layer is rigorously gated via an automated noise-filtering matrix to ensure data integrity against physiological or mechanical anomalies (e.g., EMG jaw-clench artifacts):
*   **Statistical Filter Axis:** Mean + 1.5 * StdDev (Individual Dynamic Gate applied to intervals).
*   **Critical Event Horizon:** $\ge 5.0$ Continuous Seconds required to classify a macro-level attractor shift, programmatically masking transient initialization transients.

## 4. Pipeline Architecture & Phenotypic Filtration Gates
The joint space synthesis ($Z_{\text{matrix}}$) converts LBIR and $\Lambda$ into standardized coordinates relative to a fixed, fully cleaned population floor baseline ($N=3,153$ independent human subjects operating under allostatic load):
*   **Fixed Center $\mu_{\text{LBIR}}$:** 3.7928
*   **Fixed Center $\mu_{\Lambda}$:** -0.9609

The absolute topological drift away from the neurotypical population floor is resolved as a joint Euclidean vector distance ($Z_{\text{matrix}} = \sqrt{Z_{\text{LBIR}}^2 + Z_{\Lambda}^2}$). The pipeline implements asymmetric gating thresholds to isolate stable, alternative network geometries out-of-sample:

*   **Quadrant II (Borderline Volatile Manifold):** $Z_{\text{matrix}} \ge 2.0$, $Z_{\text{LBIR}} < 0.0$, $Z_{\Lambda} < 0.0$ (Captures a compressed pacing floor paired with explosive, high-velocity error-correction snap-backs).
*   **Quadrant IV (Narcissistic Static Stance):** $Z_{\text{matrix}} \ge 1.5$, $Z_{\text{LBIR}} < 0.0$, $Z_{\Lambda} \ge 0.0$ (Captures ironclad structural boundaries paired with total relational feedback attenuation and persistent motor stagnation).
*   **Obsessive-Compulsive Attractor State:** Localized via a tightening four-tier joint probability grid, tracking the locking of motor execution into high-precision, hyper-isolated recycling loops ($\Pi_{\text{err}} \rightarrow \infty$).

## 5. Usage & Implementation
The core processing modules are exposed via `NeuroKinematicEngine` in `engine.py`. 

```python
from engine import NeuroKinematicEngine

# Initialize the mathematical extraction engine
engine = NeuroKinematicEngine()

# Apply individual dynamic gate to strip somatic artifact noise
clean_data = engine.apply_somatic_filter(raw_intervals)

# Extract scale-invariant coordinates
lbir_score = engine.extract_lbir(intra_word_intervals, hard_boundary_intervals)
lambda_score = engine.extract_traction_recovery(post_stall_horizon, baseline_80th)

# Evaluate absolute topological drift vector distance
z_matrix, z_lbir, z_lambda = engine.evaluate_topology_drift(lbir_score, lambda_score)
