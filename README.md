# SIG-Sense
### Blind Signal Intelligence for Universal RF Transmissions
 
**Smart India Hackathon 2026 — Problem Statement SIH26147**
*Automated model for analysis of .IQ and .wav files along with signal parameter extraction*
Sponsored by **NTRO** · Domain: **Space Technology**
 
---
 
## Overview
 
SIG-Sense takes a raw, unlabeled `.IQ` or `.wav` capture — with **no prior knowledge of the transmitter** — and automatically reconstructs it end-to-end: signal parameters, modulation, interleaving, forward error correction, frame structure, and payload, surfaced through a single analysis dashboard.
 
Most SDR tooling assumes you already know the modulation scheme, symbol rate, interleaver width, or FEC code. SIG-Sense assumes none of that. Every stage is a **blind estimator** — it infers the parameters from the signal itself before decoding it.
 
## Key Features
 
- **Zero-Prior Blind Reconstruction** — analyzes raw IQ/WAV data with no transmitter metadata
- **GF(2) Matrix-Rank De-Interleaving** — automatically detects interleaver width via GF(2) rank analysis, no external library
- **Hybrid Modulation Classification** — deep-learning (ONNX/ResNet1D) + explainable classical cumulants (C40/C42/C63), cross-checked
- **Coupled Preamble & Frame Validation** — combines correlation and timing for reliable frame sync
- **Entropy-Based Unknown-Protocol Fallback** — Shannon entropy segmentation when a protocol doesn't match known framing
- **Integrated RF-to-Hex Analysis GUI** — waterfall, constellation, decode status, and hex inspection in one interface
## Pipeline
 
| # | Stage | What it does | Core technique |
|---|-------|---------------|-----------------|
| 1 | **Universal Data Ingestion** | Standardizes any IQ/WAV capture into one processable stream | Format-agnostic parser |
| 2 | **Blind Signal Parameter Detection** | Detects frequency, bandwidth, modulation class, timing | Welch PSD, ONNX/ResNet1D, cumulants |
| 3 | **GF(2) Blind De-Interleaving** | Recovers original bit order without knowing interleaver depth | GF(2) matrix-rank solver (Gaussian elimination) |
| 4 | **Blind FEC Detection & Decoding** | Identifies the error-correction scheme and recovers corrupted bits | Reed-Solomon, convolutional/Viterbi |
| 5 | **Intelligent Frame Detection** | Locates frame boundaries, even for unknown/custom protocols | Autocorrelation + entropy profiling |
| 6 | **Unified Analysis GUI** | Presents the full result set to the operator | React + D3 dashboard |
 
## Tech Stack
 
**Backend / DSP core**
- Python 3.x, NumPy, SciPy
**Modulation classification**
- ONNX Runtime (CPU inference) + custom ResNet1D classifier
- Custom cumulants engine (C40, C42, C63) — explainable, no training required
**Synchronization**
- GNU Radio — Gardner timing recovery, Costas carrier recovery
**De-interleaving**
- Custom in-house GF(2) matrix-rank solver (Gaussian elimination over the binary field)
**FEC identification & decoding**
- `reedsolo` (Reed-Solomon), `commpy` (convolutional coding, Trellis, Viterbi)
**Frame mining**
- NumPy-based autocorrelation for frame-length and preamble detection
**Frontend / dashboard**
- React 19, Vite, Recharts, D3, Redux
**Dev / tooling**
- Google Colab (model training), Python `venv`, JSON-schema contracts between stages
## Repository Structure
 
```
.
├── Ingestion/        # Stage 1 — universal IQ/WAV ingestion
├── Classifier/        # Stage 2 — blind parameter & modulation classification
├── Sync_gnuradio/     # Synchronization — Gardner/Costas GNU Radio blocks
├── Deinterleaver/      # Stage 3 — GF(2) matrix-rank de-interleaving
├── Fec/               # Stage 4 — blind FEC identification & decoding
├── Framing/           # Stage 5 — frame sync, preamble & hex profiling
├── Frontend/          # Stage 6 — unified analysis dashboard (React)
├── Testdata/          # Ground-truth test signals (GNU Radio Companion configs)
├── contracts/         # JSON-schema data contracts between pipeline stages
└── README.md
```
 
