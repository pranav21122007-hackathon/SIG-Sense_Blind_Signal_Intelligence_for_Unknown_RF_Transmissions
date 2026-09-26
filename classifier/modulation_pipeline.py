import os
from pathlib import Path
import numpy as np
import onnxruntime as ort

TARGET_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '2FSK', '4FSK']

# Theoretical cumulants (C40, C42, C63) for unit-power normalized constellations
THEORETICAL_CUMULANTS = {
    'BPSK':  np.array([-2.000, -2.000, 16.000]),
    'QPSK':  np.array([ 1.000, -1.000,  4.000]),
    '8PSK':  np.array([ 0.000, -1.000,  4.000]),
    '16QAM': np.array([-0.680, -0.680,  2.080]),
    '64QAM': np.array([-0.619, -0.619,  1.797]),
    '2FSK':  np.array([ 0.000, -1.000,  2.000]),
    '4FSK':  np.array([ 0.000, -1.000,  1.000])
}

# Resolve absolute path to the ONNX model in the same folder as this script
SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = str(SCRIPT_DIR / "mod_resnet1d.onnx")

print(f"[*] Looking for ONNX model at:\n    {ONNX_PATH}")

ort_session = None
if not os.path.exists(ONNX_PATH):
    print(f"[ERROR] File does not exist at {ONNX_PATH}!")
    print(f"Files found in {SCRIPT_DIR}:")
    for f in os.listdir(SCRIPT_DIR):
        print(f"  - {f}")
else:
    try:
        # Load ONNX model directly with CPU Execution Provider
        ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        print("[SUCCESS] 'mod_resnet1d.onnx' loaded successfully!")
    except Exception as err:
        print(f"[ERROR] onnxruntime failed to initialize session:")
        print(f"        {type(err).__name__}: {err}")
# ============================================================================
# TIER 1: HIGHER-ORDER CUMULANTS ENGINE
# ============================================================================

# Classifier/classifier.py
# Classifier/classifier.py

def compute_all_cumulants(iq_complex: np.ndarray):
    """Computes normalized cumulants (C40, C42, C63) for Tier 1 constellation matching."""
    centered = iq_complex - np.mean(iq_complex)
    power = np.mean(np.abs(centered)**2)
    y = centered / np.sqrt(power) if power > 1e-12 else centered
    
    mu20 = np.mean(y**2)
    mu21 = np.mean(np.abs(y)**2)
    mu40 = np.mean(y**4)
    mu42 = np.mean((np.abs(y)**2) * (y**2))
    mu63 = np.mean(np.abs(y)**6)
    
    c40 = mu40 - 3.0 * (mu20**2)
    c42 = np.real(np.mean(np.abs(y)**4)) - np.abs(mu20)**2 - 2.0 * (mu21**2)
    c63 = mu63 - 9.0 * c42 * mu21 - (np.abs(mu20)**2) * mu21 - 6.0 * (mu21**3)
    
    return float(np.real(c40)), float(c42), float(np.real(c63))

def run_stage_2(iq_complex: np.ndarray, snr_est: float = 12.0) -> dict:
    """
    Standardized execution interface for Stage 2.
    """
    # Format (2, N) for Tier-2 CNN
    iq_samples_2d = np.stack([iq_complex.real, iq_complex.imag]).astype(np.float32)
    
    # Run two-tier classifier
    classification_res = classify(iq_samples_2d, snr_est=snr_est)
    
    # Compute all frontend explainability metrics
    cumulants_map = compute_all_cumulants(iq_complex)
    
    return {
        "class": classification_res["predicted_class"],
        "confidence": classification_res["confidence"],
        "tier_used": classification_res.get("tier_used", 1),
        "cumulants": cumulants_map,
        "probabilities": classification_res.get("reasoning", {}).get("softmax_scores", {})
    }
def tier1_classify(iq_complex: np.ndarray):
    """
    Evaluates observed cumulants against theoretical constellation baselines.
    """
    c40, c42, c63 = compute_all_cumulants(iq_complex)
    observed = np.array([c40, c42, c63])

    # Weights: Give higher priority to robust 4th-order moments (C40, C42)
    # and downscale noisy 6th-order estimates (C63) on short sequence lengths
    weights = np.array([1.5, 1.5, 0.02])

    distances = {}
    min_dist = float('inf')
    best_mod = None

    for mod, theoretical in THEORETICAL_CUMULANTS.items():
        dist = float(np.sqrt(np.sum(weights * ((observed - theoretical) ** 2))))
        distances[mod] = dist
        if dist < min_dist:
            min_dist = dist
            best_mod = mod

    # Soft Gaussian confidence mapping with scaling parameter sigma=1.8
    confidence = float(np.exp(-min_dist / 1.8))

    return {
        "predicted_class": best_mod,
        "confidence": confidence,
        "cumulants": {"C40": round(c40, 4), "C42": round(c42, 4), "C63": round(c63, 4)},
        "distance": round(min_dist, 4)
    }

# ============================================================================
# TIER 2: RESIDUAL 1D-CNN ONNX RUNNER
# ============================================================================

def tier2_classify(iq_samples: np.ndarray):
    """
    Runs ONNX inference over the (2, N) input array.
    """
    if ort_session is None:
        raise RuntimeError(f"ONNX session is not active. Check error messages above for path: {ONNX_PATH}")

    # Ensure shape is (1, 2, 1024)
    if iq_samples.ndim == 2:
        inp = np.expand_dims(iq_samples.astype(np.float32), axis=0)
    else:
        inp = iq_samples.astype(np.float32)

    seq_len = inp.shape[2]
    if seq_len != 1024:
        if seq_len > 1024:
            inp = inp[:, :, :1024]
        else:
            inp = np.pad(inp, ((0, 0), (0, 0), (0, 1024 - seq_len)))

    outputs = ort_session.run(None, {'iq_input': inp})[0]
    logits = outputs[0]

    # Numerically stable Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": TARGET_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {TARGET_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(TARGET_CLASSES))}
    }


# ============================================================================
# UNIFIED ROUTER (TIER 1 + TIER 2)
# ============================================================================

def classify(iq_samples: np.ndarray, snr_est: float, confidence_threshold: float = 0.65) -> dict:
    """
    Two-Tier Automatic Modulation Classifier with full UI explainability metrics.
    
    Args:
        iq_samples: ndarray of shape (2, N) where row 0 is I, row 1 is Q.
        snr_est: Estimated or known SNR in dB.
        confidence_threshold: Minimum Tier 1 confidence required to bypass Tier 2.
    """
    # Create 1D complex representation for Tier 1
    iq_complex = iq_samples[0, :] + 1j * iq_samples[1, :]

    # Route decision: Check SNR boundary
    if snr_est > 2.0:
        tier1_out = tier1_classify(iq_complex)
        # If Tier 1 confidence passes threshold, return directly
        if tier1_out["confidence"] >= confidence_threshold:
            return {
                "tier_used": 1,
                "predicted_class": tier1_out["predicted_class"],
                "confidence": round(tier1_out["confidence"], 4),
                "reasoning": {
                    "method": "Theoretical Cumulant Distance Matching",
                    "c40": tier1_out["cumulants"]["C40"],
                    "c42": tier1_out["cumulants"]["C42"],
                    "c63": tier1_out["cumulants"]["C63"],
                    "prototype_distance": tier1_out["distance"]
                }
            }
        else:
            tier1_fallback = tier1_out
    else:
        tier1_fallback = None

    # Fallback to Tier 2 (Low SNR or Low Tier 1 Confidence)
    tier2_out = tier2_classify(iq_samples)
    reasoning = {
        "method": "Residual 1D-CNN (ONNX)",
        "softmax_scores": tier2_out["probabilities"]
    }
    if tier1_fallback:
        reasoning["tier1_rejected_features"] = tier1_fallback["cumulants"]
        reasoning["tier1_rejection_reason"] = f"Confidence {tier1_fallback['confidence']:.2f} < threshold {confidence_threshold}"

    return {
        "tier_used": 2,
        "predicted_class": tier2_out["predicted_class"],
        "confidence": round(tier2_out["confidence"], 4),
        "reasoning": reasoning
    }


# ============================================================================
# VERIFICATION TEST
# ============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Realistic BPSK with low noise (High SNR: ~15 dB)
    # Unit-power BPSK symbols mapped to real axis with small complex AWGN
    bpsk_syms = np.random.choice([-1.0, 1.0], size=1024)
    noise_i = np.random.randn(1024) * 0.05
    noise_q = np.random.randn(1024) * 0.05
    dummy_high_snr = np.stack([bpsk_syms + noise_i, noise_q]).astype(np.float32)

    print("\n--- High SNR Test (Expected Tier 1) ---")
    result_high = classify(dummy_high_snr, snr_est=15.0)
    print(result_high)

    # 2. Low SNR Test (Expected Tier 2)
    dummy_low_snr = np.random.randn(2, 1024).astype(np.float32)

    print("\n--- Low SNR Test (Expected Tier 2) ---")
    result_low = classify(dummy_low_snr, snr_est=-2.0)
    print(result_low)