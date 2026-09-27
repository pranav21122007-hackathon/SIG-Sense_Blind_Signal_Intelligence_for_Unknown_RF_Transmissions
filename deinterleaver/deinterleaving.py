import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


def gf2_rank(matrix: np.ndarray) -> int:
    """
    Computes matrix rank over GF(2) using Gaussian elimination implemented
    purely with NumPy bitwise operations.
    """
    A = (matrix.copy() & 1).astype(np.uint8)
    rows, cols = A.shape
    rank = 0
    current_row = 0

    for col in range(cols):
        if current_row >= rows:
            break

        # Search for pivot in column `col` at or below `current_row`
        pivot_idx = np.where(A[current_row:, col] == 1)[0]
        if len(pivot_idx) == 0:
            continue

        pivot_row = current_row + pivot_idx[0]

        # Swap rows to bring pivot to current_row
        if pivot_row != current_row:
            A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        # Eliminate all other 1s in this column using bitwise XOR
        rows_to_xor = np.where(A[:, col] == 1)[0]
        rows_to_xor = rows_to_xor[rows_to_xor != current_row]
        if len(rows_to_xor) > 0:
            A[rows_to_xor, :] ^= A[current_row, :]

        rank += 1
        current_row += 1

    return rank


def detect_block_interleaver(
    bitstream: np.ndarray,
    w_min: int = 2,
    w_max: int = 64,
    confidence_threshold: float = 0.20,
    plot: bool = False
) -> Tuple[Optional[int], float, Dict[str, Any]]:
    """
    Detects block interleaver width W by hypothesizing candidate width W,
    inverting the (M x W) column-by-column readout permutation, and evaluating
    GF(2) rank deficiency across the reconstructed codeword structures.
    """
    bits = np.asarray(bitstream, dtype=np.uint8).flatten() & 1
    total_bits = len(bits)

    candidate_w = []
    raw_ranks = []
    normalized_deficiencies = []

    for w in range(w_min, w_max + 1):
        m = total_bits // w
        if m < 4:
            continue

        valid_len = m * w
        # Invert column-readout to reconstruct the original M x W block matrix
        candidate_matrix = bits[:valid_len].reshape((w, m)).T  # Shape: (M, W)
        candidate_restored = candidate_matrix.flatten()

        # Test rank over common block/codeword divisors (e.g., 4, 7, 8, 15, 16)
        # to detect the parity space alignment of the underlying linear code
        divisors_to_test = [d for d in [4, 7, 8, 12, 15, 16] if d <= candidate_matrix.shape[1]]
        if not divisors_to_test:
            divisors_to_test = [min(8, candidate_matrix.shape[1])]

        max_def_for_w = 0.0
        min_rank_for_w = candidate_matrix.shape[1]

        for n_cand in divisors_to_test:
            num_words = len(candidate_restored) // n_cand
            if num_words < n_cand:
                continue
            test_mat = candidate_restored[: num_words * n_cand].reshape((num_words, n_cand))
            r = gf2_rank(test_mat)
            full_rank = min(num_words, n_cand)
            deficiency = (full_rank - r) / float(full_rank)

            if deficiency > max_def_for_w:
                max_def_for_w = deficiency
                min_rank_for_w = r

        candidate_w.append(w)
        raw_ranks.append(min_rank_for_w)
        normalized_deficiencies.append(max_def_for_w)

    if not candidate_w:
        return None, 0.0, {"status": "bitstream too short for candidate range"}

    candidate_w = np.array(candidate_w)
    deficiencies = np.array(normalized_deficiencies)
    raw_ranks = np.array(raw_ranks)

    best_idx = int(np.argmax(deficiencies))
    best_w = int(candidate_w[best_idx])
    max_deficiency = float(deficiencies[best_idx])

    # Confidence: contrast the maximum rank dip against the average background baseline
    other_deficiencies = np.delete(deficiencies, best_idx)
    mean_baseline = float(np.mean(other_deficiencies)) if len(other_deficiencies) > 0 else 0.0
    confidence = float(np.clip(max_deficiency - mean_baseline, 0.0, 1.0))

    is_detected = bool(confidence >= confidence_threshold and max_deficiency > 0.10)

    stats = {
        "candidate_w": candidate_w.tolist(),
        "raw_ranks": raw_ranks.tolist(),
        "deficiencies": deficiencies.tolist(),
        "best_w": best_w if is_detected else None,
        "confidence": confidence,
        "detected": is_detected
    }

    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(candidate_w, raw_ranks, marker='o', markersize=3)
        plt.title("GF(2) Matrix Rank vs. Candidate Width W")
        plt.xlabel("Candidate Width W")
        plt.ylabel("GF(2) Rank")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(candidate_w, deficiencies, marker='s', color='darkred', markersize=3)
        plt.axhline(confidence_threshold, color='orange', linestyle=':', label='Threshold')
        plt.title("Rank Deficiency Ratio")
        plt.xlabel("Candidate Width W")
        plt.ylabel("Deficiency ((W - Rank) / W)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    if is_detected:
        return best_w, confidence, stats
    return None, confidence, stats


def deinterleave_block(bitstream: np.ndarray, w: Optional[int]) -> np.ndarray:
    """
    De-interleaves a standard block interleaver (written row-by-row, read column-by-column).
    Fallback: If w is None or invalid, passes the bitstream through unchanged.
    """
    bits = np.asarray(bitstream, dtype=np.uint8).flatten()
    if w is None or w <= 1:
        return bits

    total_bits = len(bits)
    m = total_bits // w
    valid_len = m * w

    if valid_len == 0:
        return bits

    # Reshape column-stream to (W, M) and transpose to restore original (M, W)
    interleaved_block = bits[:valid_len].reshape((w, m)).T
    deinterleaved = interleaved_block.flatten()

    # Pass any remaining unaligned tail bits directly
    if total_bits > valid_len:
        deinterleaved = np.concatenate([deinterleaved, bits[valid_len:]])

    return deinterleaved


def estimate_convolutional_interleaver(
    bitstream: np.ndarray,
    max_delay: int = 256,
    peak_threshold: float = 5.0,
    min_corr_amplitude: float = 0.08
) -> Tuple[Optional[int], float, np.ndarray]:
    """
    Autocorrelation-based estimator for convolutional (diagonal / Ramsey / Forney) interleavers[cite: 1].
    Detects periodic delays introduced by shift-register branch latencies[cite: 1].
    """
    bits = np.asarray(bitstream, dtype=np.float32).flatten()
    symbols = 1.0 - 2.0 * bits  # Map {0, 1} -> BPSK {-1.0, +1.0}
    n = len(symbols)

    if n <= max_delay:
        max_delay = max(2, n // 2)

    lags = np.arange(1, max_delay + 1)
    autocorr = np.zeros(len(lags))

    var = np.var(symbols)
    if var < 1e-9:
        return None, 0.0, autocorr

    mean = np.mean(symbols)
    centered = symbols - mean

    for idx, lag in enumerate(lags):
        c = np.mean(centered[: n - lag] * centered[lag:])
        autocorr[idx] = c / var

    abs_corr = np.abs(autocorr)
    med = np.median(abs_corr)
    mad = np.median(np.abs(abs_corr - med)) + 1e-6
    z_scores = 0.6745 * (abs_corr - med) / mad

    max_idx = int(np.argmax(z_scores))
    best_lag = int(lags[max_idx])
    max_z = float(z_scores[max_idx])
    max_peak_val = float(abs_corr[max_idx])

    # Require BOTH a sharp outlier (z-score) AND non-trivial correlation amplitude
    if max_z >= peak_threshold and max_peak_val >= min_corr_amplitude:
        return best_lag, max_z, autocorr
    return None, max_z, autocorr

# =====================================================================
# Pipeline Ingestion Adapter (Reads from Stage 3 or Stage 0)
# =====================================================================
def run_stage_4(input_source, w_min: int = 2, w_max: int = 64, plot: bool = False) -> Dict[str, Any]:
    """
    Universal Stage 4 runner[cite: 1].
    Accepts either:
      - A 1D NumPy array of bits directly in memory (from Stage 3)[cite: 1]
      - A string file path pointing to a binary dump file (.bin / .dat from Stage 0/3)[cite: 1]
    """
    if isinstance(input_source, str):
        # Read from file written by GNU Radio File Sink[cite: 1]
        bits = np.fromfile(input_source, dtype=np.uint8) & 1
    else:
        bits = np.asarray(input_source, dtype=np.uint8).flatten() & 1

    # 1. Block Interleaver Detection[cite: 1]
    w_est, confidence, block_stats = detect_block_interleaver(
        bits, w_min=w_min, w_max=w_max, plot=plot
    )

    # 2. Convolutional Interleaver Detection Mode[cite: 1]
    conv_lag, conv_z, _ = estimate_convolutional_interleaver(bits)

    # 3. De-interleave or Fallback[cite: 1]
    if w_est is not None:
        recovered_bits = deinterleave_block(bits, w_est)
        interleaver_type = "block"
    elif conv_lag is not None:
        recovered_bits = bits  # Fallback for block pipeline
        interleaver_type = "convolutional"
    else:
        recovered_bits = bits  # Fallback: pass-through unchanged[cite: 1]
        interleaver_type = "none"

    return {
        "interleaver_type": interleaver_type,
        "detected_width": w_est,
        "confidence": confidence,
        "conv_lag": conv_lag,
        "conv_z_score": conv_z,
        "output_bits": recovered_bits,
        "stats": block_stats
    }

# Deinterleaver/deinterleaver.py

# In run_stage_4, modify the return block:
def run_stage_4_adapted(input_source, w_min: int = 2, w_max: int = 64):
    res = run_stage_4(input_source, w_min=w_min, w_max=w_max, plot=False)
    
    # Transform arrays to [{width, rank}] structure expected by the frontend
    stats = res.get("stats", {})
    candidate_w = stats.get("candidate_w", [])
    raw_ranks = stats.get("raw_ranks", [])
    
    rank_profile = [
        {"width": int(w), "rank": float(r)}
        for w, r in zip(candidate_w, raw_ranks)
    ]
    
    return {
        "interleaver_type": res["interleaver_type"],
        "detected_width": res["detected_width"] if res["detected_width"] is not None else 0,
        "confidence": round(float(res["confidence"]), 4),
        "rank_profile": rank_profile,
        "output_bits": res["output_bits"]
    }

if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 4 PRE-COMMIT VERIFICATION SUITE")
    print("=" * 60)

    # -------------------------------------------------------------
    # TEST CASE 1: Standard Block Interleaver (True W = 16)
    # -------------------------------------------------------------
    np.random.seed(42)
    # Systematic (8, 4) generator: parity dependencies force rank deficiency
    generator = np.array([
        [1, 0, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ], dtype=np.uint8)

    num_codewords = 2400
    info_bits = np.random.randint(0, 2, size=(num_codewords, 4))
    codewords = (info_bits @ generator) % 2
    raw_stream = codewords.flatten()

    W_true = 16
    M_true = len(raw_stream) // W_true
    # Write row-by-row, transmit column-by-column
    interleaved_stream = raw_stream[: M_true * W_true].reshape((M_true, W_true)).T.flatten()

    res1 = run_stage_4(interleaved_stream, w_min=2, w_max=32, plot=False)
    mismatches1 = np.sum(res1["output_bits"] != raw_stream[:len(res1["output_bits"])])

    print("\n[TEST 1] Block Interleaved Input (W = 16):")
    print(f"  Input size:               {len(interleaved_stream)} bits")
    print(f"  Detected Interleaver:     {res1['interleaver_type']}")
    print(f"  Detected Width W:         {res1['detected_width']}")
    print(f"  Confidence Score:         {res1['confidence']:.2f}")
    print(f"  Bit reconstruction error: {mismatches1} bits")
    assert res1["detected_width"] == 16, "TEST 1 FAILED: Incorrect width detected"
    assert mismatches1 == 0, "TEST 1 FAILED: De-interleaver corrupted the bits"
    print("  --> PASS: Block interleaver correctly identified and inverted.")

    # -------------------------------------------------------------
    # TEST CASE 2: Fallback / Uninterleaved Raw Stream
    # -------------------------------------------------------------
    random_stream = np.random.randint(0, 2, size=10000, dtype=np.uint8)

    res2 = run_stage_4(random_stream, w_min=2, w_max=32, plot=False)
    mismatches2 = np.sum(res2["output_bits"] != random_stream)

    print("\n[TEST 2] Fallback Check (Uninterleaved Random Stream):")
    print(f"  Input size:               {len(random_stream)} bits")
    print(f"  Detected Interleaver:     {res2['interleaver_type']}")
    print(f"  Detected Width W:         {res2['detected_width']}")
    print(f"  Pass-through mismatches:  {mismatches2} bits")
    assert res2["detected_width"] is None, "TEST 2 FAILED: False positive detected"
    assert res2["interleaver_type"] == "none", "TEST 2 FAILED: Expected 'none'"
    assert mismatches2 == 0, "TEST 2 FAILED: Bitstream altered during fallback pass-through"
    print("  --> PASS: Graceful fallback confirmed, stream passed through unchanged.")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED — READY TO COMMIT")
    print("=" * 60)