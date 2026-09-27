import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# =====================================================================
# 1. Autocorrelation-Based Frame Length Detection
# =====================================================================
def detect_frame_length(
    bitstream: np.ndarray,
    min_frame_len: int = 64,
    max_frame_len: int = 4096,
    threshold_z: float = 3.0
) -> Tuple[Optional[int], float, np.ndarray]:
    """
    Computes normalized circular autocorrelation across candidate lags to detect
    the fundamental repeating frame period (distance between periodic preambles).
    """
    bits = np.asarray(bitstream, dtype=np.float32).flatten()
    n = len(bits)
    if n < min_frame_len * 2:
        return None, 0.0, np.array([])

    max_frame_len = min(max_frame_len, n // 3)
    if max_frame_len <= min_frame_len:
        return None, 0.0, np.array([])

    symbols = 1.0 - 2.0 * bits
    symbols -= np.mean(symbols)
    var = np.var(symbols)
    if var < 1e-9:
        return None, 0.0, np.array([])

    # Fast autocorrelation via FFT
    n_fft = 1 << (2 * n - 1).bit_length()
    f_sym = np.fft.rfft(symbols, n=n_fft)
    autocorr_raw = np.fft.irfft(f_sym * np.conj(f_sym), n=n_fft)[:n]
    norm_factors = np.arange(n, 0, -1, dtype=np.float32)
    autocorr = (autocorr_raw / norm_factors) / var

    candidate_lags = np.arange(min_frame_len, max_frame_len + 1)
    corr_slice = autocorr[candidate_lags]

    median_val = np.median(corr_slice)
    mad = np.median(np.abs(corr_slice - median_val)) + 1e-6
    z_scores = 0.6745 * (corr_slice - median_val) / mad

    peak_indices = []
    for i in range(1, len(z_scores) - 1):
        if z_scores[i] > z_scores[i - 1] and z_scores[i] > z_scores[i + 1]:
            if z_scores[i] >= threshold_z:
                peak_indices.append(i)

    if not peak_indices:
        return None, float(np.max(z_scores)), autocorr[:max_frame_len + 1]

    # Subharmonic check: favor the fundamental period
    best_idx = peak_indices[int(np.argmax([z_scores[p] for p in peak_indices]))]
    best_lag = int(candidate_lags[best_idx])
    max_z = float(z_scores[best_idx])

    for p in peak_indices:
        cand = int(candidate_lags[p])
        if best_lag > cand and best_lag % cand == 0 and z_scores[p] >= threshold_z * 0.8:
            return cand, float(z_scores[p]), autocorr[:max_frame_len + 1]

    return best_lag, max_z, autocorr[:max_frame_len + 1]


# =====================================================================
# 2. Extensible Preamble Scanner & Sync-Word Library
# =====================================================================
def hex_to_bits(hex_str: str, length: Optional[int] = None) -> np.ndarray:
    """Converts a hex pattern string to a 1D binary numpy array."""
    clean_hex = hex_str.lower().replace("0x", "")
    expected_bits = len(clean_hex) * 4 if length is None else length
    val = int(clean_hex, 16)
    bits = [(val >> (expected_bits - 1 - i)) & 1 for i in range(expected_bits)]
    return np.array(bits, dtype=np.uint8)


def correlate_dmrs_placeholder(bitstream: np.ndarray, reference_seq: Optional[np.ndarray] = None) -> List[int]:
    """Placeholder cross-correlation scanner for 5G NR / LTE DMRS sequences."""
    if reference_seq is None:
        return []
    ref = 1.0 - 2.0 * np.asarray(reference_seq, dtype=np.float32).flatten()
    stream = 1.0 - 2.0 * np.asarray(bitstream, dtype=np.float32).flatten()
    corr = np.correlate(stream, ref, mode="valid") / len(ref)
    hits = np.where(corr >= 0.90)[0]
    return hits.tolist()


KNOWN_PREAMBLES: List[Dict[str, Any]] = [
    {
        "name": "CCSDS Telemetry",
        "pattern": hex_to_bits("1ACFFC1D", length=32),
        "bit_length": 32,
        "type": "exact"
    },
    {
        "name": "AX.25 / HDLC Flag",
        "pattern": hex_to_bits("7E", length=8),
        "bit_length": 8,
        "type": "exact"
    },
    {
        "name": "Barker-11",
        "pattern": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
        "bit_length": 11,
        "type": "exact"
    },
    {
        "name": "5G NR / LTE DMRS",
        "correlation_fn": correlate_dmrs_placeholder,
        "bit_length": 32,
        "type": "dynamic"
    }
]


def scan_known_preambles(
    bitstream: np.ndarray,
    preamble_lib: List[Dict[str, Any]] = KNOWN_PREAMBLES,
    autocorr_hint: Optional[int] = None,
    max_matches_per_type: int = 200,
    min_frame_stride: int = 64
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Scans the bitstream for matches against the known preamble library.
    Requires periodic recurrence across the stream to confirm a preamble.
    """
    bits = np.asarray(bitstream, dtype=np.uint8).flatten() & 1
    total_bits = len(bits)
    all_candidate_matches: Dict[str, List[Dict[str, Any]]] = {}

    for entry in preamble_lib:
        name = entry["name"]
        match_type = entry.get("type", "exact")

        if match_type == "exact":
            pat = entry["pattern"]
            pat_len = len(pat)
            if pat_len > total_bits:
                continue

            b_stream = 1.0 - 2.0 * bits.astype(np.float32)
            b_pat = 1.0 - 2.0 * pat.astype(np.float32)
            corr = np.correlate(b_stream, b_pat, mode="valid")

            match_indices = np.where(corr >= (pat_len - 1e-3))[0]
            if len(match_indices) > 0:
                all_candidate_matches[name] = [
                    {
                        "name": name,
                        "bit_offset": int(idx),
                        "byte_offset": int(idx // 8),
                        "bit_length": pat_len,
                        "byte_length": math.ceil(pat_len / 8.0)
                    }
                    for idx in match_indices[:max_matches_per_type]
                ]

        elif match_type == "dynamic" and "correlation_fn" in entry:
            hits = entry["correlation_fn"](bits)
            if len(hits) > 0:
                all_candidate_matches[name] = [
                    {
                        "name": name,
                        "bit_offset": int(idx),
                        "byte_offset": int(idx // 8),
                        "bit_length": entry.get("bit_length", 32),
                        "byte_length": math.ceil(entry.get("bit_length", 32) / 8.0)
                    }
                    for idx in hits[:max_matches_per_type]
                ]

    if not all_candidate_matches:
        return [], None

    def spans_stream(matches_list: List[Dict[str, Any]]) -> bool:
        if len(matches_list) < 4:
            return False
        span = matches_list[-1]["bit_offset"] - matches_list[0]["bit_offset"]
        return span >= (total_bits * 0.40)

    # 1. Search for regular periodic grid recurrence (>= 4 frames, stride >= 64)
    best_name = None
    best_stride = None
    best_periodic_matches: List[Dict[str, Any]] = []
    max_periodic_hits = 0

    for name, matches in all_candidate_matches.items():
        if len(matches) < 4:
            continue

        offsets = np.array([m["bit_offset"] for m in matches])
        diffs = np.diff(offsets)
        unique_diffs, counts = np.unique(diffs, return_counts=True)

        for cand_stride, count in zip(unique_diffs, counts):
            if cand_stride < min_frame_stride:
                continue

            for start_idx in offsets[:len(offsets) - count]:
                periodic_subset = [
                    m for m in matches
                    if (m["bit_offset"] >= start_idx) and ((m["bit_offset"] - start_idx) % cand_stride == 0)
                ]
                if len(periodic_subset) > max_periodic_hits and spans_stream(periodic_subset):
                    max_periodic_hits = len(periodic_subset)
                    best_name = name
                    best_stride = int(cand_stride)
                    best_periodic_matches = periodic_subset

    if max_periodic_hits >= 4 and best_name is not None:
        best_periodic_matches.sort(key=lambda m: m["bit_offset"])
        return best_periodic_matches, best_stride

    # 2. Cross-reference with autocorrelation hint
    if autocorr_hint is not None and autocorr_hint >= min_frame_stride:
        for name, matches in all_candidate_matches.items():
            offsets = np.array([m["bit_offset"] for m in matches])
            for start_idx in offsets:
                subset = [
                    m for m in matches
                    if (m["bit_offset"] >= start_idx) and ((m["bit_offset"] - start_idx) % autocorr_hint == 0)
                ]
                if len(subset) > max_periodic_hits and spans_stream(subset):
                    max_periodic_hits = len(subset)
                    best_name = name
                    best_stride = autocorr_hint
                    best_periodic_matches = subset

        if max_periodic_hits >= 4 and best_name is not None:
            best_periodic_matches.sort(key=lambda m: m["bit_offset"])
            return best_periodic_matches, best_stride

    return [], None


# =====================================================================
# 3. Sliding-Window Normalized Shannon Entropy Fallback
# =====================================================================
def compute_entropy_profile(
    byte_array: np.ndarray,
    window_size: int = 16,
    step: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes normalized local Shannon entropy in range [0.0, 1.0].
    Normalized by log2(min(window_size, 256)) so window size doesn't artificially cap entropy.
    """
    data = np.asarray(byte_array, dtype=np.uint8)
    n = len(data)
    if n < window_size:
        return np.array([0]), np.array([0.0])

    offsets = np.arange(0, n - window_size + 1, step)
    entropies = np.zeros(len(offsets), dtype=np.float32)
    max_possible_entropy = math.log2(min(window_size, 256))

    for i, off in enumerate(offsets):
        window = data[off : off + window_size]
        _, counts = np.unique(window, return_counts=True)
        probs = counts / float(window_size)
        raw_ent = -np.sum(probs * np.log2(probs))
        entropies[i] = raw_ent / max_possible_entropy if max_possible_entropy > 0 else 0.0

    return offsets, entropies


# =====================================================================
# 4. Bitstream Packaging & Labeled Hex Viewer
# =====================================================================
def generate_annotated_hex_dump(
    bitstream: np.ndarray,
    preamble_matches: List[Dict[str, Any]],
    frame_len_bits: Optional[int] = None,
    bytes_per_row: int = 16
) -> Dict[str, Any]:
    """
    Converts bitstream into bytes and produces an annotated hex dump JSON
    with byte ranges classified into preamble, header, payload, and checksum.
    """
    bits = np.asarray(bitstream, dtype=np.uint8).flatten() & 1
    total_bits = len(bits)
    num_bytes = total_bits // 8

    packed_bytes = np.packbits(bits[: num_bytes * 8])
    labels = np.full(num_bytes, "payload", dtype=object)

    if len(preamble_matches) > 0:
        # PATH A: Framing preambles detected
        for match in preamble_matches:
            start_byte = match["byte_offset"]
            end_byte = min(num_bytes, start_byte + match["byte_length"])
            labels[start_byte:end_byte] = "preamble"

            # 4 bytes after preamble tagged as header
            hdr_end = min(num_bytes, end_byte + 4)
            labels[end_byte:hdr_end] = "header"

        # Mark checksums at frame boundaries
        if frame_len_bits is not None and frame_len_bits >= 64:
            frame_len_bytes = frame_len_bits // 8
            for frame_start in range(0, num_bytes, frame_len_bytes):
                frame_end = min(num_bytes, frame_start + frame_len_bytes)
                crc_start = max(frame_start, frame_end - 4)
                labels[crc_start:frame_end] = "checksum"

    else:
        # PATH B: Fallback - Segment using normalized local Shannon entropy
        if num_bytes >= 32:
            win_size = 16
            step_size = 4
            offsets, norm_ent_curve = compute_entropy_profile(
                packed_bytes, window_size=win_size, step=step_size
            )

            # High entropy (>= 0.70) = payload; Low entropy (< 0.70) = structured header
            for off, norm_ent in zip(offsets, norm_ent_curve):
                if norm_ent < 0.70:
                    labels[off : min(num_bytes, off + win_size)] = "header"

    # Build the structured hex editor rows
    hex_rows = []
    for row_start in range(0, num_bytes, bytes_per_row):
        row_end = min(num_bytes, row_start + bytes_per_row)
        chunk = packed_bytes[row_start:row_end]
        row_labels = labels[row_start:row_end]

        hex_tokens = [f"{b:02X}" for b in chunk]
        ascii_chars = [chr(b) if 32 <= b <= 126 else "." for b in chunk]

        hex_rows.append({
            "offset": f"0x{row_start:06X}",
            "hex": " ".join(hex_tokens),
            "ascii": "".join(ascii_chars),
            "regions": [
                {"byte_index": row_start + i, "label": row_labels[i], "hex": hex_tokens[i]}
                for i in range(len(chunk))
            ]
        })

    return {
        "total_bytes": int(num_bytes),
        "total_bits": int(total_bits),
        "hex_dump": hex_rows
    }


# =====================================================================
# Pipeline Stage 6 Runner
# =====================================================================
def run_stage_6(
    bitstream: np.ndarray,
    preamble_lib: List[Dict[str, Any]] = KNOWN_PREAMBLES
) -> Dict[str, Any]:
    """
    Master runner for Stage 6: Frame sync, preamble scanning, and hex profiling.
    """
    bits = np.asarray(bitstream, dtype=np.uint8).flatten() & 1

    # 1. Periodic Frame Length Autocorrelation
    detected_frame_len, frame_z, autocorr = detect_frame_length(bits)

    # 2. Known Preamble Library Scan
    preamble_hits, derived_frame_len = scan_known_preambles(
        bits,
        preamble_lib=preamble_lib,
        autocorr_hint=detected_frame_len
    )

    # Only treat frame length as valid if validated by preambles or high z-score
    if len(preamble_hits) > 0:
        final_frame_len = derived_frame_len if derived_frame_len is not None else detected_frame_len
    else:
        final_frame_len = None  # Unframed / fallback mode

    # 3. Labeled Hex Editor & Segment Profiler
    annotated_dump = generate_annotated_hex_dump(
        bits,
        preamble_matches=preamble_hits,
        frame_len_bits=final_frame_len,
        bytes_per_row=16
    )

    return {
        "frame_length_bits": final_frame_len,
        "frame_length_confidence_z": float(frame_z),
        "preamble_matches": preamble_hits,
        "preambles_found_count": len(preamble_hits),
        "hex_inspector": annotated_dump
    }
# Framemining/framemining.py

def run_stage_6_adapted(bitstream: np.ndarray) -> dict:
    """
    Standardized execution interface for Stage 6.
    """
    stage6_res = run_stage_6(bitstream)
    
    hex_inspector = stage6_res.get("hex_inspector", {})
    raw_rows = hex_inspector.get("hex_dump", [])
    
    flat_rows = []
    for r in raw_rows:
        # Determine dominant region label for the 16-byte chunk
        regions = [reg["label"] for reg in r.get("regions", [])]
        dominant_region = "payload"
        for label in ["preamble", "header", "checksum"]:
            if label in regions:
                dominant_region = label
                break
                
        flat_rows.append({
            "offset": int(r["offset"], 16) if isinstance(r["offset"], str) else r["offset"],
            "hex": r["hex"],
            "ascii": r["ascii"],
            "region": dominant_region
        })
        
    return {
        "frame_length": stage6_res.get("frame_length_bits") or 0,
        "preamble_matches": stage6_res.get("preambles_found_count", 0),
        "hex_dump": flat_rows
    }

# =====================================================================
# Pre-Commit Extended Verification Test Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("STAGE 6: COMPLETE PRE-COMMIT VERIFICATION SUITE (8 TESTS)")
    print("=" * 65)

    np.random.seed(42)

    # -----------------------------------------------------------------
    # TEST 1, 2, 3: CCSDS Telemetry (512-bit frames)
    # -----------------------------------------------------------------
    ccsds_preamble = hex_to_bits("1ACFFC1D", 32)
    frame_len_1 = 512
    num_frames_1 = 12

    synthetic_bits_1 = []
    for _ in range(num_frames_1):
        hdr = np.random.randint(0, 2, size=16, dtype=np.uint8)
        payload = np.random.randint(0, 2, size=448, dtype=np.uint8)
        crc = np.random.randint(0, 2, size=16, dtype=np.uint8)
        synthetic_bits_1.extend(np.concatenate([ccsds_preamble, hdr, payload, crc]))

    test_stream_1 = np.array(synthetic_bits_1, dtype=np.uint8)
    res1 = run_stage_6(test_stream_1)

    print("\n[CHECK 1-3] CCSDS Telemetry Frame Detection:")
    print(f"  Estimated Frame Length:  {res1['frame_length_bits']} bits (Expected: {frame_len_1})")
    print(f"  Preamble Matches Count:  {res1['preambles_found_count']} (Expected: {num_frames_1})")
    assert res1["frame_length_bits"] == frame_len_1, "FAIL: Test 1 Frame length mismatch"
    assert res1["preambles_found_count"] == num_frames_1, "FAIL: Test 1 Preamble count mismatch"
    assert res1["preamble_matches"][0]["name"] == "CCSDS Telemetry", "FAIL: Test 1 Wrong preamble name"
    print("  --> PASS: CCSDS 32-bit periodic framing validated.")

    # -----------------------------------------------------------------
    # TEST 4: AX.25 / HDLC Framing (0x7E, 256-bit frames)
    # -----------------------------------------------------------------
    ax25_flag = hex_to_bits("7E", 8)
    frame_len_2 = 256
    num_frames_2 = 16

    synthetic_bits_2 = []
    for _ in range(num_frames_2):
        hdr = np.random.randint(0, 2, size=16, dtype=np.uint8)
        payload = np.random.randint(0, 2, size=216, dtype=np.uint8)
        fcs = np.random.randint(0, 2, size=16, dtype=np.uint8)
        synthetic_bits_2.extend(np.concatenate([ax25_flag, hdr, payload, fcs]))

    test_stream_2 = np.array(synthetic_bits_2, dtype=np.uint8)
    res2 = run_stage_6(test_stream_2)

    print("\n[CHECK 4] AX.25 / HDLC Flag Framing (0x7E):")
    print(f"  Estimated Frame Length:  {res2['frame_length_bits']} bits (Expected: {frame_len_2})")
    print(f"  Preamble Matches Count:  {res2['preambles_found_count']} (Expected: {num_frames_2})")
    assert res2["frame_length_bits"] == frame_len_2, "FAIL: Test 4 Frame length mismatch"
    assert res2["preambles_found_count"] == num_frames_2, "FAIL: Test 4 Preamble count mismatch"
    assert res2["preamble_matches"][0]["name"] == "AX.25 / HDLC Flag", "FAIL: Test 4 Wrong preamble name"
    print("  --> PASS: 8-bit AX.25 periodic flag correctly isolated and confirmed.")

    # -----------------------------------------------------------------
    # TEST 5: Fallback Path (Unframed Random Payload / Low-Entropy Header)
    # -----------------------------------------------------------------
    low_entropy_hdr_bytes = np.tile(np.array([0xAA, 0x55, 0x00, 0x00], dtype=np.uint8), 16)
    high_entropy_payload_bytes = np.random.randint(0, 256, size=512, dtype=np.uint8)
    unframed_bytes = np.concatenate([low_entropy_hdr_bytes, high_entropy_payload_bytes])
    unframed_bits = np.unpackbits(unframed_bytes)

    res3 = run_stage_6(unframed_bits)

    print("\n[CHECK 5] Fallback Path (Unframed / Unknown Stream):")
    print(f"  Preamble Matches Count:  {res3['preambles_found_count']} (Expected: 0)")
    assert res3["preambles_found_count"] == 0, "FAIL: Test 5 False positive sync word detected"

    hex_dump_3 = res3["hex_inspector"]["hex_dump"]
    row_0_labels = [r["label"] for r in hex_dump_3[0]["regions"]]
    row_last_labels = [r["label"] for r in hex_dump_3[-1]["regions"]]
    assert "header" in row_0_labels, "FAIL: Test 5 Entropy fallback failed to detect structured header"
    assert "payload" in row_last_labels, "FAIL: Test 5 High entropy region not classified as payload"
    print("  --> PASS: Shannon entropy fallback successfully segmented header and payload.")

    # -----------------------------------------------------------------
    # TEST 6: Non-Byte-Aligned Barker-11 Sequence (128-bit frame)
    # -----------------------------------------------------------------
    barker_11 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
    frame_len_6 = 128
    num_frames_6 = 20

    synthetic_bits_6 = []
    for _ in range(num_frames_6):
        # 11-bit Barker + 117-bit random payload = 128 bits
        payload = np.random.randint(0, 2, size=frame_len_6 - 11, dtype=np.uint8)
        synthetic_bits_6.extend(np.concatenate([barker_11, payload]))

    test_stream_6 = np.array(synthetic_bits_6, dtype=np.uint8)
    res6 = run_stage_6(test_stream_6)

    print("\n[CHECK 6] Configurable Barker-11 Non-Byte-Aligned Sync:")
    print(f"  Estimated Frame Length:  {res6['frame_length_bits']} bits (Expected: {frame_len_6})")
    print(f"  Preamble Matches Count:  {res6['preambles_found_count']} (Expected: {num_frames_6})")
    assert res6["frame_length_bits"] == frame_len_6, "FAIL: Test 6 Frame length mismatch"
    assert res6["preambles_found_count"] == num_frames_6, "FAIL: Test 6 Barker-11 count mismatch"
    assert res6["preamble_matches"][0]["name"] == "Barker-11", "FAIL: Test 6 Wrong preamble name"
    print("  --> PASS: Odd-length Barker-11 sequence identified across periodic stride.")

    # -----------------------------------------------------------------
    # TEST 7: Dynamic Correlator Hook (5G NR / LTE DMRS Reference Sequence)
    # -----------------------------------------------------------------
    # 32-bit Gold-code / pseudo-random reference vector
    dmrs_ref = np.random.randint(0, 2, size=32, dtype=np.uint8)
    frame_len_7 = 256
    num_frames_7 = 10

    synthetic_bits_7 = []
    for _ in range(num_frames_7):
        payload = np.random.randint(0, 2, size=frame_len_7 - 32, dtype=np.uint8)
        synthetic_bits_7.extend(np.concatenate([dmrs_ref, payload]))

    test_stream_7 = np.array(synthetic_bits_7, dtype=np.uint8)

    # Instantiate custom dynamic library entry passing the reference vector
    custom_dmrs_entry = {
        "name": "5G NR / LTE DMRS",
        "correlation_fn": lambda s: correlate_dmrs_placeholder(s, reference_seq=dmrs_ref),
        "bit_length": 32,
        "type": "dynamic"
    }
    custom_lib = [custom_dmrs_entry]
    res7 = run_stage_6(test_stream_7, preamble_lib=custom_lib)

    print("\n[CHECK 7] Dynamic Correlator Hook (5G DMRS):")
    print(f"  Estimated Frame Length:  {res7['frame_length_bits']} bits (Expected: {frame_len_7})")
    print(f"  Preamble Matches Count:  {res7['preambles_found_count']} (Expected: {num_frames_7})")
    assert res7["frame_length_bits"] == frame_len_7, "FAIL: Test 7 Frame length mismatch"
    assert res7["preambles_found_count"] == num_frames_7, "FAIL: Test 7 DMRS matches count mismatch"
    assert res7["preamble_matches"][0]["name"] == "5G NR / LTE DMRS", "FAIL: Test 7 Wrong preamble name"
    print("  --> PASS: Dynamic cross-correlation function hooked and verified.")

    # -----------------------------------------------------------------
    # TEST 8: Channel Noise Robustness (2% Bit Error Rate on CCSDS Stream)
    # -----------------------------------------------------------------
    noisy_stream = test_stream_1.copy()
    # Flip exactly 2% of the bits uniformly at random
    error_mask = np.random.rand(len(noisy_stream)) < 0.02
    noisy_stream[error_mask] ^= 1

    res8 = run_stage_6(noisy_stream)

    print("\n[CHECK 8] Channel Noise Robustness (2% BER on CCSDS Stream):")
    print(f"  True Frame Length:       {frame_len_1} bits")
    print(f"  Estimated Frame Length:  {res8['frame_length_bits']} bits")
    print(f"  Autocorr Z-Score:        {res8['frame_length_confidence_z']:.2f}")
    assert res8["frame_length_bits"] == frame_len_1, "FAIL: Test 8 Noise broke frame lock"
    assert res8["frame_length_confidence_z"] >= 3.0, "FAIL: Test 8 Insufficient confidence under noise"
    print("  --> PASS: Periodic autocorrelation retained frame lock despite 2% bit errors.")

    print("\n" + "=" * 65)
    print("ALL 8 VERIFICATION CHECKS PASSED — READY TO COMMIT")
    print("=" * 65)