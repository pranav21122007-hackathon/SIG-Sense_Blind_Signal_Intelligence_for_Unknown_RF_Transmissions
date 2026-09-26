import numpy as np
from commpy.channelcoding import Trellis, conv_encode, viterbi_decode
import reedsolo as rs


def identify_and_decode(bitstream):
    """
    Identifies and decodes either a convolutional code or a Reed-Solomon block code
    from a corrected/de-interleaved bitstream.
    """
    bits = np.asarray(bitstream, dtype=float)

    # 1. Evaluate Reed-Solomon hypothesis
    rs_result = _detect_and_decode_rs(bits)

    # 2. Evaluate Convolutional hypothesis
    conv_result = _detect_and_decode_convolutional(bits)

    # Print candidate scores for debugging telemetry
    # print(f"[Telemetry] RS Score: {rs_result['confidence']:.4f} | Conv Score: {conv_result['confidence']:.4f}")

    # Decision logic based on validated confidence
    if conv_result["confidence"] > rs_result["confidence"] and conv_result["confidence"] >= 0.5:
        return conv_result
    elif rs_result["confidence"] >= 0.5:
        return rs_result
    else:
        # Fallback to the higher non-zero confidence if above minimum threshold
        best = conv_result if conv_result["confidence"] >= rs_result["confidence"] else rs_result
        if best["confidence"] > 0.2:
            return best
        return {
            "code_type": "unknown",
            "params": None,
            "confidence": 0.0,
            "decoded_bits": np.round(np.clip(bits, 0, 1)).astype(int),
        }


def _detect_and_decode_convolutional(bitstream):
    best_candidate = {
        "code_type": "convolutional",
        "params": None,
        "confidence": 0.0,
        "decoded_bits": np.array([], dtype=int),
    }

    rates = {
        "1/2": (2, {7: np.array([[0o133, 0o171]], dtype=int), 9: np.array([[0o561, 0o753]], dtype=int)}),
        "1/3": (
            3,
            {
                7: np.array([[0o133, 0o145, 0o175]], dtype=int),
                9: np.array([[0o557, 0o663, 0o711]], dtype=int),
            },
        ),
    }

    hard_bits = np.round(np.clip(bitstream, 0, 1)).astype(int)

    for rate_str, (n_out, poly_dict) in rates.items():
        usable_len = (len(hard_bits) // n_out) * n_out
        if usable_len < 64:
            continue
        test_stream = hard_bits[:usable_len]

        for K, poly in poly_dict.items():
            try:
                trellis = Trellis(memory=np.array([K - 1]), g_matrix=poly)
                decoded = viterbi_decode(test_stream, trellis, decoding_type="hard")

                # Flush termination bits from trellis if present
                clean_decoded = decoded[: len(test_stream) // n_out]

                # VALIDATION: Re-encode and compute parity check / BER
                re_encoded = conv_encode(clean_decoded, trellis)
                compare_len = min(len(re_encoded), len(test_stream))
                mismatches = np.sum(re_encoded[:compare_len] != test_stream[:compare_len])
                ber = mismatches / compare_len

                # True convolutional streams will have BER close to 0 (< 0.15)
                # Random / incorrect parameters will have BER ~ 0.5
                if ber < 0.20:
                    confidence = float(np.clip(1.0 - (ber * 4.0), 0.0, 1.0))
                    if confidence > best_candidate["confidence"]:
                        best_candidate = {
                            "code_type": "convolutional",
                            "params": {"rate": rate_str, "constraint_length": K},
                            "confidence": confidence,
                            "decoded_bits": clean_decoded,
                        }
            except Exception:
                continue

    return best_candidate


def _detect_and_decode_rs(bitstream):
    best_candidate = {
        "code_type": "reed-solomon",
        "params": None,
        "confidence": 0.0,
        "decoded_bits": np.array([], dtype=int),
    }

    bits_int = np.round(np.clip(bitstream, 0, 1)).astype(int)
    total_bytes = len(bits_int) // 8
    if total_bytes < 32:
        return best_candidate

    symbols = _bits_to_symbols(bits_int[: total_bytes * 8], 8)
    byte_stream = bytes(symbols)

    # Standard RS parameters on GF(2^8)
    # n=255 is primary standard (CCSDS, DVB-S, etc.)
    candidate_n = [255]
    tested_nsyms = [16, 32, 8]

    for n in candidate_n:
        for nsym in tested_nsyms:
            if len(byte_stream) < n:
                continue

            try:
                rsc = rs.RSCodec(nsym)
            except Exception:
                continue

            k = n - nsym
            successful_blocks = 0
            total_blocks = len(byte_stream) // n

            decoded_bytes = bytearray()

            for i in range(total_blocks):
                block = byte_stream[i * n : (i + 1) * n]
                try:
                    # STRICT CHECK: Must pass syndrome verification without error
                    # If this is not RS, decode() will raise ReedSolomonError
                    res = rsc.decode(block)
                    decoded_msg = res[0] if isinstance(res, tuple) else res
                    decoded_bytes.extend(decoded_msg)
                    successful_blocks += 1
                except (rs.ReedSolomonError, Exception):
                    break

            if total_blocks > 0 and successful_blocks == total_blocks:
                confidence = 0.95  # Exact parity match across all blocks
                decoded_bits = _symbols_to_bits(list(decoded_bytes), 8)
                return {
                    "code_type": "reed-solomon",
                    "params": {"m": 8, "n": n, "k": k, "nsym": nsym},
                    "confidence": confidence,
                    "decoded_bits": decoded_bits,
                }

    return best_candidate


def _bits_to_symbols(bits, m):
    symbols = []
    for i in range(0, len(bits), m):
        chunk = bits[i : i + m]
        if len(chunk) < m:
            break
        val = 0
        for b in chunk:
            val = (val << 1) | int(b)
        symbols.append(val)
    return symbols


def _symbols_to_bits(symbols, m):
    bits = []
    for sym in symbols:
        for shift in range(m - 1, -1, -1):
            bits.append((sym >> shift) & 1)
    return np.array(bits, dtype=int)
if __name__ == '__main__':
    from commpy.channelcoding import Trellis, conv_encode
    import numpy as np
    import reedsolo as rs

    np.random.seed(101)

    print("=" * 65)
    print("TEST 3: RATE 1/3, K=7 CONVOLUTIONAL CODE (WITH NOISE)")
    print("=" * 65)

    # 1. Low-entropy test message
    info_text = b"SIH-TEST-RATE-1/3-K7"
    info_bits = np.unpackbits(np.frombuffer(info_text, dtype=np.uint8))

    # 2. Encode using Rate 1/3, K=7 polynomials: [0o133, 0o145, 0o175]
    poly_rate13 = np.array([[0o133, 0o145, 0o175]], dtype=int)
    trellis_13 = Trellis(memory=np.array([6]), g_matrix=poly_rate13)
    encoded_stream = conv_encode(info_bits, trellis_13)

    # 3. Add 2% random channel noise (bit flips)
    noise_mask = np.random.rand(len(encoded_stream)) < 0.02
    noisy_stream = np.bitwise_xor(encoded_stream, noise_mask.astype(int))

    print(f"Input Bitstream (first 64 bits):\n{noisy_stream[:64]}")
    print(f"Total Stream Length: {len(noisy_stream)} bits (Bit errors injected: {np.sum(noise_mask)})\n")

    result_conv3 = identify_and_decode(noisy_stream)
    print("Detected Result:")
    print(f"  Code Type   : {result_conv3['code_type']}")
    print(f"  Parameters  : {result_conv3['params']}")
    print(f"  Confidence  : {result_conv3['confidence']:.4f}")
    print(f"  Decoded Bits: {result_conv3['decoded_bits'][:32]}... (Total: {len(result_conv3['decoded_bits'])})")

    print("\n" + "=" * 65)
    print("TEST 4: REED-SOLOMON RS(255, 223), nsym=32")
    print("=" * 65)

    # 1. 223-byte low-entropy payload (CCSDS-like parameters)
    rs_codec_32 = rs.RSCodec(32)
    payload_msg = (b"DATA-BLOCK-" * 21)[:223]  # exactly 223 bytes
    rs_encoded_bytes = rs_codec_32.encode(payload_msg)

    # 2. Convert to binary bitstream
    rs_bitstream = np.unpackbits(np.frombuffer(rs_encoded_bytes, dtype=np.uint8))

    print(f"Input Bitstream (first 64 bits):\n{rs_bitstream[:64]}")
    print(f"Total Stream Length: {len(rs_bitstream)} bits (255 bytes)\n")

    result_rs32 = identify_and_decode(rs_bitstream)
    print("Detected Result:")
    print(f"  Code Type   : {result_rs32['code_type']}")
    print(f"  Parameters  : {result_rs32['params']}")
    print(f"  Confidence  : {result_rs32['confidence']:.4f}")
    print(f"  Decoded Bits: {result_rs32['decoded_bits'][:32]}... (Total: {len(result_rs32['decoded_bits'])})")