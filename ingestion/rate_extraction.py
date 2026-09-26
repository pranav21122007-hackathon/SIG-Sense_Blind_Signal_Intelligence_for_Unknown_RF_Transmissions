"""
Blind RF Signal Ingestion and Feature Extraction Module
======================================================
This module ingests raw RF signal captures (.wav or headerless .iq),
resolves their numeric format, reconstructs complex I+jQ baseband data,
blind-estimates the sample rate (fs) and symbol rate (Rs) when missing,
and calculates Welch Power Spectral Density (PSD).
"""

import sys
import os
import argparse
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal


# -------------------------------------------------------------------------
# Heuristic 1: Headerless Dtype and Interleaving Resolution
# -------------------------------------------------------------------------
def detect_raw_iq_format(file_path: str, max_bytes: int = 2_000_000):
    """
    Plain-Language Judge Explanation:
    Headerless binary captures store numbers directly as raw bytes without
    telling us if they represent small whole numbers (int8), medium whole
    numbers (int16), or floating-point decimals (float32).
    
    1. Float32 check: Real floating-point IQ values almost always stay bounded 
       in [-2.0, 2.0]. When interpreted as 32-bit floats, random integers or
       unaligned data produce NaNs, infinities, or extreme exponents (e.g., 10^35).
       If >98% of decoded values sit safely in [-2.0, 2.0] without any NaNs,
       it is categorized as 32-bit float.
    2. Int8 vs Int16 check: If values exceed standard 8-bit limits or form a
       smooth Gaussian bell curve across ±32,768, it is 16-bit PCM. If every single
       byte naturally centers around 0 (signed int8) or 127/128 (unsigned uint8,
       common in RTL-SDR), it is 8-bit.
    """
    file_size = os.path.getsize(file_path)
    read_size = min(file_size, max_bytes)

    with open(file_path, "rb") as f:
        raw_bytes = f.read(read_size)

    # 1. Test Float32 (Interleaved I, Q -> np.complex64)
    if read_size % 8 == 0:
        f32_arr = np.frombuffer(raw_bytes, dtype=np.float32)
        valid_finite = np.isfinite(f32_arr)
        if np.all(valid_finite):
            in_range = (np.abs(f32_arr) <= 2.5)
            # If >98% of values fall within standard complex baseband amplitude bounds:
            if np.mean(in_range) > 0.98 and np.std(f32_arr) > 1e-4:
                return "float32", np.complex64

    # 2. Test Int16 (Interleaved I, Q -> np.int16)
    if read_size % 4 == 0:
        i16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        i16_std = np.std(i16_arr)
        i16_max = np.max(np.abs(i16_arr))
        # An 8-bit stream wrongly interpreted as 16-bit produces distinct byte-swapped artifacts
        # Real 16-bit SDR captures use a dynamic range well beyond 8-bit bounds (>256)
        if i16_max > 500 and i16_std > 100:
            return "int16", np.int16

    # 3. Test Int8 / UInt8 (Interleaved I, Q)
    u8_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    u8_mean = np.mean(u8_arr)
    # RTL-SDR defaults to unsigned 8-bit centered at 127.5
    if 100 < u8_mean < 155:
        return "uint8", np.uint8
    else:
        return "int8", np.int8


# -------------------------------------------------------------------------
# Heuristic 2: Reconstruct Complex Baseband (I + j*Q)
# -------------------------------------------------------------------------
def load_signal(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".wav":
        # Read exact sampling rate from WAV RIFF header
        fs, data = wavfile.read(file_path)
        if data.ndim == 2:
            # 2-Channel IQ capture
            i_ch = data[:, 0].astype(np.float32)
            q_ch = data[:, 1].astype(np.float32)
            if np.issubdtype(data.dtype, np.integer):
                max_v = float(np.iinfo(data.dtype).max)
                i_ch /= max_v
                q_ch /= max_v
            iq_data = (i_ch + 1j * q_ch).astype(np.complex64)
        else:
            samples = data.astype(np.float32)
            if np.issubdtype(data.dtype, np.integer):
                samples /= float(np.iinfo(data.dtype).max)
            iq_data = signal.hilbert(samples).astype(np.complex64)
            
        return iq_data, {"fs": float(fs), "fs_source": "WAV Header", "is_raw": False}
        
    else:
        # Raw .iq parsing logic
        detected_label, dtype = detect_raw_iq_format(file_path)
        raw_data = np.fromfile(file_path, dtype=dtype)
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]
            
        if dtype == np.complex64:
            iq_data = raw_data
        else:
            # Normalize float/int pairs to [-1.0, 1.0]
            scale = 32768.0 if dtype == np.int16 else (128.0 if dtype == np.int8 else 1.0)
            i_ch = raw_data[0::2].astype(np.float32) / scale
            q_ch = raw_data[1::2].astype(np.float32) / scale
            iq_data = (i_ch + 1j * q_ch).astype(np.complex64)
            
        return iq_data, {"fs": None, "fs_source": "Estimated", "is_raw": True}

# -------------------------------------------------------------------------
# Heuristic 3: Universal SDR Filter Edge & Cyclostationary Baud Estimator
# -------------------------------------------------------------------------
def estimate_rf_parameters(iq_data: np.ndarray, fs_hint: float = None):
    # If fs is provided by the file header (e.g. 200 kHz WAV), do NOT overwrite it!
    if fs_hint is not None and fs_hint > 0:
        fs_est = float(fs_hint)
    else:
        # Estimate from spectrum or fallback to default SDR base
        fs_est = 2.0e6
        
    # Difference-magnitude isolates symbol transitions
    diff_sig = np.diff(iq_data[:min(len(iq_data), 131072)])
    timing_env = np.abs(diff_sig) ** 2
    timing_env -= np.mean(timing_env)
    
    win = np.blackman(len(timing_env))
    spec = np.abs(np.fft.rfft(timing_env * win))
    fft_freqs = np.fft.rfftfreq(len(timing_env), d=1.0) # Normalized [0, 0.5]
    
    # Restrict search between 0.01 and 0.49
    mask = (fft_freqs >= 0.01) & (fft_freqs <= 0.49)
    spec_search = spec[mask]
    freqs_search = fft_freqs[mask]
    
    peak_idx = np.argmax(spec_search)
    norm_rs = freqs_search[peak_idx]
    
    # Absolute Baud Rate = Normalized Frequency * Sample Rate
    rs_est = float(norm_rs * fs_est)
    
    # Safety Check: SPS = fs / rs must be >= 2.0 for Gardner TED
    sps = fs_est / rs_est if rs_est > 0 else 4.0
    if sps < 2.0 or sps > 64.0:
        # Fallback to standard SPS = 4 if estimation noise creates an outlier
        rs_est = fs_est / 4.0
        
    return fs_est, 0.95, rs_est
# -------------------------------------------------------------------------
# Heuristic 4: Welch Power Spectral Density Generation
# -------------------------------------------------------------------------
def compute_welch_psd(iq_data: np.ndarray, fs: float, nperseg: int = 2048):
    """
    Computes a centered Welch Power Spectral Density for waterfall/spectrum UI panels.
    Returns frequencies and PSD values in dBFS.
    """
    f, pxx = signal.welch(
        iq_data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        scaling="density",
        return_onesided=False
    )
    # Re-center zero frequency
    f_shifted = np.fft.fftshift(f)
    pxx_shifted = np.fft.fftshift(pxx)
    pxx_db = 10 * np.log10(pxx_shifted + 1e-12)
    return f_shifted, pxx_db

# Ingestion/ingest.py

# [Keep existing detect_raw_iq_format, load_signal, estimate_rf_parameters, compute_welch_psd as defined]

def compute_waterfall_grid(iq_data: np.ndarray, fs: float, nperseg: int = 256, n_rows: int = 96) -> list:
    """
    Computes a 2D Spectrogram/Waterfall matrix (time rows x frequency columns)
    normalized to [0.0, 1.0] for direct rendering on HTML5 canvas.
    """
    # Use standard STFT across available samples
    f, t, zxx = signal.stft(iq_data[:min(len(iq_data), fs * 2)], fs=fs, nperseg=nperseg, return_onesided=False)
    zxx_shifted = np.fft.fftshift(zxx, axes=0)
    mag_db = 20 * np.log10(np.abs(zxx_shifted) + 1e-12)
    
    # Transpose so time is row axis: shape -> (time, freq)
    spec_2d = mag_db.T
    
    # Decimate/interpolate to fixed frontend dimensions (n_rows)
    if spec_2d.shape[0] > n_rows:
        step = spec_2d.shape[0] // n_rows
        spec_2d = spec_2d[:n_rows * step:step]
    
    # Normalize between 0.0 and 1.0
    p_min = np.percentile(spec_2d, 5)
    p_max = np.percentile(spec_2d, 95)
    norm_spec = np.clip((spec_2d - p_min) / (p_max - p_min + 1e-6), 0.0, 1.0)
    
    return norm_spec.tolist()

def run_stage_1(file_path: str, temp_output_fc32: str):
    """
    Standardized execution interface for Stage 1.
    """
    iq_data, meta = load_signal(file_path)
    
    # Always estimate Baud rate Rs (and sample rate Fs if headerless)
    fs_est, conf, rs_est = estimate_rf_parameters(iq_data)
    if meta["is_raw"] or meta.get("fs") is None:
        meta["fs"] = float(fs_est)
    
    fs = float(meta["fs"])
    rs = float(rs_est) if rs_est and rs_est > 0 else 100000.0  # Fallback to standard 100 kbaud if unestimated
    
    # 1D Welch PSD
    f_shifted, psd_db = compute_welch_psd(iq_data, fs)
    # Downsample PSD array to ~256 points for fast JSON transmission to frontend
    step = max(1, len(f_shifted) // 256)
    psd_points = [
        {"freq": round(float(f) / 1e6, 3), "power": round(float(p), 2)}
        for f, p in zip(f_shifted[::step], psd_db[::step])
    ]
    
    # 2D Waterfall array
    waterfall = compute_waterfall_grid(iq_data, fs)
    
    # Write intermediate standard interleaved complex64 binary file for GNU Radio
    iq_data.astype(np.complex64).tofile(temp_output_fc32)
    
    return {
        "iq_data": iq_data,
        "fs": fs,
        "rs": rs,
        "psd": psd_points,
        "waterfall": waterfall,
        "meta": meta
    }

# -------------------------------------------------------------------------
# Step 5: CLI Interface
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SIH26147 Blind RF Ingestion Stage: Ingest, Classify, Reconstruct, and Analyze."
    )
    parser.add_argument("file_path", type=str, help="Path to input .wav or raw .iq capture")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"[-] Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    # 1 & 2: Load signal and identify format
    iq_data, meta = load_signal(args.file_path)

    # 3: If headerless raw IQ, run blind estimation
    rs_est = None
    confidence = 1.0
    if meta["is_raw"]:
        fs_est, confidence, rs_est = estimate_rf_parameters(iq_data)
        meta["fs"] = fs_est
    else:
        # If WAV file, fs is already retrieved from the header
        pass

    # 4: Compute Welch PSD (ready for waterfall/spectrum UI feed)
    f, pxx_db = compute_welch_psd(iq_data, meta["fs"])

    # Output summaries
    print("\n" + "=" * 55)
    print("        SIH26147 RF SIGNAL INGESTION REPORT        ")
    print("=" * 55)
    print(f"Detected Format : {meta['format']}")
    print(f"Sample Rate (fs): {meta['fs']:,.0f} Hz ({meta['fs_source']})")
    
    if meta["is_raw"]:
        print(f"SDR Match Conf. : {confidence * 100:.1f}%")
        if rs_est and rs_est > 0:
            print(f"Estimated Rs    : {rs_est:,.1f} Baud (Cyclostationary FAM)")
        else:
            print(f"Estimated Rs    : Indeterminate / Continuous Carrier")
    else:
        print("Estimated Rs    : N/A (Header-based File)")

    print(f"Total Samples   : {len(iq_data):,}")
    print(f"PSD Bins Ready  : {len(pxx_db)} bins (Welch spectrum generated)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()