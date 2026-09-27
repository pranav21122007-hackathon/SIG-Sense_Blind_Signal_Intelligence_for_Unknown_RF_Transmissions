"""create_test_signals.py: Generates standard QPSK test vector"""
import numpy as np
import scipy.io.wavfile as wavfile

# 1. WAV
fs_wav = 48000
t = np.linspace(0, 1.0, fs_wav, endpoint=False)
wav_stereo = (np.vstack((np.sin(2*np.pi*1000*t), np.cos(2*np.pi*1000*t))).T * 32767).astype(np.int16)
wavfile.write("test_signal.wav", fs_wav, wav_stereo)

# 2. Raw IQ (fs = 2 MHz, Rs = 250 kHz -> 8 samples/symbol)
fs = 2000000.0
rs = 250000.0
sps = int(fs / rs)
n_symbols = 25000

# Random QPSK symbols
bits = np.random.choice([-1, 1], (n_symbols, 2))
syms = (bits[:, 0] + 1j * bits[:, 1]) / np.sqrt(2)

# Pulse shaping (Square-Root Raised Cosine)
upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
upsampled[::sps] = syms

# Simple raised-cosine pulse
t_pulse = np.arange(-4 * sps, 4 * sps + 1) / sps
pulse = np.sinc(t_pulse) * np.cos(np.pi * 0.35 * t_pulse) / (1 - (2 * 0.35 * t_pulse)**2 + 1e-8)
pulse /= np.sqrt(np.sum(pulse**2))

tx = np.convolve(upsampled, pulse, mode="same")
tx += (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx))) * 0.02

# Write interleaved float32
raw = np.empty(len(tx) * 2, dtype=np.float32)
raw[0::2] = tx.real
raw[1::2] = tx.imag
raw.tofile("test_signal.iq")
print("[+] Created test_signal.iq (fs=2.0 MHz, target Rs=250.0 kHz)")