#!/usr/bin/env python3
"""
Blind Signal Synchronization Stage (SIH26147)
Supports single-file execution or batch-processing all files in testdata/.
Integrates with the Modulation Classifier stage via the -m / --mod argument.

Usage:
    # Single file test:
    python sync.py testdata/test1.iq -r 1000 -fs 100000 --plot

    # With modulation from your classifier stage:
    python sync.py testdata/test1.iq -r 1000 -fs 100000 -m BPSK

    # Auto-process ALL files in testdata folder:
    python sync.py --batch -r 1000 -fs 100000
"""

import argparse
import glob
import os
import sys
import numpy as np
from gnuradio import gr, blocks, digital


# Map modulation strings from classifier to Costas Loop order
MODULATION_MAP = {
    "BPSK": 2,
    "QPSK": 4,
    "4PSK": 4,
    "4QAM": 4,
    "8PSK": 8
}


class BlindSyncPipeline(gr.top_block):
    def __init__(self, input_file, samp_rate, symbol_rate, 
                 mod_order=4, snapshot_size=4096, 
                 raw_snap_file="raw_snapshot.npy", 
                 locked_snap_file="locked_snapshot.npy",
                 output_symbols_file="recovered_symbols.dat"):
        super(BlindSyncPipeline, self).__init__("BlindSyncPipeline")

        self.input_file = input_file
        self.samp_rate = float(samp_rate)
        self.symbol_rate = float(symbol_rate)
        self.sps = self.samp_rate / self.symbol_rate
        self.snapshot_size = snapshot_size
        self.raw_snap_file = raw_snap_file
        self.locked_snap_file = locked_snap_file
        self.output_symbols_file = output_symbols_file

        if self.sps < 2.0:
            print(f"[!] Warning: SPS = {self.sps:.2f}. Gardner TED requires SPS >= 2.0.", file=sys.stderr)

        # -------------------------------------------------------------
        # 1. Source: Raw complex64 IQ file (.iq)
        # -------------------------------------------------------------
        self.file_source = blocks.file_source(gr.sizeof_gr_complex, self.input_file, False)

        # -------------------------------------------------------------
        # 2. Before Snapshot (Raw Unsynced IQ)
        # -------------------------------------------------------------
        self.head_raw = blocks.head(gr.sizeof_gr_complex, self.snapshot_size)
        self.sink_raw = blocks.vector_sink_c()
        self.connect(self.file_source, self.head_raw, self.sink_raw)

        # -------------------------------------------------------------
        # 3. Timing Recovery: Gardner TED (sps -> 1 sps)
        # -------------------------------------------------------------
        loop_bw_timing = 0.045
        damping = 1.0
        max_dev = 0.15

        self.symbol_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            self.sps,
            loop_bw_timing,
            damping,
            1.0,
            max_dev,
            1,  # Decimate to exactly 1 SPS
            digital.constellation_qpsk().base()
        )

        # -------------------------------------------------------------
        # 4. Multipath Blind Equalizer: CMA (GNU Radio 3.10 API)
        # -------------------------------------------------------------
        num_taps = 15
        modulus = 1.0
        step_size = 0.0005
        sps_eq = 1

        try:
            # Create constellation object required by GNU Radio 3.10 CMA
            cma_constellation = digital.constellation_qpsk().base()
            
            # Correct 3-argument signature: (constellation, step_size, modulus)
            alg = digital.adaptive_algorithm_cma(cma_constellation, step_size, modulus)
            
            # Linear equalizer block applying the taps
            self.cma = digital.linear_equalizer(num_taps, sps_eq, alg, True)
        except Exception as e:
            # Fallback bypass if CMA fails to construct
            print(f"[!] CMA initialization skipped ({e}). Bypassing equalizer.", file=sys.stderr)
            self.cma = blocks.copy(gr.sizeof_gr_complex)
            
        # -------------------------------------------------------------
        # 5. Carrier Recovery: Costas Loop
        # -------------------------------------------------------------
        loop_bw_costas = 0.03
        self.costas = digital.costas_loop_cc(loop_bw_costas, mod_order, False)

        # -------------------------------------------------------------
        # 6. Outputs: Full recovered symbols + After Snapshot
        # -------------------------------------------------------------
        self.file_sink_out = blocks.file_sink(gr.sizeof_gr_complex, self.output_symbols_file, False)
        self.file_sink_out.set_unbuffered(False)

        self.head_locked = blocks.head(gr.sizeof_gr_complex, self.snapshot_size)
        self.sink_locked = blocks.vector_sink_c()

        # Connect graph
        self.connect(self.file_source, self.symbol_sync)
        self.connect(self.symbol_sync, self.cma)
        self.connect(self.cma, self.costas)
        self.connect(self.costas, self.file_sink_out)
        self.connect(self.costas, self.head_locked, self.sink_locked)

    def save_snapshots(self):
        raw = np.array(self.sink_raw.data(), dtype=np.complex64)
        locked = np.array(self.sink_locked.data(), dtype=np.complex64)

        np.save(self.raw_snap_file, raw)
        np.save(self.locked_snap_file, locked)

        print(f"  [+] Saved raw snapshot    : {self.raw_snap_file} ({len(raw)} samples)")
        print(f"  [+] Saved locked snapshot : {self.locked_snap_file} ({len(locked)} symbols)")
        print(f"  [+] Full recovered stream : {self.output_symbols_file}")
        return raw, locked


def plot_comparison(raw, locked, title_suffix=""):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Run 'pip install matplotlib' to plot.")
        return

    plt.figure(figsize=(10, 4.5))

    # Before
    plt.subplot(1, 2, 1)
    plt.scatter(raw.real, raw.imag, s=4, alpha=0.4, color="crimson")
    plt.title(f"Before: Raw Input {title_suffix}")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.grid(True)

    # After
    plt.subplot(1, 2, 2)
    plt.scatter(locked.real, locked.imag, s=4, alpha=0.4, color="navy")
    plt.title(f"After: Locked Constellation (1 SPS)")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def process_file(file_path, rs, fs, mod_str, plot=False):
    # Parse modulation to Costas loop order
    order = MODULATION_MAP.get(mod_str.upper(), 4)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Keep output filenames unique per test file
    raw_snap = f"{base_name}_raw.npy"
    locked_snap = f"{base_name}_locked.npy"
    out_symbols = f"{base_name}_symbols.dat"

    print(f"\n=======================================================")
    print(f"[*] Processing: {file_path}")
    print(f"[*] Modulation: {mod_str.upper()} (Costas Order: {order})")
    print(f"[*] Fs: {fs} Hz | Rs: {rs} Hz | SPS: {fs/rs:.2f}")
    print(f"=======================================================")

    pipeline = BlindSyncPipeline(
        input_file=file_path,
        samp_rate=fs,
        symbol_rate=rs,
        mod_order=order,
        raw_snap_file=raw_snap,
        locked_snap_file=locked_snap,
        output_symbols_file=out_symbols
    )

    pipeline.run()
    raw, locked = pipeline.save_snapshots()

    if plot:
        plot_comparison(raw, locked, title_suffix=f"({base_name} - {mod_str.upper()})")


def main():
    parser = argparse.ArgumentParser(description="GNU Radio Sync Stage for SIH26147")
    parser.add_argument("ip", nargs="?", default=None, help="Path to input .iq file (e.g. testdata/signal_1.iq)")
    parser.add_argument("-r", "--rs", type=float, required=True, help="Estimated Symbol Rate Rs (e.g., 1000)")
    parser.add_argument("-fs", "--samp-rate", type=float, default=2e6, help="Sampling rate in Hz (default: 2 MHz)")
    parser.add_argument("-m", "--mod", default="QPSK", help="Modulation type from classifier stage: BPSK, QPSK, 8PSK (default: QPSK)")
    parser.add_argument("--batch", action="store_true", help="Process ALL .iq files in testdata/ folder sequentially")
    parser.add_argument("--plot", action="store_true", help="Plot Before vs After constellations")

    args = parser.parse_args()

    if args.batch:
        iq_files = sorted(glob.glob("testdata/*.iq"))
        if not iq_files:
            print("[!] No .iq files found inside 'testdata/' folder.", file=sys.stderr)
            sys.exit(1)
        print(f"[*] Found {len(iq_files)} file(s) in testdata/. Running batch sync...")
        for f in iq_files:
            process_file(f, args.rs, args.samp_rate, args.mod, args.plot)
    else:
        if not args.ip:
            print("[!] Please provide a file path or use --batch. Example: python sync.py testdata/test1.iq -r 1000", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.ip):
            print(f"[!] File not found: {args.ip}", file=sys.stderr)
            sys.exit(1)
        process_file(args.ip, args.rs, args.samp_rate, args.mod, args.plot)


if __name__ == "__main__":
    main()